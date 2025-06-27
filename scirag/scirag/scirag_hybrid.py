from typing import List, Dict, Any
# from IPython.display import display, Markdown
import asyncio
import time
import os
import shutil
from pathlib import Path
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

from .config import AnswerFormat
import json

from google.genai.types import (
    GenerateContentConfig,
    Retrieval,
    Tool,
    VertexRagStore,
    VertexRagStoreRagResource,
)

from vertexai import rag
import faiss
import pickle

import numpy as np


from .config import (vertex_client,
                     credentials, 
                     GEMINI_EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP,
                     TOP_K, DISTANCE_THRESHOLD,
                     TEMPERATURE,
                     GEMINI_GEN_MODEL,
                     display_name,
                     folder_id,
                     markdown_files_path,
                     embeddings_path,
                     semantic_weight,
                     openai_client,
                     OpenAI_Embedding_Model)

from .scirag import SciRag, DocumentChunk
from tqdm.notebook import tqdm

# Google Cloud imports
from google.cloud import aiplatform
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel
import google.generativeai as genai
import tiktoken

# OpenAI imports - using client from config
import tiktoken

# LangChain imports for document processing and utilities
from langchain.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    CSVLoader,
    JSONLoader,
    UnstructuredWordDocumentLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.memory import ConversationBufferWindowMemory
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema.messages import BaseMessage
# Scikit-learn for TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
import glob
import json
import time


class SciRagHybrid(SciRag):
    def __init__(self, 
                 client = vertex_client,
                 credentials = credentials,
                 markdown_files_path = markdown_files_path,
                 corpus_name = display_name,
                 gen_model = GEMINI_GEN_MODEL,
                 vector_db_backend="chromadb",#"faiss",   # <--- add this line
                 chroma_collection_name="sci_rag_chunks",
                 chroma_db_path=str(embeddings_path / "chromadb"),
                 embedding_provider="gemini",  # New parameter: "gemini" or "openai"
                 openai_client=openai_client,  # Use client from config
                 openai_embedding_model=OpenAI_Embedding_Model,  # Use model from config
                 n_chunks = None,
                 ):
        super().__init__(client, credentials, markdown_files_path, corpus_name, gen_model)

        self.vector_db_backend = vector_db_backend
        self.chroma_collection_name = chroma_collection_name
        self.chroma_db_path = chroma_db_path
        self.chromadb_built = False
        
        # Embedding configuration
        self.embedding_provider = embedding_provider.lower()
        if self.embedding_provider not in ["gemini", "openai"]:
            raise ValueError("embedding_provider must be either 'gemini' or 'openai'")
        
        # Initialize embedding models based on provider
        if self.embedding_provider == "gemini":
            self.embedding_model = TextEmbeddingModel.from_pretrained(GEMINI_EMBEDDING_MODEL)
            self.embedding_model_name = GEMINI_EMBEDDING_MODEL
            self.embedding_dim = 768  # Gemini embedding dimension
        else:  # openai
            if openai_client is None:
                raise ValueError("openai_client must be provided when using OpenAI embeddings")
            self.openai_client = openai_client
            self.openai_embedding_model = openai_embedding_model
            self.embedding_model_name = openai_embedding_model
            # Set embedding dimension based on OpenAI model
            if "text-embedding-3-large" in openai_embedding_model:
                self.embedding_dim = 3072
            elif "text-embedding-3-small" in openai_embedding_model:
                self.embedding_dim = 1536
            elif "text-embedding-ada-002" in openai_embedding_model:
                self.embedding_dim = 1536
            else:
                self.embedding_dim = 1536  # Default fallback
        # Cost tracking configuration
        self._setup_cost_tracking()

        self.docs = self.load_markdown_files()
        self.split_documents() ## create self.all_chunks

        self.rate_limit_seconds = 0.2
        self.max_tokens_per_minute = 200000 if self.embedding_provider == "gemini" else 1000000  # OpenAI has higher limits

        self._token_bucket = 0
        self._bucket_start_time = 0

        # TF-IDF for lexical search
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=10000,
            ngram_range=(1, 2)
        )

        self.n_chunks = n_chunks

        self.chunks_store = []
        self.chunk_id_to_index = {}

        self._get_texts()

        self.rag_prompt = rf"""
You are a helpful assistant. Answer based on the provided context.
You must respond in valid JSON format with the following structure:

{{
  "answer": "your detailed answer here",
  "sources": ["source1", "source2", "source3"]
}}

The sources must be from the **Context** material provided.
Include source names, page numbers, equation numbers, table numbers, section numbers when available.
Ensure your response is valid JSON only.
"""

        self.enhanced_query = lambda context, query: (
rf"""
Question: {query}

Context:
{context}

Instructions: Based on the context provided above, answer the question in valid JSON format:
{{
  "answer": "your detailed answer here",
  "sources": ["source1", "source2"]
}}
"""
        )
        self._initialize_embeddings_and_vector_db()
    
    def _setup_cost_tracking(self):
        """Setup cost tracking for generation costs only"""
        # Pricing configuration (per 1K tokens) - Generation costs only
        self.pricing = {
            "gemini-2.5-flash-preview-05-20": {
                "input_price_per_1k": 0.00015,    # $0.15 per 1M tokens
                "output_price_per_1k": 0.0006,    # $0.60 per 1M tokens (no thinking)
                "output_price_per_1k_thinking": 0.0035,  # $3.50 per 1M tokens (with thinking)
            },
            "gemini-2.5-flash": {
                "input_price_per_1k": 0.000075,   # $0.075 per 1M tokens
                "output_price_per_1k": 0.0003,    # $0.30 per 1M tokens
            },
            "gemini-2.0-flash": {
                "input_price_per_1k": 0.0001,     # $0.10 per 1M tokens
                "output_price_per_1k": 0.0004,    # $0.40 per 1M tokens
            },
        }
        
        # Enhanced cost tracking
        self.cost_dict['Generation Model'] = []
        
        # Track total generation cost only
        self.total_generation_cost = 0.0
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken encoding."""
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception as e:
            print(f"Warning: Could not count tokens: {e}")
            # Fallback: rough estimate of 1 token per 4 characters
            return len(text) // 4
    def _calculate_generation_cost(self, input_tokens: int, output_tokens: int, thinking_enabled: bool = False) -> float:
        """Calculate generation cost for Gemini models"""
        model_pricing = self.pricing.get(self.gen_model, {})
        
        if not model_pricing:
            print(f"Warning: No pricing found for generation model {self.gen_model}")
            return 0.0
        
        # Calculate input cost
        input_cost = (input_tokens / 1000) * model_pricing.get("input_price_per_1k", 0.0)
        
        # Calculate output cost based on thinking mode
        if "output_price_per_1k" in model_pricing:
            output_cost = (output_tokens / 1000) * model_pricing["output_price_per_1k"]
        else:
            output_cost = (output_tokens / 1000) * model_pricing.get("output_price_per_1k", 0.0)
        
        return input_cost + output_cost
    def _log_cost_summary(self, generation_cost: float, input_tokens: int, output_tokens: int, ):
        """Log cost summary for generation only"""
        total_tokens = input_tokens + output_tokens
        
        # Update tracking
        self.cost_dict['Generation Model'].append(self.gen_model)
        
        # Update totals
        self.total_generation_cost += generation_cost
        
        # Update parent class tracking
        self.cost_dict['Cost'].append(generation_cost)
        self.cost_dict['Prompt Tokens'].append(input_tokens)
        self.cost_dict['Completion Tokens'].append(output_tokens)
        self.cost_dict['Total Tokens'].append(total_tokens)
        
        # Print detailed summary
        print(f"\n--- SciRagHybrid Cost Summary ---")
        print(f"Generation Model: {self.gen_model}")
        print(f"Input tokens: {input_tokens:,}")
        print(f"Output tokens: {output_tokens:,}")
        print(f"Total tokens: {total_tokens:,}")
        print(f"Generation cost: ${generation_cost:.6f}")
        print(f"Total session cost: ${self.get_total_cost():.6f}")
        print(f"--- End Summary ---\n")

    def _get_collection_name(self):
        """Generate consistent collection name based on embedding provider and model."""
        model_name_clean = self.embedding_model_name.replace('/', '_').replace('-', '_').replace(':', '_')
        return f"{self.chroma_collection_name}_{self.embedding_provider}_{model_name_clean}"
    def get_total_cost(self) -> float:
        """Get the total cost of all generation operations"""
        return self.total_generation_cost
    def get_cost_breakdown(self) -> dict:
        """Get detailed cost breakdown"""
        if not self.cost_dict['Cost']:
            return {
                "total_cost": 0.0,
                "total_calls": 0,
                "average_cost_per_call": 0.0,
                "generation_model": self.gen_model
            }
        
        return {
            "total_cost": self.get_total_cost(),
            "total_calls": len(self.cost_dict['Cost']),
            "average_cost_per_call": sum(self.cost_dict['Cost']) / len(self.cost_dict['Cost']),
            "generation_model": self.gen_model,
            "total_tokens": sum(self.cost_dict['Total Tokens']),
            "total_input_tokens": sum(self.cost_dict['Prompt Tokens']),
            "total_output_tokens": sum(self.cost_dict['Completion Tokens'])
        }

    def store_to_chromadb(self):
        import chromadb
        from chromadb.config import Settings
        
        # Ensure ChromaDB directory exists
        chroma_path = Path(self.chroma_db_path)
        chroma_path.mkdir(parents=True, exist_ok=True)
        
        # Reset any existing ChromaDB instance to avoid conflicts
        try:
            chromadb.reset()
        except:
            pass
            
        client = chromadb.PersistentClient(
            path=self.chroma_db_path, 
            settings=Settings(allow_reset=True, anonymized_telemetry=False)
        )
        
        # Create collection name using consistent method
        collection_name = self._get_collection_name()
        
        # Check if collection already exists
        try:
            existing_collection = client.get_collection(name=collection_name)
            print(f"Collection '{collection_name}' already exists with {existing_collection.count()} items")
            self.chroma_collection_name = collection_name
            return
        except:
            print(f"Creating new collection: {collection_name}")
        
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine", "embedding_provider": self.embedding_provider}
        )
        
        ids = [m.get('chunk_id', f'chunk_{i}') for i, m in enumerate(self.all_metadata)]
        embeddings = self.embeddings.tolist() if isinstance(self.embeddings, np.ndarray) else self.embeddings
        documents = self.all_texts
        metadatas = self.all_metadata
        
        batch_size = 1000
        for start in range(0, len(ids), batch_size):
            collection.add(
                ids=ids[start:start+batch_size],
                embeddings=embeddings[start:start+batch_size],
                documents=documents[start:start+batch_size],
                metadatas=metadatas[start:start+batch_size],
            )
        
        print(f"Stored {len(ids)} chunks in ChromaDB collection '{collection_name}' at {self.chroma_db_path}")
        self.chroma_collection_name = collection_name

    def load_chromadb_collection(self):
        import chromadb
        from chromadb.config import Settings
        
        # Ensure ChromaDB directory exists
        chroma_path = Path(self.chroma_db_path)
        chroma_path.mkdir(parents=True, exist_ok=True)
        
        # Reset any existing ChromaDB instance to avoid conflicts
        try:
            chromadb.reset()
        except:
            pass
            
        client = chromadb.PersistentClient(
            path=self.chroma_db_path,
            settings=Settings(allow_reset=True, anonymized_telemetry=False)
        )
        
        # Use consistent collection name generation
        collection_name = self._get_collection_name()
        
        try:
            self.chroma_collection = client.get_collection(name=collection_name)
            self.chroma_collection_name = collection_name
            print(f"Loaded existing ChromaDB collection: {collection_name}")
            return
        except Exception as e:
            print(f"Collection {collection_name} not found: {e}")
            
            # Try fallback to original collection name (for backward compatibility)
            try:
                original_name = self.chroma_collection_name
                self.chroma_collection = client.get_collection(name=original_name)
                print(f"Loaded fallback ChromaDB collection: {original_name}")
                return
            except Exception as e2:
                print(f"No existing collection found: {e2}")
                raise FileNotFoundError(f"No ChromaDB collection found. Will create new one.")

    def _initialize_embeddings_and_vector_db(self):
        """Initialize embeddings and vector database in the correct order."""
        print("Initializing embeddings and vector database...")

        embeddings_path.mkdir(parents=True, exist_ok=True)
        
        # Check if embeddings already exist
        embedding_filename = f'{self.embedding_provider}_{self.embedding_model_name.replace("/", "-").replace("-", "-").replace(":", "-")}_embeddings.npy'
        embedding_path = embeddings_path / embedding_filename
        print(f"Checking for existing embeddings at: {embedding_path}")
        
        if embedding_path.exists():
            print(f"Loading existing embeddings from: {embedding_path}")
            self.load_embeddings()
        else:
            print("Embeddings not found. Generating embeddings...")
            self.get_embeddings()
        
        # Now create the vector database
        print("Creating vector database...")
        self._create_vector_db_after_embeddings()

    def _create_vector_db_after_embeddings(self):
        """Create vector database after embeddings are ready."""
        if self.vector_db_backend == "faiss":
            self.faiss_index = self._create_faiss_index()
            self.index_built = True
            self._build_tf_idf_index()
            self.chunks_store = []
            for i in range(len(self.all_texts)):
                chunk_obj = DocumentChunk(
                    original_text=self.all_texts[i],
                    contextualized_text=None,
                    embedding=self.embeddings[i],
                    tfidf_vector=self.tfidf_matrix[i],
                    metadata=self.all_metadata[i],
                    chunk_id=self.all_metadata[i]['chunk_id']
                )
                self.chunks_store.append(chunk_obj)
                self.chunk_id_to_index[chunk_obj.chunk_id] = i
            print(f"FAISS index built successfully with {len(self.chunks_store)} chunks")
            self._save_index()
        elif self.vector_db_backend == "chromadb":
            try:
                # First try to load existing collection
                self.load_chromadb_collection()
                print(f"Loaded existing ChromaDB collection")
            except FileNotFoundError:
                print("Creating new ChromaDB collection...")
                self.store_to_chromadb()
                self.load_chromadb_collection()
                print(f"ChromaDB vector DB built successfully")
            self.chromadb_built = True
        else:
            raise ValueError(f"Unknown vector_db_backend: {self.vector_db_backend}")

    def query_chromadb(self, query, n_results=5):
        self._embed_query(query)
        query_embedding = self.query_embedding[0]
        results = self.chroma_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        # Flatten results to match your semantic_search interface
        return {
            "chunks": results["documents"][0],
            "metadata": results["metadatas"][0],
            "similarities": [1 - d for d in results["distances"][0]]  # Chroma returns L2, invert if using cosine
        }

    def _count_tokens(self, text: str) -> int:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

    def _get_texts(self):
        print("Building contextual retrieval index...")

        all_original_texts = []
        all_metadata = []

        # Group chunks by document for contextualization
        doc_groups = {}
        for chunk in self.all_chunks[:self.n_chunks]:
            source = chunk.metadata.get('source_file', chunk.metadata.get('source', 'unknown'))
            if source not in doc_groups:
                doc_groups[source] = []
            doc_groups[source].append(chunk)

        chunk_counter = 0

        # Process each document group
        for source, doc_chunks in doc_groups.items():
            # Reconstruct full document text for context
            full_doc_text = "\n\n".join([chunk.page_content for chunk in doc_chunks])
            
            # Contextualize each chunk
            for chunk in doc_chunks:
                all_original_texts.append(chunk.page_content)
                all_metadata.append({
                    **chunk.metadata,
                    'chunk_id': f"chunk_{chunk_counter}",
                })
                
                chunk_counter += 1
        print(f"Processed {chunk_counter} chunks")
        self.all_metadata = all_metadata
        self.all_texts = all_original_texts

    def _build_tf_idf_index(self):
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.all_texts)
        self.tfidf_fitted = True

    def _embed_texts_gemini(self) -> np.ndarray:
        """Generate embeddings using Google's embedding model."""
        embeddings = []
        texts = self.all_texts

        for i, text in enumerate(tqdm(texts, desc="Embedding texts (Gemini)", unit="doc")):
            tokens = self._count_tokens(text)
            now = time.time()
            
            # Reset token bucket every minute
            if now - self._bucket_start_time > 60 or self._bucket_start_time == 0:
                self._token_bucket = 0
                self._bucket_start_time = now

            # If adding this text would exceed the quota, wait for the next minute
            if self._token_bucket + tokens > self.max_tokens_per_minute:
                sleep_time = 60 - (now - self._bucket_start_time)
                if sleep_time > 0:
                    print(f"[TokenTracker] Token quota reached ({self._token_bucket} tokens). Sleeping for {sleep_time:.1f} seconds...")
                    time.sleep(sleep_time)
                self._token_bucket = 0
                self._bucket_start_time = time.time()

            while True:
                try:
                    batch_embeddings = self.embedding_model.get_embeddings([text])
                    for emb in batch_embeddings:
                        embeddings.append(emb.values)
                    break
                except Exception as e:
                    if "429" in str(e):
                        print("[TokenTracker] 429 error: Quota exceeded. Sleeping for 60 seconds before retrying...")
                        time.sleep(60)
                        self._token_bucket = 0
                        self._bucket_start_time = time.time()
                    else:
                        print(f"Error generating embeddings for text {i}: {e}")
                        embeddings.append([0.0] * self.embedding_dim)
                        break

            self._token_bucket += tokens
            time.sleep(self.rate_limit_seconds)

        return np.array(embeddings, dtype=np.float32)

    def _embed_texts_openai(self) -> np.ndarray:
        """Generate embeddings using OpenAI's embedding model."""
        embeddings = []
        texts = self.all_texts
        
        # OpenAI allows batch processing, so we can process in larger batches
        batch_size = 100
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            while True:
                try:
                    response = self.openai_client.embeddings.create(
                        input=batch_texts,
                        model=self.openai_embedding_model
                    )
                    
                    batch_embeddings = [item.embedding for item in response.data]
                    embeddings.extend(batch_embeddings)
                    
                    print(f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
                    break
                    
                except Exception as e:
                    if "rate_limit" in str(e).lower() or "429" in str(e):
                        print(f"Rate limit hit. Sleeping for 60 seconds...")
                        time.sleep(60)
                    else:
                        print(f"Error generating embeddings for batch {i//batch_size + 1}: {e}")
                        # Add zero embeddings for failed batch
                        batch_embeddings = [[0.0] * self.embedding_dim] * len(batch_texts)
                        embeddings.extend(batch_embeddings)
                        break
            
            time.sleep(self.rate_limit_seconds)

        return np.array(embeddings, dtype=np.float32)

    
    def _embed_texts(self) -> np.ndarray:
        """Generate embeddings using the configured embedding provider."""
        if self.embedding_provider == "gemini":
            self.embeddings = self._embed_texts_gemini()
        else:  # openai
            self.embeddings = self._embed_texts_openai()
        
        # Ensure embeddings directory exists
        embeddings_path.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings with provider-specific filename
        embedding_filename = f'{self.embedding_provider}_{self.embedding_model_name.replace("/", "_")}_embeddings.npy'
        np.save(embeddings_path / embedding_filename, self.embeddings)
        print(f"Embeddings saved to: {embeddings_path / embedding_filename}")

    def _embed_query_gemini(self, query: str) -> np.ndarray:
        """Generate query embedding using Gemini."""
        embeddings = []
        texts = [query]

        for i, text in enumerate(texts):
            tokens = self._count_tokens(text)
            now = time.time()
            
            if now - self._bucket_start_time > 60 or self._bucket_start_time == 0:
                self._token_bucket = 0
                self._bucket_start_time = now

            if self._token_bucket + tokens > self.max_tokens_per_minute:
                sleep_time = 60 - (now - self._bucket_start_time)
                if sleep_time > 0:
                    print(f"[TokenTracker] Token quota reached. Sleeping for {sleep_time:.1f} seconds...")
                    time.sleep(sleep_time)
                self._token_bucket = 0
                self._bucket_start_time = time.time()

            while True:
                try:
                    batch_embeddings = self.embedding_model.get_embeddings([text])
                    for emb in batch_embeddings:
                        embeddings.append(emb.values)
                    break
                except Exception as e:
                    if "429" in str(e):
                        print("[TokenTracker] 429 error: Quota exceeded. Sleeping for 60 seconds...")
                        time.sleep(60)
                        self._token_bucket = 0
                        self._bucket_start_time = time.time()
                    else:
                        print(f"Error generating embeddings for query: {e}")
                        embeddings.append([0.0] * self.embedding_dim)
                        break

            self._token_bucket += tokens
            time.sleep(self.rate_limit_seconds)

        return np.array(embeddings, dtype=np.float32)

    def _embed_query_openai(self, query: str) -> np.ndarray:
        """Generate query embedding using OpenAI."""
        while True:
            try:
                response = self.openai_client.embeddings.create(
                    input=[query],
                    model=self.openai_embedding_model
                )
                embedding = response.data[0].embedding
                return np.array([embedding], dtype=np.float32)
                
            except Exception as e:
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    print(f"Rate limit hit. Sleeping for 60 seconds...")
                    time.sleep(60)
                else:
                    print(f"Error generating query embedding: {e}")
                    return np.array([[0.0] * self.embedding_dim], dtype=np.float32)

    def _embed_query(self, query: str) -> np.ndarray:
        """Generate query embedding using the configured embedding provider."""
        if self.embedding_provider == "gemini":
            self.query_embedding = self._embed_query_gemini(query)
        else:  # openai
            self.query_embedding = self._embed_query_openai(query)

    def get_embeddings(self):
        self._embed_texts()

    def load_embeddings(self):
        embedding_filename = f'{self.embedding_provider}_{self.embedding_model_name.replace("/", "_")}_embeddings.npy'
        try:
            self.embeddings = np.load(embeddings_path / embedding_filename).astype(np.float32)
        except FileNotFoundError:
            raise FileNotFoundError(f"Embeddings file {embedding_filename} not found. You must call get_embeddings() first...")

    def _create_faiss_index(self, use_gpu: bool = False) -> faiss.Index:
        """Create and populate FAISS index."""
        d = self.embeddings.shape[1]
        
        if len(self.embeddings) > 1000:
            nlist = min(int(np.sqrt(len(self.embeddings))), 100)
            quantizer = faiss.IndexFlatIP(d)
            index = faiss.IndexIVFFlat(quantizer, d, nlist)
            print("Training FAISS index...")
            index.train(self.embeddings)
        else:
            index = faiss.IndexFlatIP(d)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(self.embeddings)
        
        print(f"Adding {len(self.embeddings)} vectors to FAISS index...")
        index.add(self.embeddings)
        
        if use_gpu and faiss.get_num_gpus() > 0:
            print("Moving FAISS index to GPU...")
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        
        return index

    

    def _save_index(self):
        """Save the index to disk."""
        model_prefix = f'{self.embedding_provider}_{self.embedding_model_name.replace("/", "_")}'
        
        if self.faiss_index is not None:
            faiss.write_index(self.faiss_index, str(embeddings_path/f'{model_prefix}_faiss.index'))
        else:
            print("Index not built. Call create_vector_db method first.")
        
        index_state = {
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'tfidf_fitted': self.tfidf_fitted,
            'chunks_count': len(self.chunks_store),
            'embedding_dim': self.embedding_dim,
            'chunk_id_to_index': self.chunk_id_to_index,
            'index_built': self.index_built,
            'embedding_provider': self.embedding_provider,
            'embedding_model_name': self.embedding_model_name
        }
        
        with open(embeddings_path/f'{model_prefix}_index_state.pkl', 'wb') as f:
            pickle.dump(index_state, f)
        
        # Save chunks metadata
        chunks_metadata = []
        for chunk in self.chunks_store:
            chunks_metadata.append({
                'original_text': chunk.original_text,
                'contextualized_text': chunk.contextualized_text,
                'metadata': chunk.metadata,
                'chunk_id': chunk.chunk_id
            })
        
        with open(embeddings_path/f'{model_prefix}_chunks_metadata.json', 'w') as f:
            json.dump(chunks_metadata, f, indent=2)

    def semantic_search(self, query: str, n_results: int = 20) -> Dict[str, Any]:
        if self.vector_db_backend == "faiss":
            if not self.index_built:
                raise ValueError("Index not built. Call build_index_* method first.")
            self._embed_query(query)
            query_embedding = self.query_embedding
            faiss.normalize_L2(query_embedding)
            similarities, indices = self.faiss_index.search(query_embedding, n_results)
            results = {'chunks': [], 'metadata': [], 'similarities': similarities[0].tolist()}
            for idx in indices[0]:
                if idx != -1:
                    chunk = self.chunks_store[idx]
                    results['chunks'].append(chunk.original_text)
                    results['metadata'].append(chunk.metadata)
            return results
        elif self.vector_db_backend == "chromadb":
            return self.query_chromadb(query, n_results)
        else:
            raise ValueError(f"Unknown vector_db_backend: {self.vector_db_backend}")

    def lexical_search(self, query: str, n_results: int = 20) -> List[Dict]:
        """Perform lexical search using TF-IDF."""
        if not self.tfidf_fitted:
            raise ValueError("TF-IDF not fitted. Build index first.")
        
        query_vector = self.tfidf_vectorizer.transform([query])
        
        similarities = []
        for i, chunk in enumerate(self.chunks_store):
            similarity = (query_vector * chunk.tfidf_vector.T).toarray()[0][0]
            similarities.append((similarity, i))
        
        similarities.sort(reverse=True, key=lambda x: x[0])
        
        results = []
        for similarity, idx in similarities[:n_results]:
            chunk = self.chunks_store[idx]
            results.append({
                'text': chunk.original_text,
                'metadata': chunk.metadata,
                'similarity': similarity
            })
        
        return results

    def hybrid_search(self, query: str, n_results: int = TOP_K, 
                     semantic_weight: float = semantic_weight) -> List[Dict]:
        """Combine semantic and lexical search."""
        semantic_results = self.semantic_search(query, n_results)
        
        combined_scores = {}
        
        # Add semantic scores
        for chunk, metadata, similarity in zip(
            semantic_results['chunks'],
            semantic_results['metadata'], 
            semantic_results['similarities']
        ):
            chunk_id = metadata['chunk_id']
            combined_scores[chunk_id] = {
                'semantic_score': similarity * semantic_weight,
                'lexical_score': 0,
                'text': chunk,
                'metadata': metadata
            }

        # Calculate final scores
        final_results = []
        for chunk_id, scores in combined_scores.items():
            final_score = scores['semantic_score'] + scores['lexical_score']
            final_results.append({
                'text': scores['text'],
                'metadata': scores['metadata'],
                'final_score': final_score,
                'semantic_score': scores['semantic_score'],
                'lexical_score': 0.,
            })
        
        final_results.sort(key=lambda x: x['final_score'], reverse=True)
        return final_results[:n_results]

    def get_chunks(self, query: str):
        pass
    
    def delete_vector_db(self):
        pass

    def get_response(self, query: str):
        contexts = self.hybrid_search(query)
        # Prepare context text
        context_pieces = []
        for i, ctx in enumerate(contexts, 1):
            source = ctx['metadata'].get('file_name', ctx['metadata'].get('file_name', 'Unknown'))
            context_pieces.append(f"[Context {i} - Source: {source}]\n{ctx['text']}\n")
        
        context_text = "\n".join(context_pieces)
        self.context_text = context_text
        content = self.enhanced_query(context_text, query)
        # Count input tokens
        input_tokens = self._count_tokens(content)
        response = self.client.models.generate_content(
            model=self.gen_model,
            contents=content,
            config=GenerateContentConfig(
                temperature=TEMPERATURE,
                system_instruction=self.rag_prompt,
                response_mime_type='application/json',
                response_schema=AnswerFormat,
            ),
        )
        output_tokens = self._count_tokens(response.text)
        # Calculate and log cost
        generation_cost = self._calculate_generation_cost(input_tokens, output_tokens)
        self._log_cost_summary(generation_cost, input_tokens, output_tokens)
        
        return self.format_agent_output(response.text)
    
    def format_agent_output(self, response):
        """Format agent output with robust JSON parsing and fallback handling."""
        try:
            # Try to parse as JSON first
            parsed = json.loads(response)
            answer = parsed.get("answer") or parsed.get("Answer") or ""
            sources = parsed.get("sources") or parsed.get("Sources") or []
            if isinstance(sources, list):
                sources_str = ", ".join(sources)
            else:
                sources_str = str(sources)
            return f"""**Answer**:

{answer}

**Sources**:

{sources_str}
"""
        except json.JSONDecodeError as e:
            print(f"JSON parsing failed: {e}")
            print(f"Raw response: {response[:200]}...")
            
            # Fallback: try to extract answer and sources from markdown-like format
            if "**Answer**:" in response and "**Sources**:" in response:
                parts = response.split("**Sources**:")
                answer = parts[0].replace("**Answer**:", "").strip()
                sources = parts[1].strip() if len(parts) > 1 else ""
                return f"""**Answer**:

{answer}

**Sources**:

{sources}
"""
            else:
                # Last resort: return the raw response
                return f"""**Answer**:

{response}

**Sources**:

Unable to parse sources from response
"""