import os
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import faiss
import pickle
import re
from pathlib import Path

# Google Cloud imports
from google.cloud import aiplatform
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel
import google.generativeai as genai
import tiktoken
import json

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

@dataclass
class DocumentChunk:
    """Represents a document chunk with context and metadata"""
    original_text: str
    contextualized_text: str
    embedding: np.ndarray
    tfidf_vector: Any
    metadata: Dict[str, Any]
    chunk_id: str

def load_samples_from_jsonl(jsonl_path, n=None):
    """
    Load all questions, ideal answers, citations, key_passages, and source_files from a JSONL file.
    Args:
        jsonl_path: Path to the JSONL file
        n: Number of samples to load. If None, load all samples.
    Returns:
        A list of dictionaries containing the questions, ideal answers, citations, key_passages, and source_files.
    """
    samples = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            sample = json.loads(line)
            samples.append(sample)
            if n is not None and len(samples) >= n:
                break

class CustomCallbackHandler(BaseCallbackHandler):
    """Custom callback handler for monitoring LangChain operations"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.logs = []
    
    def on_text(self, text: str, **kwargs) -> None:
        if self.verbose:
            print(f"Generated text: {text[:100]}...")
        self.logs.append({"type": "text", "content": text})
    
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs) -> None:
        if self.verbose:
            print(f"Chain started: {serialized.get('name', 'Unknown')}")
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        if self.verbose:
            print("Chain completed")


class HybridContextualRAG:
    """
    Hybrid system combining LangChain utilities with custom contextual retrieval.
    Uses LangChain for document processing and custom implementation for core RAG.
    """
        
    def __init__(self, project_id: str, location: str = "europe-west2", 
                 embedding_dim: int = 768, verbose: bool = False,
                 rate_limit_seconds: float = 0.2,
                 max_tokens_per_minute: int = 200000, contextual_RAG: bool = False):
        """Initialize the hybrid contextual retrieval system."""
        
        # Initialize Vertex AI
        aiplatform.init(project=project_id, location=location)
        
        # Core models
        self.embedding_model = TextEmbeddingModel.from_pretrained("gemini-embedding-001")
        self.context_model = GenerativeModel("gemini-2.0-flash")
        
        # LangChain components
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.callback_handler = CustomCallbackHandler(verbose=verbose)
        
        # Custom FAISS components
        self.embedding_dim = embedding_dim
        self.faiss_index = None
        self.index_built = False
        
        # TF-IDF for lexical search
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=10000,
            ngram_range=(1, 2)
        )
        self.tfidf_fitted = False
        
        # Storage
        self.chunks_store = []
        self.chunk_id_to_index = {}
        self.document_metadata = {}
        
        self.verbose = verbose
        self.rate_limit_seconds = rate_limit_seconds
        self.max_tokens_per_minute = max_tokens_per_minute
        self._token_bucket = 0
        self._bucket_start_time = time.time()
        self.contextual_RAG = contextual_RAG

    def load_documents_from_directory(self, directory_path: str, 
                                    file_types: List[str] = None) -> List[Document]:
        """Load documents from directory using LangChain loaders."""
        if file_types is None:
            file_types = ['.txt', '.pdf', '.docx', '.csv', '.json']
        
        documents = []
        directory = Path(directory_path)
        
        if not directory.exists():
            raise ValueError(f"Directory {directory_path} does not exist")
        
        print(f"Loading documents from {directory_path}...")
        
        for file_path in directory.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in file_types:
                try:
                    # Choose appropriate loader based on file type
                    if file_path.suffix.lower() == '.pdf':
                        loader = PyPDFLoader(str(file_path))
                    elif file_path.suffix.lower() == '.csv':
                        loader = CSVLoader(str(file_path))
                    elif file_path.suffix.lower() == '.json':
                        loader = JSONLoader(str(file_path), jq_schema='.', text_key='text')
                    elif file_path.suffix.lower() in ['.doc', '.docx']:
                        loader = UnstructuredWordDocumentLoader(str(file_path))
                    else:  # Default to text loader
                        loader = TextLoader(str(file_path), encoding='utf-8')
                    
                    file_docs = loader.load()
                    
                    # Add file metadata
                    for doc in file_docs:
                        doc.metadata.update({
                            'source_file': str(file_path),
                            'file_type': file_path.suffix.lower(),
                            'file_name': file_path.name
                        })
                    
                    documents.extend(file_docs)
                    print(f"  Loaded {len(file_docs)} documents from {file_path.name}")
                    
                except Exception as e:
                    print(f"  Error loading {file_path}: {e}")
        
        print(f"Total documents loaded: {len(documents)}")
        return documents

    def load_documents_from_files(self, file_paths: List[str]) -> List[Document]:
        """
        Load specific .txt files using LangChain's TextLoader.
        """
        documents = []
        for file_path in file_paths:
            path = Path(file_path)
            if not path.exists() or not path.is_file() or path.suffix.lower() != '.txt':
                print(f"Skipping non-existent or non-txt file: {file_path}")
                continue
            try:
                loader = TextLoader(str(file_path), encoding='utf-8')
                file_docs = loader.load()
                for doc in file_docs:
                    doc.metadata.update({
                        'source_file': str(file_path),
                        'file_type': path.suffix.lower(),
                        'file_name': path.name
                    })
                documents.extend(file_docs)
                print(f"  Loaded {len(file_docs)} documents from {path.name}")
            except Exception as e:
                print(f"  Error loading {file_path}: {e}")
        print(f"Total .txt documents loaded: {len(documents)}")
        return documents
    

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks using LangChain text splitter."""
        print("Splitting documents into chunks...")
        
        all_chunks = []
        for doc in documents:
            chunks = self.text_splitter.split_documents([doc])
            
            # Add chunk metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'chunk_id': f"{doc.metadata.get('file_name', 'unknown')}_{i}"
                })
            
            all_chunks.extend(chunks)
        
        print(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks

    def _generate_context_prompt(self, document_text: str, chunk_text: str) -> str:
        """Generate the prompt for contextualizing a chunk (Anthropic's approach)."""
        return f"""<document>
{document_text}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk_text}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""

    def _contextualize_chunk(self, document_text: str, chunk_text: str) -> str:
        """Add context to a chunk using the context model."""
        try:
            prompt = self._generate_context_prompt(document_text, chunk_text)
            response = self.context_model.generate_content(
                prompt,
                generation_config={'temperature': 0.1,
                                   'max_output_tokens': 200}
            )
            
            context = response.text.strip()
            contextualized_chunk = f"{context} {chunk_text}"
            
            if self.verbose:
                print(f"Original: {chunk_text[:100]}...")
                print(f"Contextualized: {contextualized_chunk[:150]}...")
                print("---")
            
            return contextualized_chunk
            
        except Exception as e:
            print(f"Error contextualizing chunk: {e}")
            return chunk_text

    def _count_tokens(self, text: str) -> int:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using Google's embedding model, with token-based rate limiting and robust 429 error handling."""
        embeddings = []
        for i, text in enumerate(texts):
            tokens = self._count_tokens(text)
            now = time.time()
            # Reset token bucket every minute
            if now - self._bucket_start_time > 60:
                if self.verbose:
                    print(f"[TokenTracker] Resetting token bucket. Sent {self._token_bucket} tokens in the last minute.")
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
                    break  # Success, break out of retry loop
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
            if self.verbose:
                print(f"[TokenTracker] Sent {tokens} tokens (total this minute: {self._token_bucket}/{self.max_tokens_per_minute})")
            time.sleep(self.rate_limit_seconds)  # Still respect per-request delay

        return np.array(embeddings, dtype=np.float32)

    def _create_faiss_index(self, embeddings: np.ndarray, use_gpu: bool = False) -> faiss.Index:
        """Create and populate FAISS index."""
        d = embeddings.shape[1]
        
        if len(embeddings) > 1000:
            nlist = min(int(np.sqrt(len(embeddings))), 100)
            quantizer = faiss.IndexFlatIP(d)
            index = faiss.IndexIVFFlat(quantizer, d, nlist)
            print("Training FAISS index...")
            index.train(embeddings)
        else:
            index = faiss.IndexFlatIP(d)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        print(f"Adding {len(embeddings)} vectors to FAISS index...")
        index.add(embeddings)
        
        if use_gpu and faiss.get_num_gpus() > 0:
            print("Moving FAISS index to GPU...")
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        
        return index

    def build_index_from_directory(self, directory_path: str, file_types: List[str] = None,
                                 use_gpu: bool = False) -> None:
        """Build index from a directory of documents."""
        # Step 1: Load documents using LangChain
        documents = self.load_documents_from_directory(directory_path, file_types)
        
        # Step 2: Split documents using LangChain
        chunks = self.split_documents(documents)
        
        # Step 3: Build index using custom contextual approach
        self._build_contextual_index(chunks, use_gpu)

    def build_index_from_files(self, file_paths: List[str], use_gpu: bool = False) -> None:
        """Build index from specific files."""
        # Step 1: Load documents using LangChain
        documents = self.load_documents_from_files(file_paths)
        
        # Step 2: Split documents using LangChain
        chunks = self.split_documents(documents)
        
        # Step 3: Build index using custom contextual approach
        self._build_contextual_index(chunks, use_gpu)

    def build_index_from_texts(self, texts: List[Dict[str, Any]], use_gpu: bool = False) -> None:
        """Build index from raw text data."""
        # Convert to LangChain Document format
        documents = []
        for i, text_data in enumerate(texts):
            doc = Document(
                page_content=text_data['text'],
                metadata={
                    'title': text_data.get('title', f'Document_{i}'),
                    'source': text_data.get('source', f'text_{i}'),
                    **text_data.get('metadata', {})
                }
            )
            documents.append(doc)
        
        # Split and build index
        chunks = self.split_documents(documents)
        self._build_contextual_index(chunks, use_gpu)

    def _build_contextual_index(self, chunks: List[Document], use_gpu: bool = False) -> None:
        """Core method to build contextual index from document chunks."""
        print("Building contextual retrieval index...")
        
        all_original_texts = []
        all_contextualized_texts = []
        all_metadata = []
        
        # Group chunks by document for contextualization
        doc_groups = {}
        for chunk in chunks:
            source = chunk.metadata.get('source_file', chunk.metadata.get('source', 'unknown'))
            if source not in doc_groups:
                doc_groups[source] = []
            doc_groups[source].append(chunk)
        
        chunk_counter = 0
        
        # Process each document group
        for source, doc_chunks in doc_groups.items():
            print(f"Processing {len(doc_chunks)} chunks from {source}")
            
            # Reconstruct full document text for context
            full_doc_text = "\n\n".join([chunk.page_content for chunk in doc_chunks])
            
            # Contextualize each chunk
            for chunk in doc_chunks:
                if self.verbose:
                    print(f"  Contextualizing chunk {chunk_counter + 1}")
                
                # Generate contextual version based on the contextual_RAG flag
                if self.contextual_RAG:
                    contextualized_text = self._contextualize_chunk(full_doc_text, chunk.page_content)
                else:
                    contextualized_text = chunk.page_content
                # Store both versions
                all_original_texts.append(chunk.page_content)
                all_contextualized_texts.append(contextualized_text)
                all_metadata.append({
                    **chunk.metadata,
                    'chunk_id': f"chunk_{chunk_counter}",
                    'contextualized': False
                })
                
                chunk_counter += 1
        
        print(f"Generated context for {chunk_counter} chunks")
        
        # Generate embeddings for contextualized texts
        print("Generating embeddings...")
        embeddings = self._embed_texts(all_contextualized_texts)
        
        # Create FAISS index
        print("Building FAISS index...")
        self.faiss_index = self._create_faiss_index(embeddings, use_gpu)
        self.index_built = True
        
        # Build TF-IDF index
        print("Building TF-IDF index...")
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_contextualized_texts)
        self.tfidf_fitted = True
        
        # Store all chunks
        self.chunks_store = []
        for i in range(len(all_original_texts)):
            chunk_obj = DocumentChunk(
                original_text=all_original_texts[i],
                contextualized_text=all_contextualized_texts[i],
                embedding=embeddings[i],
                tfidf_vector=tfidf_matrix[i],
                metadata=all_metadata[i],
                chunk_id=all_metadata[i]['chunk_id']
            )
            self.chunks_store.append(chunk_obj)
            self.chunk_id_to_index[chunk_obj.chunk_id] = i
        
        print(f"Index built successfully with {len(self.chunks_store)} chunks")
        self._save_index()

    def semantic_search(self, query: str, n_results: int = 20) -> Dict[str, Any]:
        """Perform semantic search using FAISS."""
        if not self.index_built:
            raise ValueError("Index not built. Call build_index_* method first.")
        
        query_embedding = self._embed_texts([query])
        faiss.normalize_L2(query_embedding)
        
        similarities, indices = self.faiss_index.search(query_embedding, n_results)
        
        results = {'chunks': [], 'metadata': [], 'similarities': similarities[0].tolist()}
        
        for idx in indices[0]:
            if idx != -1:
                chunk = self.chunks_store[idx]
                results['chunks'].append(chunk.contextualized_text)
                results['metadata'].append(chunk.metadata)
        
        return results

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
                'text': chunk.contextualized_text,
                'metadata': chunk.metadata,
                'similarity': similarity
            })
        
        return results

    def hybrid_search(self, query: str, n_results: int = 20, 
                     semantic_weight: float = 0.7) -> List[Dict]:
        """Combine semantic and lexical search."""
        semantic_results = self.semantic_search(query, n_results * 2)
        lexical_results = self.lexical_search(query, n_results * 2)
        
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
        
        # Add lexical scores
        for result in lexical_results:
            chunk_id = result['metadata']['chunk_id']
            lexical_score = result['similarity'] * (1 - semantic_weight)
            
            if chunk_id in combined_scores:
                combined_scores[chunk_id]['lexical_score'] = lexical_score
            else:
                combined_scores[chunk_id] = {
                    'semantic_score': 0,
                    'lexical_score': lexical_score,
                    'text': result['text'],
                    'metadata': result['metadata']
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
                'lexical_score': scores['lexical_score']
            })
        
        final_results.sort(key=lambda x: x['final_score'], reverse=True)
        return final_results[:n_results]

    def answer_question(self, question: str, n_contexts: int = 10, 
                       temperature: float = 0.1) -> Dict[str, Any]:
        """Answer a question using retrieved contexts."""
        if not self.index_built:
            raise ValueError("Index not built. Build index first.")
        
        # Retrieve contexts
        contexts = self.hybrid_search(question, n_results=n_contexts)
        
        # Prepare context text
        context_pieces = []
        for i, ctx in enumerate(contexts, 1):
            source = ctx['metadata'].get('file_name', ctx['metadata'].get('source', 'Unknown'))
            context_pieces.append(f"[Context {i} - Source: {source}]\n{ctx['text']}\n")
        
        context_text = "\n".join(context_pieces)
        
        # Generate answer
        prompt = f"""You are a helpful AI assistant that answers questions based on the provided context. Follow these guidelines:

        1. Answer the question directly using ONLY the information provided in the context
        2. If the context doesn't contain enough information, say so clearly
        3. Cite specific sources when making claims (e.g., "According to source: {source}\n{ctx['text']}...")
        4. Be precise and factual - don't add information not present in the context

        Context:
        {context_text}

        Question: {question}

        Answer:"""

        try:
            response = self.context_model.generate_content(
                prompt,
                generation_config={
                    'temperature': temperature,
                    'top_p': 0.8
                }
            )
            
            answer = response.text.strip()
            # confidence = self._estimate_confidence(answer, context_text)
            
            # Update conversation memory
            self.memory.save_context(
                {"input": question},
                {"output": answer}
            )
            
            return {
                'answer': answer,
                # 'confidence': confidence,
                'sources': [
                    {
                        'text': ctx['text'][:200] + "..." if len(ctx['text']) > 200 else ctx['text'],
                        'source': ctx['metadata'].get('file_name', 'Unknown'),
                        'score': ctx['final_score']
                    }
                    for ctx in contexts
                ],
                'num_contexts_used': len(contexts)
            }
            
        except Exception as e:
            return {
                'question': question,
                'answer': f"Error generating answer: {str(e)}",
                'confidence': 0.0,
                'sources': [],
                'error': True
            }

    # def _estimate_confidence(self, answer: str, context: str) -> float:
    #     """Simple confidence estimation."""
    #     confidence = 0.5
        
    #     if "Context" in answer or "According to" in answer:
    #         confidence += 0.2
    #     if len(answer) > 100:
    #         confidence += 0.1
        
    #     uncertainty_phrases = ["I don't know", "not enough information", "unclear"]
    #     if any(phrase in answer.lower() for phrase in uncertainty_phrases):
    #         confidence -= 0.3
    #     if len(answer) < 50:
    #         confidence -= 0.2
        
    #     return max(0.0, min(1.0, confidence))

    def conversational_chat(self) -> None:
        """Interactive chat with conversation memory."""
        print("🤖 Hybrid Contextual RAG Assistant")
        print("Ask me questions about your documents. Type 'quit' to exit.")
        print("Type 'memory' to see conversation history.")
        print("=" * 60)
        
        while True:
            try:
                question = input("\n❓ Your question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Goodbye!")
                    break
                
                if question.lower() == 'memory':
                    history = self.memory.load_memory_variables({})
                    print("\n💭 Conversation History:")
                    for msg in history.get('chat_history', []):
                        if hasattr(msg, 'content'):
                            print(f"  {type(msg).__name__}: {msg.content}")
                    continue
                
                if not question:
                    continue
                
                # Get answer with conversation context
                chat_history = self.memory.load_memory_variables({})
                
                result = self.answer_question(question, n_contexts=5)
                
                print(f"\n🎯 Answer:")
                print(f"{result['answer']}")
                # print(f"\n📊 Confidence: {result['confidence']:.2f}")
                print(f"📚 Sources: {result['num_contexts_used']}")
                
                show_sources = input("\n🔍 Show sources? (y/n): ").lower().startswith('y')
                if show_sources:
                    print("\n📖 Sources:")
                    for i, source in enumerate(result['sources'], 1):
                        print(f"  {i}. {source['source']} (Score: {source['score']:.3f})")
                        print(f"     {source['text']}")
                        print()
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

    def _save_index(self):
        """Save the index to disk."""
        if self.faiss_index is not None:
            faiss.write_index(self.faiss_index, "index/hybrid_faiss.index")
        
        index_state = {
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'tfidf_fitted': self.tfidf_fitted,
            'chunks_count': len(self.chunks_store),
            'embedding_dim': self.embedding_dim,
            'chunk_id_to_index': self.chunk_id_to_index,
            'index_built': self.index_built
        }
        
        with open('index/hybrid_index_state.pkl', 'wb') as f:
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
        
        with open('index/hybrid_chunks_metadata.json', 'w') as f:
            json.dump(chunks_metadata, f, indent=2)
        
        # Save embeddings
        embeddings_array = np.array([chunk.embedding for chunk in self.chunks_store])
        np.save('index/hybrid_embeddings.npy', embeddings_array)

    def load_index(self, use_gpu: bool = False):
        """Load a previously saved index."""
        try:
            # Load FAISS index
            self.faiss_index = faiss.read_index("index/hybrid_faiss.index")
            
            if use_gpu and faiss.get_num_gpus() > 0:
                res = faiss.StandardGpuResources()
                self.faiss_index = faiss.index_cpu_to_gpu(res, 0, self.faiss_index)
            
            # Load other components
            with open('index/hybrid_index_state.pkl', 'rb') as f:
                index_state = pickle.load(f)
            
            self.tfidf_vectorizer = index_state['tfidf_vectorizer']
            self.tfidf_fitted = index_state['tfidf_fitted']
            self.embedding_dim = index_state['embedding_dim']
            self.chunk_id_to_index = index_state['chunk_id_to_index']
            self.index_built = index_state['index_built']
            
            # Load chunks metadata and embeddings
            with open('index/hybrid_chunks_metadata.json', 'r') as f:
                chunks_metadata = json.load(f)
            
            embeddings_array = np.load('index/hybrid_embeddings.npy')
            
            # After loading chunks_metadata and embeddings_array
            contextualized_texts = [chunk['contextualized_text'] for chunk in chunks_metadata]
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(contextualized_texts)
            self.tfidf_fitted = True

            # Reconstruct chunks with tfidf_vector
            self.chunks_store = []
            for i, chunk_meta in enumerate(chunks_metadata):
                chunk_obj = DocumentChunk(
                    original_text=chunk_meta['original_text'],
                    contextualized_text=chunk_meta['contextualized_text'],
                    embedding=embeddings_array[i],
                    tfidf_vector=tfidf_matrix[i],
                    metadata=chunk_meta['metadata'],
                    chunk_id=chunk_meta['chunk_id']
                )
                self.chunks_store.append(chunk_obj)
            
            print(f"Loaded hybrid index with {len(chunks_metadata)} chunks")
            return True
            
        except FileNotFoundError as e:
            print(f"No saved index found: {e}")
            return False

# Example usage and demonstration
if __name__ == "__main__":
    #load 5 questions from data/questions.txt
   #load the samples
    samples = load_samples_from_jsonl("../../data/formatted_summary_questions/summary_questions.jsonl",n=5)
    print(f"Loaded {len(samples)} questions.")
    #find the data
    txt_files = glob.glob("../../data/ocr_output/*.txt")
    print(f"Found {len(txt_files)} .txt files in data/ocr_output.")

    # Initialize the hybrid system
    hybrid_rag = HybridContextualRAG(
        project_id="camels-453517",
        location="us-central1",
        verbose=True,
        contextual_RAG=False
    )
    

    # Load documents and build the index
    # documents = hybrid_rag.load_documents_from_files(txt_files)
    # chunks = hybrid_rag.split_documents(documents)
    # hybrid_rag._build_contextual_index(chunks)
    
    #create a new directory to save the results
    os.makedirs("../results/summary_questions_ocr_embedding", exist_ok=True)

    hybrid_rag.load_index()

    #generate the response from the rag.
    for idx, sample in enumerate(samples[0:5], 0):
        question = sample['question']
        ideal_answer = sample['ideal']
        expected_citations = sample['citations']
        key_passage = sample['key_passage']
        source_file = sample['source_file']
        print(f"Generating response for question: {question}")
        response = hybrid_rag.answer_question(question)
        print(f"Response: {response}")
    
        # Store each question, ideal, response, expected citations, key passage, source file in a separate json file
        output_path = f"../extension/results/summary_questions_ocr_embedding/ocr_embedding_response_{idx}.json"
        with open(output_path, "w") as f:
            json.dump({
                "question": question,
                "ideal_answer": ideal_answer,
                "generated_answer": response['answer'],
                "expected_citations": expected_citations,
                "key_passage": key_passage,
            }, f, indent=2)