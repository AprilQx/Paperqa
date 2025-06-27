from typing import List, Dict, Any
from typing import Any, Optional, Union
import asyncio
import time
import os
import re
import json
import pandas as pd
import numpy as np

import shutil
from pathlib import Path
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

from .config import (openai_client,
                     credentials, 
                     CHUNK_SIZE, CHUNK_OVERLAP,
                     TOP_K, DISTANCE_THRESHOLD,
                     OPENAI_GEN_MODEL,
                     TEMPERATURE,
                     display_name,
                     folder_id,
                     markdown_files_path,
                     embeddings_path,
                     OAI_PRICE1K,
                     AnswerFormat,
                     assistant_name,
                     OpenAI_Embedding_Model)

from .scirag import SciRag
import chromadb
from chromadb.config import Settings

class SciRagOpenAI(SciRag):
    def __init__(self, 
                 client=openai_client,
                 credentials=credentials,
                 markdown_files_path=markdown_files_path,
                 corpus_name=display_name,
                 gen_model=OPENAI_GEN_MODEL,
                 vector_db_backend="chromadb",  # "openai" or "chromadb"
                 chroma_collection_name="sci_rag_openai_chunks",
                 chroma_db_path=str(embeddings_path / "chromadb_openai"),
                 ):
        super().__init__(client, credentials, markdown_files_path, corpus_name, gen_model)
        
        self.vector_db_backend = vector_db_backend
        self.chroma_collection_name = chroma_collection_name
        self.chroma_db_path = chroma_db_path
        self.chromadb_built = False

        
        if vector_db_backend == "openai":
            # Original OpenAI vector store implementation
            self._setup_openai_vector_store()
        elif vector_db_backend == "chromadb":
            # New ChromaDB implementation
            self._setup_chromadb()
        else:
            raise ValueError(f"Unknown vector_db_backend: {vector_db_backend}")

    def _setup_openai_vector_store(self):
        """Original OpenAI vector store setup - only for existing vector stores"""
        print("Listing existing RAG Corpora:")
        vector_stores = self.client.vector_stores.list()

        rag_corpus_found = False
        for vs in vector_stores:
            if vs.name == self.corpus_name:
                print(f"--- Found existing corpus: {vs.name} ---")
                self.rag_corpus = vs
                rag_corpus_found = True
                break
        
        # Only create assistant if we found an existing corpus
        if rag_corpus_found:
            print("Creating assistant for existing vector store...")
            self.rag_assistant = self.client.beta.assistants.create(
                name=assistant_name,
                instructions=self.rag_prompt,
                tools=[
                    {"type": "file_search",
                        "file_search":{
                            'max_num_results': TOP_K,
                            "ranking_options": {
                                "ranker": "auto",
                                "score_threshold": DISTANCE_THRESHOLD
                            }
                        }
                    }
                ],
                tool_resources={"file_search": {"vector_store_ids":[self.rag_corpus.id]}},
                model=self.gen_model, 
                temperature=TEMPERATURE,
                top_p=0.2,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "answer",
                        "schema": AnswerFormat.model_json_schema()
                    },
                }
            )
            print(f"Assistant created for existing corpus with ID: {self.rag_assistant.id}")
        else:
            print(f"No existing vector store found with name '{self.corpus_name}'.")
            print("Use create_vector_db() to create a new one.")
            self.rag_corpus = None
            self.rag_assistant = None
    def _setup_chromadb(self):
        """Setup ChromaDB implementation"""
        # Load and process documents
        self.docs = self.load_markdown_files()
        self.split_documents()  # create self.all_chunks
        
        # Prepare texts and metadata
        self._get_texts()
        
        # Create or load ChromaDB collection
        try:
            self.load_chromadb_collection()
            print(f"Loaded existing ChromaDB collection '{self.chroma_collection_name}'")
        except Exception as e:
            print(f"Could not load existing collection: {e}")
            print("Creating new ChromaDB collection...")
            self._create_chromadb_embeddings()
            self._store_to_chromadb_safe()

    def load_embeddings(self):
        try:
            self.embeddings = np.load(embeddings_path/'chroma.sqlite3')
        except FileNotFoundError:
            # print(f"Embeddings file not found. You must call get_embeddings() first...")
            raise FileNotFoundError(f"Embeddings file not found. You must call get_embeddings() first...")


    def _get_texts(self):
        """Prepare texts and metadata from chunks"""
        print("Preparing texts for ChromaDB...")
        
        all_original_texts = []
        all_metadata = []
        
        chunk_counter = 0
        for chunk in self.all_chunks:
            all_original_texts.append(chunk.page_content)
            
            # Extract just filename from source path
            source_path = chunk.metadata.get('source', '')
            filename = os.path.basename(source_path) if source_path else 'Unknown'
            
            all_metadata.append({
                **chunk.metadata,
                'chunk_id': f"chunk_{chunk_counter}",
                'filename': filename,  # Store clean filename separately
            })
            chunk_counter += 1
            
        print(f"Processed {chunk_counter} chunks")
        self.all_metadata = all_metadata
        self.all_texts = all_original_texts

    def _create_chromadb_embeddings(self):
        """Create embeddings using OpenAI's embedding model"""
        print("Creating embeddings using OpenAI...")
        embeddings = []
        
        # Use OpenAI's text-embedding-3-small or text-embedding-ada-002
        embedding_model = OpenAI_Embedding_Model
        
        batch_size = 100  # Process in batches to avoid rate limits
        
        for i in range(0, len(self.all_texts), batch_size):
            batch_texts = self.all_texts[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(self.all_texts) + batch_size - 1)//batch_size}")
            
            try:
                response = self.client.embeddings.create(
                    input=batch_texts,
                    model=embedding_model
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                
                # Rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error creating embeddings for batch {i//batch_size + 1}: {e}")
                # Add zero embeddings as fallback
                embeddings.extend([[0.0] * 1536] * len(batch_texts))  # 1536 is dimension for text-embedding-3-small
        
        self.embeddings = embeddings
        print(f"Created {len(embeddings)} embeddings")

    def _store_to_chromadb_safe(self):
        """Store embeddings and texts in ChromaDB with proper error handling"""
        import chromadb
        from chromadb.config import Settings
        import time
        
        # Try to clean up any existing ChromaDB instances
        try:
            # Force reset any existing instances
            if hasattr(chromadb, '_client_cache'):
                chromadb._client_cache.clear()
        except:
            pass
            
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Create unique path for this attempt if needed
                db_path = self.chroma_db_path
                collection_name = self.chroma_collection_name
                
                if attempt > 0:
                    db_path = f"{self.chroma_db_path}_attempt_{attempt}"
                    collection_name = f"{self.chroma_collection_name}_attempt_{attempt}"
                    print(f"Retry {attempt}: Using path {db_path} and collection {collection_name}")
                
                # Create client with unique settings
                client = chromadb.PersistentClient(
                    path=db_path,
                    settings=Settings(
                        allow_reset=True,
                        anonymized_telemetry=False,
                        is_persistent=True
                    )
                )
                
                # Delete existing collection if it exists
                try:
                    existing_collection = client.get_collection(name=collection_name)
                    client.delete_collection(name=collection_name)
                    print(f"Deleted existing collection '{collection_name}'")
                except:
                    pass  # Collection doesn't exist, which is fine
                
                # Create new collection
                collection = client.create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                
                # Store the successful paths
                self.chroma_db_path = db_path
                self.chroma_collection_name = collection_name
                break
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise Exception(f"Failed to create ChromaDB client after {max_retries} attempts: {e}")
                time.sleep(1)  # Wait before retry
        
        # Prepare data for storage
        ids = [m.get('chunk_id', f'chunk_{i}') for i, m in enumerate(self.all_metadata)]
        embeddings = self.embeddings
        documents = self.all_texts
        metadatas = self.all_metadata
        
        # Add in batches to avoid memory issues
        batch_size = 1000
        total_batches = (len(ids) + batch_size - 1) // batch_size
        
        for start in range(0, len(ids), batch_size):
            end = min(start + batch_size, len(ids))
            batch_num = start // batch_size + 1
            print(f"Storing batch {batch_num}/{total_batches}")
            
            try:
                collection.add(
                    ids=ids[start:end],
                    embeddings=embeddings[start:end],
                    documents=documents[start:end],
                    metadatas=metadatas[start:end],
                )
            except Exception as batch_error:
                print(f"Error storing batch {batch_num}: {batch_error}")
                # Continue with next batch instead of failing completely
                continue
        
        self.chroma_collection = collection
        print(f"Successfully stored {len(ids)} chunks in ChromaDB collection '{self.chroma_collection_name}'")
        print(f"Collection stored at: {self.chroma_db_path}")
        self.chromadb_built = True

    def load_chromadb_collection(self):
        """Load existing ChromaDB collection with better error handling"""
        import chromadb
        from chromadb.config import Settings
        
        try:
            client = chromadb.PersistentClient(
                path=self.chroma_db_path,
                settings=Settings(
                    allow_reset=True,
                    anonymized_telemetry=False,
                    is_persistent=True
                )
            )
            self.chroma_collection = client.get_collection(name=self.chroma_collection_name)
            print(f"Successfully loaded ChromaDB collection '{self.chroma_collection_name}'")
        except Exception as e:
            # If we can't load, we'll need to create a new one
            raise Exception(f"Could not load collection: {e}")

    def query_chromadb(self, query: str, n_results: int = TOP_K) -> Dict[str, Any]:
        """Query ChromaDB collection"""
        # Create query embedding using OpenAI
        response = self.client.embeddings.create(
            input=[query],
            model=OpenAI_Embedding_Model
        )
        query_embedding = response.data[0].embedding
        
        results = self.chroma_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        return {
            "chunks": results["documents"][0],
            "metadata": results["metadatas"][0],
            "similarities": [1 - d for d in results["distances"][0]]  # Convert distance to similarity
        }
        
    def create_vector_db(self, folder_id=folder_id):
        """Create vector database - supports both OpenAI and ChromaDB backends"""
        if self.vector_db_backend == "openai":
            # Original OpenAI implementation
            chunking_strategy = {
                "type": "static",
                "static": {
                    "max_chunk_size_tokens": CHUNK_SIZE,
                    "chunk_overlap_tokens": CHUNK_OVERLAP
                }
            }

            vector_store = self.client.vector_stores.create(
                name=self.corpus_name, 
                chunking_strategy=chunking_strategy
            )
            self.rag_corpus = vector_store

            # Get all local .md files
            file_paths = [
                os.path.join(markdown_files_path, f)
                for f in os.listdir(markdown_files_path)
                if f.endswith('.md')
            ]
            if not file_paths:
                print("No markdown files found.")
                return

            print(f"Uploading {len(file_paths)} markdown files to OpenAI vector store...")

            # Open files in binary mode
            files = [open(path, "rb") for path in file_paths]
            try:
                # Upload and poll the file batch
                response = self.client.vector_stores.file_batches.upload_and_poll(
                    vector_store_id=self.rag_corpus.id,
                    files=files,
                )
                print("Upload complete. Status:", response.status)
                
                # NOW CREATE THE ASSISTANT after successful upload
                print("Creating OpenAI assistant...")
                self.rag_assistant = self.client.beta.assistants.create(
                    name=assistant_name,
                    instructions=self.rag_prompt,
                    tools=[
                        {"type": "file_search",
                            "file_search":{
                                'max_num_results': TOP_K,
                                "ranking_options": {
                                    "ranker": "auto",
                                    "score_threshold": DISTANCE_THRESHOLD
                                }
                            }
                        }
                    ],
                    tool_resources={"file_search": {"vector_store_ids":[self.rag_corpus.id]}},
                    model=self.gen_model, 
                    temperature=TEMPERATURE,
                    top_p=0.2,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "answer",
                            "schema": AnswerFormat.model_json_schema()
                        },
                    }
                )
                print(f"Assistant created successfully with ID: {self.rag_assistant.id}")
                
            finally:
                for f in files:
                    f.close()
                    
        elif self.vector_db_backend == "chromadb":
            # ChromaDB implementation already handled in __init__
            print("ChromaDB vector database ready")

    def get_response(self, query: str):
        """Get response - supports both OpenAI and ChromaDB backends"""
        if self.vector_db_backend == "openai":
            # Check if assistant exists
            if not hasattr(self, 'rag_assistant') or self.rag_assistant is None:
                raise AttributeError("OpenAI assistant not found. Make sure to create the vector store first or check if the corpus exists.")
            
            thread = self.client.beta.threads.create(
                messages=[],
            )

            parsed = self.client.beta.threads.messages.create(
                            thread_id=thread.id,
                            content=self.enhanced_query(query),
                            role='user',
                        )

            run = self.client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=self.rag_assistant.id,
                # pass the latest system message as instructions
                instructions=self.rag_prompt,
                tool_choice={"type": "file_search", "function": {"name": "file_search"}}
            )

            return self._get_run_response(thread, run)
            
        elif self.vector_db_backend == "chromadb":
            # ChromaDB implementation
            search_results = self.query_chromadb(query)
            
            # Prepare context with clean filenames
            context_pieces = []
            for i, (chunk, metadata) in enumerate(zip(search_results['chunks'], search_results['metadata']), 1):
                # Use the clean filename we stored
                filename = metadata.get('filename', metadata.get('source_file', 'Unknown'))
                context_pieces.append(f"[Context {i} - Source: {filename}]\n{chunk}\n")
            
            context_text = "\n".join(context_pieces)
            
            # Create ChromaDB-specific prompt
            chromadb_prompt = rf"""
You are a helpful assistant. 
Your answer should be in markdown format with the following structure: 

**Answer**:

{{answer}}

**Sources**:

{{sources}}

The sources must be from the **Context** material provided in the *Context* section.
You must report the source names in the sources field, if possible, the page number, equation number, table number, section number, etc.

"""

            # Get response from OpenAI
            response = self.client.chat.completions.create(
                model=self.gen_model,
                messages=[
                    {"role": "user", "content": chromadb_prompt}
                ],
                temperature=TEMPERATURE,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "answer",
                        "schema": AnswerFormat.model_json_schema()
                    }
                }
            )
            
            return self.format_chromadb_response(response.choices[0].message.content)
        
    def format_chromadb_response(self, response_content):
        """Format ChromaDB response similar to OpenAI assistant format"""
        try:
            parsed = json.loads(response_content)
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
        except Exception as e:
            return f"Could not parse response: {response_content}\nError: {e}"

    # Keep all the original methods for OpenAI compatibility
    def delete_assistant_by_name(self, assistant_name=assistant_name):
        """Delete assistant by name (OpenAI only)"""
        if self.vector_db_backend != "openai":
            print("Assistant deletion only supported for OpenAI backend")
            return []
            
        assistants = self.client.beta.assistants.list()
        deleted_ids = []
        for a in assistants:
            try:
                name = getattr(a, "name", None) or a.get("name")
                if name == assistant_name:
                    a_id = getattr(a, "id", None) or a.get("id")
                    if a_id:
                        self.client.beta.assistants.delete(a_id)
                        deleted_ids.append(a_id)
            except Exception as e:
                continue
        return deleted_ids

    def _wait_for_run(self, run_id: str, thread_id: str) -> Any:
        """Wait for OpenAI assistant run to complete"""
        in_progress = True
        while in_progress:
            run = self.client.beta.threads.runs.retrieve(run_id, thread_id=thread_id)
            in_progress = run.status in ("in_progress", "queued")
            if in_progress:
                time.sleep(0.1)
        return run

    def _format_assistant_message(self, message_content):
        """Format OpenAI assistant message with citations"""
        annotations = message_content.annotations
        citations = []

        for index, annotation in enumerate(annotations):
            message_content.value = message_content.value.replace(annotation.text, f" [{index}]")

            if file_citation := getattr(annotation, "file_citation", None):
                try:
                    cited_file = self.client.files.retrieve(file_citation.file_id)
                    citations.append(f"[{index}] {cited_file.filename}")
                except Exception as e:
                    print(f"Error retrieving file citation: {e}")
            elif file_path := getattr(annotation, "file_path", None):
                try:
                    cited_file = self.client.files.retrieve(file_path.file_id)
                    citations.append(f"[{index}] Click <here> to download {cited_file.filename}")
                except Exception as e:
                    print(f"Error retrieving file citation: {e}")

        return message_content.value

    def format_assistant_json_response(self, messages):
        """Format OpenAI assistant JSON response"""
        outputs = []
        for msg in messages:
            if msg.get("role") == "assistant":
                raw_content = msg.get("content", "").strip()
                try:
                    parsed = json.loads(raw_content)
                    answer = parsed.get("answer") or parsed.get("Answer") or ""
                    sources = parsed.get("sources") or parsed.get("Sources") or []
                    if isinstance(sources, list) and len(sources) == 1 and isinstance(sources[0], str):
                        sources_str = sources[0]
                    elif isinstance(sources, list):
                        sources_str = ", ".join(sources)
                    else:
                        sources_str = str(sources)
                    
                    # Fix the formatting - remove the extra indentation
                    outputs.append(f"""**Answer**:

{answer}

**Sources**:

{sources_str}
""")
                except Exception as e:
                    outputs.append(f"Could not parse content for message: {raw_content}...\nError: {e}")
        return "\n---\n".join(outputs)

    def _get_run_response(self, thread, run):
        while True:
            run = self._wait_for_run(run.id, thread.id)
            if run.status == "completed":
                response_messages = self.client.beta.threads.messages.list(thread.id, order="asc")

                # register cost 
                prompt_tokens = run.usage.prompt_tokens
                completion_tokens = run.usage.completion_tokens
                total_tokens = run.usage.total_tokens

                cost = get_cost(run)
                tokens_dict = {
                    "model": run.model,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "cost": cost
                }
                print_usage_summary(tokens_dict, self.cost_dict)




                new_messages = []
                for msg in response_messages:
                    if msg.run_id == run.id:
                        for content in msg.content:
                            if content.type == "text":
                                # Remove numerical references from the content
                                cleaned_content = remove_numerical_references(self._format_assistant_message(content.text))
                                new_messages.append(
                                    {"role": msg.role, 
                                    "content": cleaned_content}
                                )
                return self.format_assistant_json_response(new_messages)



# Keep the utility functions
def remove_numerical_references(text):
    cleaned_text = re.sub(r'\[\d+\]', '', text)
    return cleaned_text

def get_cost(run):
    """Calculate the cost of the run."""
    model = run.model
    if model not in OAI_PRICE1K:
        print(
            f'Model {model} is not found. The cost will be 0. In your config_list, add field {{"price" : [prompt_price_per_1k, completion_token_price_per_1k]}} for customized pricing.'
        )
        return 0

    n_input_tokens = run.usage.prompt_tokens if run.usage is not None else 0
    n_output_tokens = run.usage.completion_tokens if run.usage is not None else 0
    if n_output_tokens is None:
        n_output_tokens = 0
    tmp_price1K = OAI_PRICE1K[model]
    if isinstance(tmp_price1K, tuple):
        return (tmp_price1K[0] * n_input_tokens + tmp_price1K[1] * n_output_tokens) / 1000
    return tmp_price1K * (n_input_tokens + n_output_tokens) / 1000

def print_usage_summary(tokens_dict, cost_dict):
    model = tokens_dict["model"]
    prompt_tokens = tokens_dict["prompt_tokens"]
    completion_tokens = tokens_dict["completion_tokens"]
    total_tokens = tokens_dict["total_tokens"]
    cost = tokens_dict["cost"]

    df = pd.DataFrame([{
        "Model": model,
        "Cost": f"{cost:.5f}",
        "Prompt Tokens": prompt_tokens,
        "Completion Tokens": completion_tokens,
        "Total Tokens": total_tokens,
    }])

    cost_dict['Cost'].append(cost) 
    cost_dict['Prompt Tokens'].append(prompt_tokens)
    cost_dict['Completion Tokens'].append(completion_tokens)
    cost_dict['Total Tokens'].append(total_tokens)