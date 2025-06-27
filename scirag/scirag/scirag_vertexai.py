from typing import List
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
from google.oauth2 import service_account

import tiktoken

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


from .config import (vertex_client,
                     credentials, 
                     VERTEX_EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP,
                     TOP_K, DISTANCE_THRESHOLD,
                     GEMINI_GEN_MODEL,
                     TEMPERATURE,
                     display_name,
                     authenticate_gdrive,
                     folder_id,SCOPES,
                     markdown_files_path)

from .scirag import SciRag
# Enhanced cost tracking functions
def remove_numerical_references(text):
    """Remove numerical references from text"""
    import re
    cleaned_text = re.sub(r'\[\d+\]', '', text)
    return cleaned_text

def print_usage_summary(tokens_dict, cost_dict):
    """Print usage summary and update cost tracking"""
    import pandas as pd
    
    model = tokens_dict["model"]
    prompt_tokens = tokens_dict["prompt_tokens"]
    completion_tokens = tokens_dict["completion_tokens"]
    total_tokens = tokens_dict["total_tokens"]
    cost = tokens_dict["cost"]

    df = pd.DataFrame([{
        "Model": model,
        "Cost": f"${cost:.6f}",
        "Prompt Tokens": prompt_tokens,
        "Completion Tokens": completion_tokens,
        "Total Tokens": total_tokens,
    }])
    
    print(f"\n--- Vertex AI Usage Summary ---")
    print(df.to_string(index=False))
    print(f"--- End Summary ---\n")

    cost_dict['Cost'].append(cost) 
    cost_dict['Prompt Tokens'].append(prompt_tokens)
    cost_dict['Completion Tokens'].append(completion_tokens)
    cost_dict['Total Tokens'].append(total_tokens)
    cost_dict['Model'].append(model)


class SciRagVertexAI(SciRag):
    def __init__(self, 
                 client = vertex_client,
                 credentials = credentials,
                 markdown_files_path = markdown_files_path,
                 corpus_name = display_name,
                 gen_model = GEMINI_GEN_MODEL,
                 ):
        super().__init__(client, credentials, markdown_files_path, corpus_name, gen_model)
        self.pricing = self._get_vertex_ai_pricing()
        self.cost_dict['Model'] = []

        print("Listing RAG Corpora:")

        corpora_found = False
        for corpus in rag.list_corpora():
            corpora_found = True
            print(f"--- Corpus: {corpus.display_name} ---")
            print(f"  Name (Resource Path): {corpus.name}")
            print(f"  Display Name: {corpus.display_name}")

            # Access other common attributes:
            # Use hasattr() to safely check if an attribute exists before trying to access it,
            # as some fields might not be present for all corpora or in all states.


            if hasattr(corpus, 'create_time') and corpus.create_time:
                # Directly use corpus.create_time (which is a DatetimeWithNanoseconds object
                # or similar datetime-compatible object) with strftime
                print(f"  Create Time: {corpus.create_time.strftime('%Y-%m-%d %H:%M:%S')}")

            if hasattr(corpus, 'update_time') and corpus.update_time:
                # Directly use corpus.update_time with strftime
                print(f"  Update Time: {corpus.update_time.strftime('%Y-%m-%d %H:%M:%S')}")

            if hasattr(corpus, 'state') and corpus.state:
                print(f"  State: {corpus.state}") # e.g., Corpus.State.ACTIVE, Corpus.State.CREATING


            # You can inspect the object's attributes further if needed
            # print(f"  Full object representation: {corpus}") # This can be very verbose

            self.rag_corpus = corpus

            self.rag_retrieval_tool = Tool(
                retrieval=Retrieval(
                    vertex_rag_store=VertexRagStore(
                        rag_resources=[
                            VertexRagStoreRagResource(
                                rag_corpus=self.rag_corpus.name  # Currently only 1 corpus is allowed.
                            )
                        ],
                        similarity_top_k=TOP_K,
                        vector_distance_threshold=DISTANCE_THRESHOLD,
                    )
                )
            )

            print("-" * 30)

        if not corpora_found:
            print("No RAG corpora found in your project/region.")
    def _get_vertex_ai_pricing(self):
        """Get Vertex AI pricing for Gemini 2.5 Flash Preview model"""
        pricing = {
            "gemini-2.5-flash-preview-05-20": {
                "input_price_per_1k": 0.00015,    # $0.15 per 1M tokens (thinking version)
                "output_price_per_1k": 0.0035 # $3.50 per 1M tokens (with thinking)
            },
        }
        return pricing
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate the cost of the API call based on token usage."""
        model_name = "gemini-2.5-flash-preview-05-20"
        model_pricing = self.pricing.get(model_name, {})
        
        if not model_pricing:
            print(f"Warning: No pricing found for model {model_name}")
            return 0.0
        
        # Calculate input cost
        input_cost = (input_tokens / 1000) * model_pricing["input_price_per_1k"]
        
        # Calculate output cost based on thinking mode
    
        output_cost = (output_tokens / 1000) * model_pricing["output_price_per_1k"]

        
        total_cost = input_cost + output_cost
        return total_cost
    
    def _log_usage_and_cost(self, input_tokens: int, output_tokens: int, total_cost: float):
        """Log usage and cost information."""
        total_tokens = input_tokens + output_tokens
        
        tokens_dict = {
            "model": "gemini-2.5-flash-preview-05-20",
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cost": total_cost
        }
        
        # Enhanced logging with thinking mode info
        print(f"\n--- Vertex AI Usage Summary ---")
        print(f"Model: gemini-2.5-flash-preview-05-20")
        print(f"Input tokens: {input_tokens:,}")
        print(f"Output tokens: {output_tokens:,}")
        print(f"Total tokens: {total_tokens:,}")
        print(f"Cost: ${total_cost:.6f}")
        print(f"Input cost: ${(input_tokens / 1000) * 0.00015:.6f}")

        print(f"Output cost: ${(output_tokens / 1000) * 0.0035:.6f} (thinking)")

        print(f"--- End Summary ---\n")
        
        print_usage_summary(tokens_dict, self.cost_dict)
    def get_cost_summary(self) -> dict:
        """Get a summary of all costs and usage."""
        if not self.cost_dict['Cost']:
            return {
                "total_cost": 0.0,
                "total_calls": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_tokens": 0,
                "average_cost_per_call": 0.0,
                "model": self.gen_model
            }
        
        return {
            "total_cost": sum(self.cost_dict['Cost']),
            "total_calls": len(self.cost_dict['Cost']),
            "total_input_tokens": sum(self.cost_dict['Prompt Tokens']),
            "total_output_tokens": sum(self.cost_dict['Completion Tokens']),
            "total_tokens": sum(self.cost_dict['Total Tokens']),
            "average_cost_per_call": sum(self.cost_dict['Cost']) / len(self.cost_dict['Cost']),
            "model": self.gen_model
        }
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken encoding."""
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception as e:
            print(f"Warning: Could not count tokens: {e}")
            # Fallback: rough estimate of 1 token per 4 characters
            return len(text) // 4

    def create_vector_db(self, folder_id = folder_id):

        rag_corpus = rag.create_corpus(
            display_name=self.corpus_name,
            backend_config=rag.RagVectorDbConfig(
                rag_embedding_model_config=rag.RagEmbeddingModelConfig(
                    vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
                        publisher_model=VERTEX_EMBEDDING_MODEL
                    )
                )
            ),
        )

        print(f"Created corpus: {rag_corpus.name}")



        # service = authenticate_gdrive()

        # results = service.files().list(
        #     q=f"'{folder_id}' in parents and trashed = false",
        #     fields="files(id, name, mimeType)").execute()
        # files = results.get('files', [])

        # # Download files or get their links
        # file_paths = []
        # for file in files:
        #     file_id = file['id']
        #     # For a download link:
        #     file_paths.append(f"https://drive.google.com/uc?id={file_id}")

        # print(f"Importing files to corpus: {rag_corpus.name}")
        # print(file_paths)
        # import sys
        # sys.exit()

        rag.import_files(
            corpus_name=rag_corpus.name,
            paths=["gs://cmbagent"],
            transformation_config=rag.TransformationConfig(
                chunking_config=rag.ChunkingConfig(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            ),
        )
        # rag.import_files(
        #     corpus_name=rag_corpus.name,
        #     paths=[str(markdown_file_path) for markdown_file_path in markdown_files_path.glob("*.md")],
        #     transformation_config=rag.TransformationConfig(
        #         chunking_config=rag.ChunkingConfig(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        #     ),
        # )
        # paths=[markdown_file_path for markdown_file_path in markdown_files_path.glob("*.md")],
        # for file_path in paths[0]:
        #     print(file_path)
        #     file_display_name = file_path.stem
        #     rag_file = rag.upload_file(
        #         corpus_name=rag_corpus.name,
        #         path=str(file_path),
        #         display_name=file_display_name,
        #         description=f"{file_display_name}",
        #         transformation_config=rag.TransformationConfig(
        #             chunking_config=rag.ChunkingConfig(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        #         )
        #     )
            # # uploaded_rag_files_info.append({"name": rag_file.name, "display_name": rag_file.display_name})
            # print(f"Successfully uploaded: {rag_file.name}")



        print(f"Imported files to corpus: {rag_corpus.name}")

        self.rag_corpus = rag_corpus



    
    def delete_vector_db(self):
        rag.delete_corpus(name=self.rag_corpus.name)



    def get_chunks(self, query: str):
        response = rag.retrieval_query(
            rag_resources=[
                rag.RagResource(
                    rag_corpus=self.rag_corpus.name,
                    # Optional: supply IDs from `rag.list_files()`.
                    # rag_file_ids=["rag-file-1", "rag-file-2", ...],
                )
            ],
            rag_retrieval_config=rag.RagRetrievalConfig(
                top_k=TOP_K,  # Optional
                filter=rag.Filter(
                    vector_distance_threshold=DISTANCE_THRESHOLD,  # Optional
                ),
            ),
                    text=query,
                )
        return response



    def get_response(self, query: str):
         # Count input tokens

        input_tokens = self._count_tokens(query)
        
        response = self.client.models.generate_content(
            model=self.gen_model,
            contents=self.enhanced_query(query),
            config=GenerateContentConfig(tools=[self.rag_retrieval_tool],
                                         temperature=TEMPERATURE,
                                         system_instruction=self.rag_prompt,
                                         tool_config=tool_config,
                                         response_mime_type='application/json',
                                         response_schema=AnswerFormat,
                                         ),
        )
        output_text = response.text
        output_tokens = self._count_tokens(output_text)
        # Calculate and log cost
        total_cost = self._calculate_cost(input_tokens, output_tokens)
        self._log_usage_and_cost(input_tokens, output_tokens, total_cost)
        return self.format_agent_output(response.text)
    

    def format_agent_output(self, response):
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


        
# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import enum
import warnings
from typing import Any, Optional, Type, TypeVar, Union, get_args, get_origin

from pydantic import BaseModel as BaseModel
from pydantic import ConfigDict, Field, alias_generators


def _remove_extra_fields(model: Any, response: dict[str, object]) -> None:
    """Removes extra fields from the response that are not in the model.

    Mutates the response in place.
    """

    key_values = list(response.items())

    for key, value in key_values:
        # Need to convert to snake case to match model fields names
        # ex: UsageMetadata
        alias_map = {field_info.alias: key for key, field_info in model.model_fields.items()}

        if key not in model.model_fields and key not in alias_map:
            response.pop(key)
            continue

        key = alias_map.get(key, key)

        annotation = model.model_fields[key].annotation

        # Get the BaseModel if Optional
        if get_origin(annotation) is Union:
            annotation = get_args(annotation)[0]

        # if dict, assume BaseModel but also check that field type is not dict
        # example: FunctionCall.args
        if isinstance(value, dict) and get_origin(annotation) is not dict:
            _remove_extra_fields(annotation, value)
        elif isinstance(value, list):
            for item in value:
                # assume a list of dict is list of BaseModel
                if isinstance(item, dict):
                    _remove_extra_fields(get_args(annotation)[0], item)


T = TypeVar("T", bound="BaseModel")


class CommonBaseModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=alias_generators.to_camel,
        populate_by_name=True,
        from_attributes=True,
        protected_namespaces=(),
        extra="forbid",
        # This allows us to use arbitrary types in the model. E.g. PIL.Image.
        arbitrary_types_allowed=True,
        ser_json_bytes="base64",
        val_json_bytes="base64",
        ignored_types=(TypeVar,),
    )

    @classmethod
    def _from_response(cls: Type[T], *, response: dict[str, object], kwargs: dict[str, object]) -> T:
        # To maintain forward compatibility, we need to remove extra fields from
        # the response.
        # We will provide another mechanism to allow users to access these fields.
        _remove_extra_fields(cls, response)
        validated_response = cls.model_validate(response)
        return validated_response

    def to_json_dict(self) -> dict[str, object]:
        return self.model_dump(exclude_none=True, mode="json")


class CaseInSensitiveEnum(str, enum.Enum):
    """Case insensitive enum."""

    @classmethod
    def _missing_(cls, value: Any) -> Optional["CaseInSensitiveEnum"]:
        try:
            return cls[value.upper()]  # Try to access directly with uppercase
        except KeyError:
            try:
                return cls[value.lower()]  # Try to access directly with lowercase
            except KeyError:
                warnings.warn(f"{value} is not a valid {cls.__name__}")
                try:
                    # Creating a enum instance based on the value
                    # We need to use super() to avoid infinite recursion.
                    unknown_enum_val = super().__new__(cls, value)
                    unknown_enum_val._name_ = str(value)  # pylint: disable=protected-access
                    unknown_enum_val._value_ = value  # pylint: disable=protected-access
                    return unknown_enum_val
                except:  # noqa: E722
                    return None


class FunctionCallingConfigMode(CaseInSensitiveEnum):
    """Config for the function calling config mode."""

    MODE_UNSPECIFIED = "MODE_UNSPECIFIED"
    AUTO = "AUTO"
    ANY = "ANY"
    NONE = "NONE"


class FunctionCallingConfig(CommonBaseModel):
    """Function calling config."""

    mode: Optional[FunctionCallingConfigMode] = Field(default=None, description="""Optional. Function calling mode.""")
    allowed_function_names: Optional[list[str]] = Field(
        default=None,
        description="""Optional. Function names to call. Only set when the Mode is ANY. Function names should match [FunctionDeclaration.name]. With mode set to ANY, model will predict a function call from the set of function names provided.""",
    )


class ToolConfig(CommonBaseModel):
    """Tool config.

    This config is shared for all tools provided in the request.
    """

    function_calling_config: Optional[FunctionCallingConfig] = Field(
        default=None, description="""Optional. Function calling config."""
    )


# Must call a tool
function_calling_config = FunctionCallingConfig(
    mode=FunctionCallingConfigMode.ANY,
    )

tool_config = ToolConfig(
    function_calling_config=function_calling_config,
    )
