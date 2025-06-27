from pathlib import Path
import vertexai
from google import genai
from glob import glob
import os
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle
from google.oauth2 import service_account

# Using pathlib (modern approach) to define the base directory as the directory that contains this file.
BASE_DIR = Path(__file__).resolve().parent

# REPO_DIR is defined as one directory above the package
REPO_DIR = BASE_DIR.parent

markdown_files_path = REPO_DIR / "markdowns"
datasets_path = REPO_DIR / "datasets"
embeddings_path = REPO_DIR / "embeddings"

DATASET = "CosmoPaperQA.parquet"

import google.auth

credentials, project = google.auth.default()


VERTEX_EMBEDDING_MODEL = "publishers/google/models/text-embedding-005"  # @param {type:"string", isTemplate: true}
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"


CHUNK_SIZE = 5000
CHUNK_OVERLAP = 250

display_name = "corpus"
assistant_name = "rag_agent"


TOP_K = 20
DISTANCE_THRESHOLD = 0.5
semantic_weight = 0.7

GEMINI_GEN_MODEL = "gemini-2.5-flash-preview-05-20"
OPENAI_GEN_MODEL = "gpt-4.1"

TEMPERATURE = 0.01

PERPLEXITY_MODEL="sonar-reasoning-pro"



LOCATION="us-central1"
PROJECT = "camels-453517"
vertexai.init(project=PROJECT,location=LOCATION,credentials=credentials)

# creds = service_account.IDTokenCredentials.from_service_account_file(
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"],
# target_audience="https://generativelanguage.googleapis.com"
# )
# creds.refresh(Request())
# vertexai.init(project=PROJECT, location=LOCATION, credentials=creds)
vertex_client = genai.Client(vertexai=True,project=PROJECT,location=LOCATION)




SCOPES = ['https://www.googleapis.com/auth/drive.file']
folder_id="10PHjfTh-2Ur8n7pgiTDP1TsxURTaZATS" 

# you must create this folder in your drive and enable the drive api in the console
# see here https://chatgpt.com/share/6840a0ed-bf88-800c-9c71-1cc5d49960a8 to enable the drive api






def authenticate_gdrive(scopes=SCOPES):
    creds = None
    # Token stores the user's access/refresh tokens
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    # If no credentials or invalid, let user log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', scopes)
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    return build('drive', 'v3', credentials=creds)

def upload_markdowns_to_gdrive():
    """Upload all .md files in output_dir to the given Google Drive folder ID."""
    service = authenticate_gdrive()
    md_files = glob(os.path.join(markdown_files_path, '*.md'))
    for md_file in md_files:
        file_metadata = {
            'name': os.path.basename(md_file),
            'parents': [folder_id]
        }
        media = MediaFileUpload(md_file, mimetype='text/markdown')
        uploaded = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()
        print(f"Uploaded {md_file} to Google Drive with ID: {uploaded.get('id')}")


from google.cloud import storage
from pathlib import Path


bucket_name = "cmbagent"

def upload_markdown_files_to_gcs():
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    local_dir = Path(markdown_files_path)
    gs_paths = []
    for file_path in local_dir.glob("*.md"):
        blob = bucket.blob(file_path.name)
        blob.upload_from_filename(str(file_path))
        print(f"Uploaded {file_path.name} to gs://{bucket_name}/{file_path.name}")
        gs_paths.append(f"gs://{bucket_name}/{file_path.name}")
    return gs_paths

# PaperQA2 configuration variables
PAPERQA2_EMBEDDING = "text-embedding-3-large"
PAPERQA2_LLM = "gpt-4.1"
PAPERQA2_TEMPERATURE = 0.01
PAPERQA2_EVIDENCE_K = 10
PAPERQA2_EVIDENCE_K = 30
PAPERQA2_ANSWER_MAX_SOURCES = 5

from openai import OpenAI
import os
openai_client = OpenAI(api_key = os.environ["OPENAI_API_KEY"])

OpenAI_Embedding_Model = "text-embedding-3-large"



# The below pricing is for 1K tokens. Whenever there is an update in the LLM's pricing,
# Please convert it to 1K tokens and update in the below dictionary in the format: (input_token_price, output_token_price).
OAI_PRICE1K = {
    # https://openai.com/api/pricing/
    # o1
    "o1-preview-2024-09-12": (0.0015, 0.0060),
    "o1-preview": (0.0015, 0.0060),
    # 4.5 mini
    "gpt-4.5-preview-2025-02-27": (0.075, 0.15),
    "o1-mini-2024-09-12": (0.0003, 0.0012),
    "o1-mini": (0.0003, 0.0012),
    "o1": (0.0015, 0.0060),
    "o1-2024-12-17": (0.0015, 0.0060),
    # o1 pro
    "o1-pro": (0.15, 0.6),  # $150 / $600!
    "o1-pro-2025-03-19": (0.15, 0.6),
    # o3
    "o3": (0.0011, 0.0044),
    "o3-mini-2025-01-31": (0.0011, 0.0044),
    # gpt-4o
    "gpt-4o": (0.005, 0.015),
    "gpt-4o-2024-05-13": (0.005, 0.015),
    "gpt-4o-2024-08-06": (0.0025, 0.01),
    "gpt-4o-2024-11-20": (0.0025, 0.01),
    # gpt-4o-mini
    "gpt-4o-mini": (0.000150, 0.000600),
    "gpt-4o-mini-2024-07-18": (0.000150, 0.000600),
    # gpt-4-turbo
    "gpt-4-turbo-2024-04-09": (0.01, 0.03),
    # gpt-4
    "gpt-4": (0.03, 0.06),
    "gpt-4-32k": (0.06, 0.12),
    # gpt-4.1
    "gpt-4.1": (0.002, 0.008),
    "gpt-4.1-2025-04-14": (0.002, 0.008),
    # gpt-4.1 mini
    "gpt-4.1-mini": (0.0004, 0.0016),
    "gpt-4.1-mini-2025-04-14": (0.0004, 0.0016),
    # gpt-4.1 nano
    "gpt-4.1-nano": (0.0001, 0.0004),
    "gpt-4.1-nano-2025-04-14": (0.0001, 0.0004),
    # gpt-3.5 turbo
    "gpt-3.5-turbo": (0.0005, 0.0015),  # default is 0125
    "gpt-3.5-turbo-0125": (0.0005, 0.0015),  # 16k
    "gpt-3.5-turbo-instruct": (0.0015, 0.002),
    # base model
    "davinci-002": 0.002,
    "babbage-002": 0.0004,
    # old model
    "gpt-4-0125-preview": (0.01, 0.03),
    "gpt-4-1106-preview": (0.01, 0.03),
    "gpt-4-1106-vision-preview": (0.01, 0.03),  # TODO: support vision pricing of images
    "gpt-3.5-turbo-1106": (0.001, 0.002),
    "gpt-3.5-turbo-0613": (0.0015, 0.002),
    # "gpt-3.5-turbo-16k": (0.003, 0.004),
    "gpt-3.5-turbo-16k-0613": (0.003, 0.004),
    "gpt-3.5-turbo-0301": (0.0015, 0.002),
    "text-ada-001": 0.0004,
    "text-babbage-001": 0.0005,
    "text-curie-001": 0.002,
    "code-cushman-001": 0.024,
    "code-davinci-002": 0.1,
    "text-davinci-002": 0.02,
    "text-davinci-003": 0.02,
    "gpt-4-0314": (0.03, 0.06),  # deprecate in Sep
    "gpt-4-32k-0314": (0.06, 0.12),  # deprecate in Sep
    "gpt-4-0613": (0.03, 0.06),
    "gpt-4-32k-0613": (0.06, 0.12),
    "gpt-4-turbo-preview": (0.01, 0.03),
    # https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/#pricing
    "gpt-35-turbo": (0.0005, 0.0015),  # what's the default? using 0125 here.
    "gpt-35-turbo-0125": (0.0005, 0.0015),
    "gpt-35-turbo-instruct": (0.0015, 0.002),
    "gpt-35-turbo-1106": (0.001, 0.002),
    "gpt-35-turbo-0613": (0.0015, 0.002),
    "gpt-35-turbo-0301": (0.0015, 0.002),
    "gpt-35-turbo-16k": (0.003, 0.004),
    "gpt-35-turbo-16k-0613": (0.003, 0.004),
    # deepseek
    "deepseek-chat": (0.00027, 0.0011),
}

# Gemini 2.5 Flash pricing (per 1K tokens) - Add this to the existing OAI_PRICE1K dictionary
GEMINI_PRICE1K = {
    # Gemini 2.5 Flash pricing based on search results
    "gemini-2.5-flash": (0.00015, 0.0006),  # (input, output) without thinking
    "gemini-2.5-flash-thinking": (0.00015, 0.0035),  # (input, output) with thinking
    "gemini-2.5-flash-preview-05-20": (0.00015, 0.0006),  # specific version without thinking
    "gemini-2.5-flash-preview-05-20-thinking": (0.00015, 0.0035),  # specific version with thinking
    
    # Additional Gemini models for reference
    "gemini-2.0-flash": (0.0001, 0.0004),  # Based on search results mentioning $0.10/$0.40 per 1M
    "gemini-1.5-pro": (0.00125, 0.005),    # Based on search results
    "gemini-1.5-flash": (0.0, 0.0),        # Free tier mentioned in search results
}

# Add to the existing OAI_PRICE1K dictionary
OAI_PRICE1K.update(GEMINI_PRICE1K)

from pydantic import BaseModel, Field
from typing import List
from paperqa.settings import Settings, AnswerSettings, AgentSettings

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
OCR_OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'txt_files')

class AnswerFormat(BaseModel):
    answer: str = Field(
        description="The answer to the question using the given files only. Must be concise. At most 2 sentences."
    )
    sources: List[str] = Field(
        description=(
            "A list of source names used to formulate the answer. "
            "If possible, include page number, equation number, table number, section number, etc."
        )
    )

CANNOT_ANSWER_PHRASE = "I cannot answer"
CITATION_KEY_CONSTRAINTS = (
    "## Valid citation examples: \n"
    "- Example2024Example pages 3-4 \n"
    "- Example2024 pages 3-4 \n"
    "- Example2024 pages 3-4, Example2024 pages 5-6 \n"
    "## Invalid citation examples: \n"
    "- Example2024Example pages 3-4 and pages 4-5 \n"
    "- Example2024Example (pages 3-4) \n"
    "- Example2024Example pages 3-4, pages 5-6 \n"
    "- Example2024Example et al. (2024) \n"
    "- Example's work (pages 17–19) \n"  # noqa: RUF001
    "- (pages 17–19) \n"  # noqa: RUF001
)
qa_prompt = (
    "Provide a concise answer in 1-2 sentences maximum. "
    "Context (with relevance scores):\n\n{context}\n\n----\n\n"
    "Question: {question}\n\n"
    "Write a concise answer based on the context, focusing on astronomical facts and concepts. "
    "If the context provides insufficient information reply "
    f'"{CANNOT_ANSWER_PHRASE}." '
    "Write in the style of a scientific astronomy reference, with precise and "
    "factual statements. The context comes from a variety of sources and is "
    "only a summary, so there may be inaccuracies or ambiguities. \n\n"
    "{prior_answer_prompt}"
    "Answer (maximum one sentence):"
)



paperqa2_settings = Settings(
        llm=PAPERQA2_LLM,
        llm_config={
            "model_list": [
                {
                    "model_name": PAPERQA2_LLM,
                    "litellm_params": {
                        "model": PAPERQA2_LLM,
                        "temperature": PAPERQA2_TEMPERATURE,
                        "max_tokens": 4096,
                    },
                }
            ]
        },
        summary_llm=PAPERQA2_LLM,
        summary_llm_config={
            "rate_limit": {PAPERQA2_LLM: "30000 per 1 minute"},
        },
        answer=AnswerSettings(
            evidence_k=PAPERQA2_EVIDENCE_K,
            answer_max_sources=PAPERQA2_ANSWER_MAX_SOURCES,
            evidence_skip_summary=False,
            answer_length = "1-2 sentences maximum"
        ),
        agent=AgentSettings(
            agent_llm=PAPERQA2_LLM,
            agent_llm_config={
                "rate_limit": {PAPERQA2_LLM: "30000 per 1 minute"},
            }
        ),
        embedding=PAPERQA2_EMBEDDING,
        temperature=PAPERQA2_TEMPERATURE,
        paper_directory=OCR_OUTPUT_DIR
        # prompt={
        #     "qa": qa_prompt
        # }
    )

index_settings = Settings(
                paper_directory=OCR_OUTPUT_DIR,
                agent={"index": {
                    "sync_with_paper_directory": True,
                    "recurse_subdirectories": True
                }}
            )