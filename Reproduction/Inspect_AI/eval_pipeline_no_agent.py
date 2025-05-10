import csv
import re
from paperqa import Docs
from inspect_ai import Task, task, Epochs
from inspect_ai.dataset import Dataset, Sample
from scorer import precision_choice
from inspect_ai.solver import generate, multiple_choice, solver
from inspect_ai.scorer import exact 
from inspect_ai.log import transcript
from inspect_ai.agent import bridge
from inspect_ai.util._limited_conversation import ChatMessageList
from inspect_ai.dataset import Sample, json_dataset
from inspect_ai.solver._task_state import TaskState
from inspect_ai.model import ModelOutput, StopReason
from inspect_ai.solver import generate

from paperqa import  ask
from paperqa.agents.main import agent_query
from paperqa.agents.search import get_directory_index
from paperqa.settings import Settings, AgentSettings

import json
import nest_asyncio
from shortuuid import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

# Import AutoGen components
import autogen
from autogen import UserProxyAgent,AssistantAgent

from paperqa.settings import AnswerSettings

import datetime

from pydantic import BaseModel, Field

import os
# Set environment variable to disable Docker
os.environ["AUTOGEN_USE_DOCKER"] = "False"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
PAPERS_DIR = os.path.join(PROJECT_ROOT, 'papers')
QUESTIONS_FILE = os.path.join(PROJECT_ROOT, 'Reproduction', 'Questions', 'formatted_questions_test', 'questions.jsonl')

# Define structured output models
class Evidence(BaseModel):
    source: str = Field(description="Source paper that contains evidence (DOI)")
    quote: str = Field(description="Relevant quote supporting the answer")
    relevance: str = Field(description="Why this evidence supports the chosen answer")

class MCAnswer(BaseModel):
    answer_letter: str = Field(
        description="Single letter (A-Z) corresponding to the correct answer option"
    )
    
    def format(self) -> str:
        """Return just the answer letter when stringified"""
        return self.answer_letter

class PaperQAClient:
    """A singleton client for PaperQA2 that builds the index once and reuses it."""
    
    _instance = None
    _docs = None
    _index_built = False
    _docs_initialized = False 
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(PaperQAClient, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, settings: Optional[Settings] = None):
        # Set up the settings
        self.paper_directory = PAPERS_DIR
        if settings is None:
            self.settings = Settings(
                temperature=0.1,
                llm='gpt-4o-mini',
                paper_directory=self.paper_directory,
                summary_llm='gpt-4o-mini'
            )
        else:
            self.settings = settings
            self.paper_directory = settings.paper_directory
        
        # Apply nest_asyncio
        nest_asyncio.apply()
        
        # Initialize docs but don't load papers yet (that happens in init_docs)
        if self._docs is None:
            self._docs = Docs(index_path=self.paper_directory)
    
    async def init_docs(self):
        """Initialize documents asynchronously (must be called before first query)"""
        if not self._docs or len(self._docs.docs) == 0:
            # Add the pdf files in the doc_paths
            doc_paths = [os.path.join(self.paper_directory, doc) 
                        for doc in os.listdir(self.paper_directory) 
                        if doc.endswith('.pdf')]
            
            for doc_path in doc_paths:
                try:
                    await self._docs.aadd(doc_path, settings=self.settings)
                except Exception as e:
                    print(f"Error loading document {doc_path}: {e}")
            #indicate that the docs have been initialized
            self._docs_initialized = True

    
    async def _build_index_if_needed(self):
        """Build the document index if not already built"""
        if not self._index_built:
            print("Building PaperQA2 document index (only happens once)...")
            await get_directory_index(settings=self.settings)
            self._index_built = True
    
    async def aask(self, question: str) -> str:
        """Async method to query PaperQA2"""
        # Initialize docs if needed
        await self.init_docs()
        
        # Ensure index is built
        await self._build_index_if_needed()
        
        # Query PaperQA2
        response_pqa = await self._docs.aquery(question, settings=self.settings)
        
        # Return the answer
        return response_pqa.answer

    def ask(self, question: str) -> str:
        """Synchronous method to query PaperQA2"""
        import asyncio
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.aask(question))

def record_to_sample(record: dict[str, Any]) -> Sample:
    # Format the choices as they appear in the JSONL file
    choices_text = "\n".join([choice for choice in record["choices"]])
    
    # Create the proper Sample object with correct target letter
    return Sample(
        input=f"Question: {record['question']}\nChoices:\n{choices_text}",
        target=record["correct_answer"],  # Use the actual correct answer letter
        choices=record["choices"],  # Pass the entire choices array
        metadata={
            "sources": record["sources"],
            "unsure_option": record["unsure_option"],
            "ideal": record["ideal"],
            "distractors": record["distractors"]
        }
    )




UNSURE= "Insufficient information to answer the question"



def paperqa2_agent():
   
    # Create AG2 config with structured output
    
    # The actual agent function that processes a sample
    async def run(sample:dict) -> dict:

        """Agent function for PaperQA2 using the Bridge pattern"""
        pqa_client = PaperQAClient(
            settings = Settings(
        llm="gpt-4o-mini",
        llm_config={
            "model_list": [
                {
                    "model_name": "gpt-4o-mini",
                    "litellm_params": {
                        "model": "gpt-4o-mini",
                        "temperature": 0.1,
                        "max_tokens": 4096,
                    },
                }
            ],
            "rate_limit": {
                "gpt-4o-mini": "30000 per 1 minute",
            },
        },
        summary_llm="gpt-4o-mini",
        summary_llm_config={
            "rate_limit": {
                "gpt-4o-mini": "30000 per 1 minute",
            },
        },
        answer=AnswerSettings(
            evidence_k=30, #top_k
            answer_max_sources=15,#max_cut_off in the figure
            evidence_skip_summary=True),
        agent=AgentSettings(
            agent_llm="gpt-4o-mini",
            agent_llm_config={
                "rate_limit": {
                    "gpt-4o-mini": "30000 per 1 minute",
                },
            }
        ),
        embedding="text-embedding-3-small",
        temperature=0.5,  # Keep deterministic
        paper_directory=PAPERS_DIR)
            )

        # Extract question and choices from the input messages
        user_message = next((msg["content"] for msg in sample["messages"] if msg["role"] == "user"), "")
            
         # Parse the question and choices from the user message
        question_match = re.search(r"Question:\s*(.*?)(?:\nChoices:|\Z)", user_message, re.DOTALL)
        question = question_match.group(1).strip() if question_match else user_message
        
        # Extract choices
        choices_text = ""
        choices_match = re.search(r"Choices:(.*?)$", user_message, re.DOTALL)
        if choices_match:
            choices_text = choices_match.group(1).strip()
    

        transcript().info(f"Starting paper search for '{question[:50]}...'")
        try:
             # Create AG2 agents
            researcher = AssistantAgent(
                name="researcher",
                system_message="""Your task is to analyze PaperQA2 responses to multiple-choice questions and determine which answer option 
                is being selected by the model. Be precise and objective.

                You will receive:
                1. A question with multiple-choice options
                2. The model's response

                For each response, you must:
                1. Analyze the content carefully
                2. Determine which option the response is selecting
                3. Return ONLY the single letter of the selected option (A, B, C, D, etc.)
                4. If the model is indicating it doesn't have enough information, select the "Insufficient information" option

                IMPORTANT: Your response must contain EXACTLY ONE LETTER, nothing else. 

                IMPORTANT RULES FOR NUMERICAL QUESTIONS:
            - When the question involves numerical values, pay attention to significant figures
            - If PaperQA2's response contains a numeric value with higher precision than the options (e.g., 45.67% vs 46%), 
            round to the same number of significant figures as in the options
            - Match to the closest option after appropriate rounding
            - If two options are equally close after rounding, choose the one that appears in the response
            - If the response is significantly different from all options, select the "Insufficient information" option
                Do not include explanations, punctuation, or any other text.""",
                # Simpler configuration without structured output initially
                llm_config = {
            "model": "gpt-4o-mini",
            "temperature": 0.1
            }
            )

            user_proxy = UserProxyAgent(
                name="user",
                human_input_mode="NEVER",
                max_consecutive_auto_reply=10,
                is_termination_msg=lambda x: True if "TERMINATE" in x.get("content", "") else False,
                code_execution_config={"use_docker": False}
            )

            
            # First collect evidence about the question
            evidence_query = f"Find evidence from scientific papers about: {question}"
            evidence = pqa_client.ask(evidence_query)

            # Format the task for AG2
            task_description = f"""
            Based on scientific evidence, answer this multiple choice question:
            
            Question: {question}
            
            Options:
            {choices_text}
            
            Evidence from papers:
            {evidence}
            
            Analyze the evidence and determine the correct answer option letter.
            """
            
            # Get structured answer from AG2
            transcript().info("Getting structured answer from AG2...")
            # Clear previous chat history
            user_proxy.reset()
            researcher.reset()

            # Get grader's response
            transcript().info(f"API Key available: {'Yes' if os.environ.get('OPENAI_API_KEY') else 'No'}")
            user_proxy.initiate_chat(
                    researcher, 
                    message=task_description,
                    max_turns=1
                )
            transcript().info(f"Chat completed successfully")

            
            # Extract the answer
            last_message = researcher.last_message()
            content = last_message["content"]

            transcript().info(f"Last message content type: {type(content)}")
            transcript().info(f"Last message content: {content}")
            
            # Check if we got a structured response
            if isinstance(content, MCAnswer):
                answer_letter = content.answer_letter.upper()
                letter_index = ord(answer_letter) - ord('A')
                
                transcript().info(f"AG2 structured answer: {answer_letter} ")
                return {
                    "output": answer_letter,
                    "choice_correct": letter_index
                }
            else:
                # Fallback if structured output fails
                answer_text = str(content)
                match = re.search(r"([A-Z])", answer_text, re.IGNORECASE)
                
                if match:
                    answer_letter = match.group(1).upper()
                    letter_index = ord(answer_letter) - ord('A')
                    
                    transcript().info(f"Extracted answer: {answer_letter}")
                    return {
                        "output": answer_letter,
                        "choice_correct": letter_index
                    }
                else:
                    transcript().info("Could not extract a letter answer")
                    return {"output": answer_text}
        
        except Exception as e:
            transcript().info(f"Error in AG2+PaperQA2 agent: {str(e)}")
            return {"output": str(e)}
    
    return run

    
@task( parameters={
    "model": "gpt-4o-mini",
    "evidence_k": 30,
    "max_sources": 15,
    "skip_summary": True,
    "dataset": "test"
})
def evaluate_paperqa2_custom(
    model: str = "gpt-4o-mini",
    evidence_k: int = 30,
    max_sources: int = 15,
    skip_summary: bool = True,
    dataset: str = "test"
):
    """Task to evaluate PaperQA2 on multiple choice biology questions"""
    dataset = json_dataset(QUESTIONS_FILE, record_to_sample)

    timestamp = datetime.datetime.now().strftime("%Y%m%d")
    experiment_name = f"paperqa2_{model}_evK{evidence_k}_maxS{max_sources}_skip{skip_summary}_dataset{dataset}_{timestamp}"
    
    return Task(
        dataset=dataset,
        solver=bridge(paperqa2_agent()),
        scorer=precision_choice(no_answer=UNSURE),
        epochs=Epochs(1, "mode"),
        metadata={"experiment_name": experiment_name}
    )

# # Keep the built-in multiple_choice solver task for comparison
# @task
# def evaluate_paperqa2_mc():
#     """Alternative task using built-in multiple_choice solver"""
#     dataset = json_dataset("/Users/apple/Documents/GitLab_Projects/master_project/xx823/Reproduction/Questions/formatted_questions_test/questions.jsonl", record_to_sample)
    
#     return Task(
#         dataset=dataset,
#         solver=[multiple_choice(template=MULTIPLE_CHOICE_TEMPLATE, cot=True)],
#         scorer=precision_choice(no_answer=UNCERTAIN_ANSWER_CHOICE),
#         epochs=Epochs(1, "mode"),
#     )