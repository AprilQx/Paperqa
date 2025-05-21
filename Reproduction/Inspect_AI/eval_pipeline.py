import csv
import re

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
from paperqa.settings import AnswerSettings

import json
import nest_asyncio
from shortuuid import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

# Import AutoGen components
import autogen
from autogen import UserProxyAgent,AssistantAgent
import datetime

from pydantic import BaseModel, Field


import os
# Set environment variable to disable Docker
os.environ["AUTOGEN_USE_DOCKER"] = "False"


#change the path of papers
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
PAPERS_DIR = os.path.join(PROJECT_ROOT, 'data', 'papers')
QUESTIONS_FILE = os.path.join(PROJECT_ROOT, 'data', 'Questions', 'formatted_questions_test', 'questions.jsonl')
#Multiple choice template for multiple choice in Instect AI

MULTIPLE_CHOICE_TEMPLATE = """
The following is a multiple choice question about biology.
Answer the following multiple choice question about biology with ONLY the letter of the correct answer.

Think step by step.
Think step by step about the information in the scientific papers available to you.
Use the provided paper content to determine the answer.

Question: {question}
Options:
{choices}

Your entire response must consist of exactly the format 'ANSWER: $LETTER' where $LETTER is the single correct option letter.
Your entire response must consist of exactly the format 'LETTER' where LETTER is the single correct option letter.
Do not include any explanations, reasoning, or additional text before or after.
Respond with ONLY 'ANSWER: A' or 'ANSWER: B' or 'ANSWER: C', etc.
Respond with ONLY 'A' or 'B' or 'C', etc.
"""

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




UNCERTAIN_ANSWER_CHOICE = "Insufficient information to answer the question"



def paperqa2_agent():
    """Agent function for PaperQA2 using the Bridge pattern"""
    # Initialize settings and state
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
        ]
        # "rate_limit": {
        #     "gpt-4o-mini": "30000 per 1 minute",
        # },
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
        evidence_skip_summary=False),
    agent=AgentSettings(
        agent_llm="gpt-4o-mini",
        agent_llm_config={
            "rate_limit": {
                "gpt-4o-mini": "30000 per 1 minute",
            },
        }
    ),
    embedding="text-embedding-3-large",
    temperature=0.5,  # Keep deterministic
    paper_directory=PAPERS_DIR)

    
    index_built = False

    
    # Build the PaperQA2 document index
    async def build_index():
        nonlocal index_built
        if not index_built:
            transcript().info("Building PaperQA2 document index...")
            
            settings = Settings(
                paper_directory=PAPERS_DIR,
                agent={"index": {
                    "sync_with_paper_directory": True,
                    "recurse_subdirectories": True
                }})
         # Check if directory exists
            if not os.path.exists(PAPERS_DIR):
                transcript().error(f"Papers directory not found: {PAPERS_DIR}")
                raise FileNotFoundError(f"Papers directory not found: {PAPERS_DIR}")
                
            # Print files in directory (for debugging)
            transcript().info(f"Files in {PAPERS_DIR}: {os.listdir(PAPERS_DIR)[:5]}")
            
            built_index = await get_directory_index(settings=settings)

            # Print index information
            transcript().info(f"Using index: {settings.get_index_name()}")
            index_files = await built_index.index_files
            transcript().info(f"Indexed {len(index_files)} files")
            index_built = True
        return built_index
    
    # Create AG2 config with structured output
    
    
    # The actual agent function that processes a sample
    async def run(sample:dict) -> dict:
        # Build index if not already built
        if not index_built:
            await build_index()

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
                llm_config = {
            "model": "gpt-4o-mini",
            "temperature": 0.1,
            "functions": [{"name": "answer", "parameters": MCAnswer.model_json_schema()}],
            "function_call": {"name": "answer"}}
            )

            user_proxy = UserProxyAgent(
                name="user",
                human_input_mode="NEVER",
                max_consecutive_auto_reply=10,
                is_termination_msg=lambda x: True if "TERMINATE" in x.get("content", "") else False,
                code_execution_config={"use_docker": False}
            )
            async def query_papers(query: str) -> str:
                """Query PaperQA2 for scientific evidence"""
                transcript().info(f"Searching papers for: {query[:50]}...")
                nest_asyncio.apply()
                response = ask(query, settings=settings)
                return response.dict()['session']['answer']
    
            researcher.register_function(
                function_map={"query_papers": query_papers}
            )
            
            # First collect evidence about the question
            evidence_query = f"Find evidence from scientific papers about: {question}"
            evidence = await query_papers(evidence_query)

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
            if "function_call" in last_message:
                # Extract from function call
                function_call = last_message["function_call"]
                if function_call.get("name") == "answer":
                    try:
                        args = json.loads(function_call.get("arguments", "{}"))
                        if "answer_letter" in args:
                            answer_letter = args["answer_letter"].upper()
                            letter_index = ord(answer_letter) - ord('A')
                            
                            transcript().info(f"Extracted answer from function call: {answer_letter}")
                            return {
                                "output": answer_letter,
                                "choice_correct": letter_index
                            }
                    except json.JSONDecodeError:
                        transcript().info("Failed to parse function arguments as JSON")
            
            # Get content (might be None)
            content = last_message.get("content")
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

    
@task(parameters={
    "model": "gpt-4o-mini",
    "evidence_k": 30,
    "max_sources": 15,
    "skip_summary": False,
    "dataset": "test"
})
def evaluate_paperqa2_custom(
    model: str = "gpt-4o-mini",
    evidence_k: int = 30,
    max_sources: int = 15,
    skip_summary: bool = False,
    dataset: str = "test"
):
    """Task to evaluate PaperQA2 on multiple choice biology questions"""
    dataset = json_dataset(QUESTIONS_FILE, record_to_sample)

    timestamp = datetime.datetime.now().strftime("%Y%m%d")
    experiment_name = f"paperqa2_{model}_evK{evidence_k}_maxS{max_sources}_skip{skip_summary}_dataset{dataset}_{timestamp}"
    # Add file to store results

    
    return Task(
        dataset=dataset,
        solver=bridge(paperqa2_agent()),
        scorer=precision_choice(no_answer=UNCERTAIN_ANSWER_CHOICE),
        epochs=Epochs(1, "mode"),
        metadata={"experiment_name": experiment_name}
    )

# # # Keep the built-in multiple_choice solver task for comparison
# @task
# def evaluate_paperqa2_mc():
#     """Alternative task using built-in multiple_choice solver"""
#     dataset = json_dataset(QUESTIONS_FILE, record_to_sample)

#     return Task(
#         dataset=dataset,
#         solver=[multiple_choice(template=MULTIPLE_CHOICE_TEMPLATE, cot=True)],
#         scorer=precision_choice(no_answer=UNCERTAIN_ANSWER_CHOICE),
#         epochs=Epochs(1, "mode"),
#     )