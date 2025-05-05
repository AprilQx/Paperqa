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

import json
import nest_asyncio
from shortuuid import uuid
from typing import Any, Dict


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



# Define the multiple choice template
MULTIPLE_CHOICE_TEMPLATE = """
The following is a multiple choice question about biology.
Answer the following multiple choice question about biology with ONLY the letter of the correct answer.

Think step by step about the information in the scientific papers available to you.
Use the provided paper content to determine the answer.

Question: {question}
Options:
{choices}

Your entire response must consist of exactly the format 'LETTER' where LETTER is the single correct option letter.
Do not include any explanations, reasoning, or additional text before or after.
Respond with ONLY 'A' or 'B' or 'C', etc.
"""

UNCERTAIN_ANSWER_CHOICE = "Insufficient information to answer the question."



def paperqa2_agent(template=MULTIPLE_CHOICE_TEMPLATE):
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
    paper_directory="/Users/apple/Documents/GitLab_Projects/master_project/xx823/papers" )

    
    index_built = False

    
    # Build the PaperQA2 document index
    async def build_index():
        nonlocal index_built
        if not index_built:
            transcript().info("Building PaperQA2 document index...")
            
            settings = Settings(
                paper_directory='/Users/apple/Documents/GitLab_Projects/master_project/xx823/papers',
                agent={"index": {
                    "sync_with_paper_directory": True,
                    "recurse_subdirectories": True
                }})
            
            built_index = await get_directory_index(settings=settings)

            # Print index information
            transcript().info(f"Using index: {settings.get_index_name()}")
            index_files = await built_index.index_files
            transcript().info(f"Indexed {len(index_files)} files")

                
            index_built = True
        return built_index
    
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
       
        # Format the question using the template
        formatted_question = template.format(
            question=question,
            choices=choices_text
        )
        transcript().info(f"Starting paper search for '{question[:50]}...'")
        try:
    
            # Call PaperQA2 with the formatted question
            nest_asyncio.apply()
            response_pqa = ask(formatted_question, settings=settings)
            answer_dict = response_pqa.dict()
            full_answer = answer_dict['session']['answer']
             # Log the full answer for debugging
            transcript().info(f"PaperQA2 response: {full_answer[:100]}...")

            # Extract just the letter answer from PaperQA2's response
            match = re.search(r"^([A-Z])$", full_answer.strip())
            # Corrected code:
            if match:
               answer_letter = match.group(1)
               transcript().info(f"Extracted answer: {answer_letter}")
               letter_index = ord(answer_letter) - ord('A')

               return {
                    "output": answer_letter,
                    "choice_correct": letter_index
                }
            else:
                transcript().info("Could not extract letter answer from response")
                return {"output": full_answer}  # Return actual answer text
            
                
        except Exception as e:
            transcript().info(f"Error querying PaperQA2: {str(e)}")
            return {"output": {e}}
    
    return run

    
@task
def evaluate_paperqa2_custom():
    """Task to evaluate PaperQA2 on multiple choice biology questions"""
    dataset = json_dataset("/Users/apple/Documents/GitLab_Projects/master_project/xx823/Reproduction/Questions/formatted_questions_test/questions.jsonl", record_to_sample)
    
    return Task(
        dataset=dataset,
        solver=bridge(paperqa2_agent()),
        scorer=precision_choice(no_answer=UNCERTAIN_ANSWER_CHOICE),
        epochs=Epochs(1, "mode"),
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