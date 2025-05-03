import csv
import re
from inspect_ai import Task, task, Epochs
from inspect_ai.dataset import Dataset, Sample
from scorer import precision_choice
from inspect_ai.solver import generate, multiple_choice, solver
from inspect_ai.scorer import exact 
import sys
import os
import asyncio
from paperqa import  ask
from paperqa.agents.main import agent_query
from paperqa.agents.search import get_directory_index
from paperqa.settings import Settings, AgentSettings
import json
import nest_asyncio
from shortuuid import uuid
from inspect_ai.util._limited_conversation import ChatMessageList
from inspect_ai.dataset import Sample, json_dataset
from inspect_ai.solver._task_state import TaskState
from inspect_ai.model import ModelOutput, StopReason
from inspect_ai.solver import generate
def record_to_sample(record):
    # Create formatted input with question and choices
    choices_list = record["choices"]
    choices_text = "\n".join(record["choices"])
    formatted_input = f"Question: {record['question']}\nChoices:\n{choices_text}"
    
    return Sample(
        input=formatted_input,  # Include both question and choices
        target=record["correct_answer"].strip(),
        # The choices field needs to be properly structured for multiple_choice solver
        choices=choices_list,
        metadata={
            "sources": record["sources"],
            "unsure_option": record.get("unsure_option", "")
        }
    )




# Define the multiple choice template
MULTIPLE_CHOICE_TEMPLATE = """
The following is a multiple choice question about biology.
Answer the following multiple choice question about biology with ONLY the letter of the correct answer.

Think step by step.

Question: {question}
Options:
{choices}

Your entire response must consist of exactly the format 'ANSWER: $LETTER' where $LETTER is the single correct option letter.
Do not include any explanations, reasoning, or additional text before or after.
Respond with ONLY 'ANSWER: A' or 'ANSWER: B' or 'ANSWER: C', etc.
"""

UNCERTAIN_ANSWER_CHOICE = "Insufficient information to answer the question."


@solver
def paperqa2_multiple_choice_solver(template=MULTIPLE_CHOICE_TEMPLATE):
    """Function-based solver that uses PaperQA2 to answer multiple choice questions."""
    # Initialize settings and state
    settings = Settings(
        llm="gpt-4o-mini",
        # Your settings configuration...
        paper_directory="/Users/apple/Documents/GitLab_Projects/master_project/xx823/papers"
    )
    
    index_built = False
    built_index = None
    
    # Build the PaperQA2 document index
    async def build_index():
        nonlocal index_built, built_index
        if not index_built:
            print("Building PaperQA2 document index...")
            
            settings = Settings(
                paper_directory='/Users/apple/Documents/GitLab_Projects/master_project/xx823/papers',
                agent={"index": {
                    "sync_with_paper_directory": True,
                    "recurse_subdirectories": True
                }})
            
            built_index = await get_directory_index(settings=settings)

            # Print index information
            print(f"Using index: {settings.get_index_name()}")
            index_files = await built_index.index_files
            print(f"Number of indexed files: {len(index_files)}")
            print("Indexed files:")
            for file in index_files:
                print(f"- {file}")
                
            index_built = True
    
    # The actual solver function
    async def solve(state: TaskState, generate) -> TaskState:
        # Build index if not already built
        if not index_built:
            await build_index()
            
        # Process problem - extract the question from state
        question = state.input_text.split("\n")[0].replace("Question: ", "").strip()
        choices = state.choices
            
        # Format the question using the template
        formatted_question = template.format(
            question=question,
            choices=choices
        )
        
        try:
            # Update the user message with the formatted question
            if state.messages:
                for i, msg in enumerate(state.messages):
                    if hasattr(msg, 'role') and msg.role == 'user':
                        state.messages[i].content = formatted_question
                        break
            
            # Call PaperQA2 with the formatted question
            nest_asyncio.apply()
            response_pqa = ask(formatted_question, settings=settings)
            answer_dict = response_pqa.dict()
            full_answer = answer_dict['session']['answer']

            # Extract just the letter answer from PaperQA2's response
            match = re.search(r"ANSWER:\s*([A-Z])", full_answer)
            answer_letter = match.group(1) if match else full_answer
            
            # Instead of returning a custom object, use generate to properly update state
            # This will add the assistant message and set the output
            from inspect_ai.util import ChatMessageAssistant
            state.messages.append(ChatMessageAssistant(content=f"ANSWER: {answer_letter}"))
            
            # Set the output directly if you don't want to use generate
            state.output = ModelOutput(
                completion=f"ANSWER: {answer_letter}",
                stop_reason=StopReason.EOS
            )
            
            return state

        except Exception as e:
            print(f"Error querying PaperQA2: {str(e)}")
            # Still return the state even in case of error
            state.output = ModelOutput(
                completion=f"Error: {str(e)}",
                stop_reason=StopReason.ERROR
            )
            return state
    
    return solve
    
@task
def evaluate_paperqa2_custom():
    """Task to evaluate PaperQA2 on multiple choice biology questions"""
    dataset = json_dataset("/Users/apple/Documents/GitLab_Projects/master_project/xx823/Reproduction/Questions/formatted_questions_test/questions.jsonl", record_to_sample)
    
    return Task(
        dataset=dataset,
        solver=[paperqa2_multiple_choice_solver()],
        scorer=precision_choice(no_answer=UNCERTAIN_ANSWER_CHOICE),
        epochs=Epochs(1, "mode"),
    )

# Keep the built-in multiple_choice solver task for comparison
@task
def evaluate_paperqa2_mc():
    """Alternative task using built-in multiple_choice solver"""
    dataset = json_dataset("/Users/apple/Documents/GitLab_Projects/master_project/xx823/Reproduction/Questions/formatted_questions_test/questions.jsonl", record_to_sample)
    
    return Task(
        dataset=dataset,
        solver=[multiple_choice(template=MULTIPLE_CHOICE_TEMPLATE, cot=True)],
        scorer=precision_choice(no_answer=UNCERTAIN_ANSWER_CHOICE),
        epochs=Epochs(1, "mode"),
    )