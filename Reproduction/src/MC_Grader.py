
import os
import json
import time
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
import argparse

from autogen import AssistantAgent, UserProxyAgent

import re



# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("grader.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MultipleChoiceGrader:
    """
    Grader for multiple-choice questions using AutoGen agents.
    Uses GPT-3.5 to grade PaperQA2 responses.
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        gpt3_5_model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        results_dir: str = "graded_results"
    ):
        """
        Initialize the grader.
        
        Args:
            openai_api_key: OpenAI API key (if None, will use from environment)
            gpt3.5_model: GPT-3.5 model to use (default: gpt-3.5)
            temperature: Temperature for generation
            results_dir: Directory to save results
        """
        # Set API key
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        
        # Configure agents
        self.configure_agents(gpt3_5_model, temperature)
        
        # Set results directory
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize metrics
        self.metrics = {
            "total": 0,
            "correct": 0,
            "incorrect": 0,
            "unsure": 0
        }
    
    def configure_agents(self, gpt3_5_model: str, temperature: float):
        """
        Configure AutoGen agents.
        
        Args:
            gpt3.5_model: GPT-3.5 model to use
            temperature: Temperature for generation
        """
        # Configure LLM
        config_list = [
            {
                "model": gpt3_5_model,
                "api_key": os.environ.get("OPENAI_API_KEY"),
                "temperature": temperature
            }
        ]
        
        # Create the grader agent with a modified prompt to ensure proper formatting
        self.grader_agent = AssistantAgent(
            name="GraderAgent",
            system_message="""You are an expert grader for multiple-choice questions. 
    Your task is to analyze PaperQA2 responses to multiple-choice questions and determine which answer option 
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
    Do not include explanations, punctuation, or any other text.""",
            llm_config={"config_list": config_list}
        )
        
        # Create the judge agent with a modified prompt to directly accept the grader's output
        self.judge_agent = AssistantAgent(
            name="JudgeAgent",
            system_message="""You are a judge who determines if a graded answer matches the ground truth.
    You will be given:
    1. The grader's selected option (a letter)
    2. The ground truth answer (a letter)
    3. The "unsure" option (a letter)

    You must:
    1. Compare the graded answer with the ground truth
    2. Classify the result as one of: "correct", "incorrect", or "unsure"
    3. Return ONLY one of these three words: "correct", "incorrect", or "unsure"

    IMPORTANT: Your response must contain EXACTLY ONE WORD, either "correct", "incorrect", or "unsure".""",
            llm_config={"config_list": config_list}
        )
        
        # Create a human proxy agent
        self.human_proxy = UserProxyAgent(
            name="HumanProxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            is_termination_msg=lambda x: True if "TERMINATE" in x.get("content", "") else False,
            code_execution_config={"use_docker": False}
        )
        
    def grade_response(
    self, 
    question: str, 
    choices: List[str], 
    response: str, 
    correct_answer: str,
    unsure_option: str
) -> Dict[str, Any]:
        """
        Grade a single response using the grader agent.
        
        Args:
            question: The question text
            choices: List of options (A, B, C, etc.)
            response: PaperQA2's response
            correct_answer: The correct option letter
            unsure_option: The "insufficient information" option letter
            
        Returns:
            Dictionary with grading results
        """
        logger.info(f"Grading question: {question[:50]}...")

        # Format the input for the grader
        grader_input = f"""
    Question:
    {question}

    Options:
    {' '.join(choices)}

    PaperQA2 Response:
    {response}

    What option did PaperQA2 select? Remember to return ONLY the letter.
    """
         # Clear previous chat history to avoid confusion
        self.human_proxy.reset()
        self.grader_agent.reset()
        self.judge_agent.reset()
        
        # Get grader's response
        self.human_proxy.initiate_chat(
            self.grader_agent,
            message=grader_input,
            max_turns=1
        )
        
        # Extract graded answer - should be a single letter now
        graded_answer = self.grader_agent.last_message()["content"].strip()
        
        # Compare with ground truth
        judge_input = f"""
    Graded answer: {graded_answer}
    Correct answer: {correct_answer}
    Unsure option: {unsure_option}

    Is this "correct", "incorrect", or "unsure"? Return ONLY one word.
    """
        
        self.human_proxy.initiate_chat(
            self.judge_agent,
            message=judge_input,
            max_turns=1
        )
        
        # Get judge's verdict - should be just one word now
        verdict = self.judge_agent.last_message()["content"].strip().lower()

         
    # Make sure we have one of the expected verdicts
        if verdict not in ["correct", "incorrect", "unsure"]:
            logger.warning(f"Unexpected verdict: {verdict}, defaulting to 'incorrect'")
            if "correct" in verdict:
                grade = "correct"
            elif "unsure" in verdict:
                grade = "unsure"
            else:
                grade = "incorrect"
        else:
            grade = verdict
    
    # Update metrics
        self.metrics["total"] += 1
        self.metrics[grade] += 1
            
        # Update metrics
        self.metrics["total"] += 1
        self.metrics[grade] += 1
        
        return {
            "question": question,
            "choices": choices,
            "response": response,
            "correct_answer": correct_answer,
            "unsure_option": unsure_option,
            "graded_answer": graded_answer,
            "grade": grade
        }
    def grade_questions(self, questions_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Grade a list of questions.
        
        Args:
            questions_data: List of question dictionaries with responses
            
        Returns:
            List of grading results
        """
        results = []
        
        for i, q in enumerate(questions_data):
            logger.info(f"Processing question {i+1}/{len(questions_data)}")
            
            # Extract the data needed for grading
            question = q["question"]
            choices = q["choices"]
            response = q["response"]
            correct_answer = q["correct_answer"]
            unsure_option = q["unsure_option"]
            
            # Grade the response
            result = self.grade_response(
                question=question,
                choices=choices,
                response=response,
                correct_answer=correct_answer,
                unsure_option=unsure_option
            )
            
            # Add to results
            results.append(result)
            
            # Add a small delay to avoid rate limits
            time.sleep(1)
        
        return results
    
    def calculate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate metrics from grading results.
        
        Args:
            results: List of grading results
            
        Returns:
            Dictionary with metrics
        """
        total = len(results)
        correct = sum(1 for r in results if r["grade"] == "correct")
        unsure = sum(1 for r in results if r["grade"] == "unsure")
        incorrect = sum(1 for r in results if r["grade"] == "incorrect")
        answered = total - unsure
        
        metrics = {
            "total_questions": total,
            "correct_answers": correct,
            "incorrect_answers": incorrect,
            "unsure_answers": unsure,
            "answered_questions": answered,
            "accuracy": correct / total if total > 0 else 0,
            "precision": correct / answered if answered > 0 else 0,
            "unsure_rate": unsure / total if total > 0 else 0
        }
        
        return metrics
    

def load_response_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Load PaperQA2 responses from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        List of question dictionaries with responses
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    return data

def process_results(grader, response_data, save_prefix="paperqa2"):
    """
    Process PaperQA2 results.
    
    Args:
        grader: The MultipleChoiceGrader instance
        response_data: List of response data dictionaries
        save_prefix: Prefix for saved files
        
    Returns:
        The calculated metrics
    """
    # First, we need to make the input data serializable
    serializable_input = []
    for item in response_data:
        # Make a copy of the item
        input_item = dict(item)
        
        # Check what type of response object we have
        response_obj = item["response"]
        
        # Extract the answer text based on the object type
        if hasattr(response_obj, "answer") and isinstance(response_obj.answer, str):
            # If it has a direct 'answer' attribute, use that
            response_text = response_obj.answer
        elif hasattr(response_obj, "model_dump"):
            # If it's a Pydantic model, dump it and extract the answer
            dump = response_obj.model_dump()
            response_text = dump.get("answer", str(dump))
        elif hasattr(response_obj, "dict") and callable(response_obj.dict):
            # For older Pydantic versions
            dump = response_obj.dict()
            response_text = dump.get("answer", str(dump))
        else:
            # Fallback to string conversion
            response_text = str(response_obj)
            # Replace the response object with the extracted text
        input_item["response"] = response_text
        serializable_input.append(input_item)

    # Grade the responses
    results = grader.grade_questions(response_data)
    
    # Save the results
    metrics = grader.calculate_metrics(results)


    # Save the results manually to ensure serializability
    results_dir = grader.results_dir
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Save detailed results to CSV
    df = pd.DataFrame([
        {
            "question": r["question"][:100] + "..." if len(r["question"]) > 100 else r["question"],
            "correct_answer": r["correct_answer"],
            "graded_answer": r["graded_answer"],
            "grade": r["grade"]
        }
        for r in results
    ])
    
    df.to_csv(results_dir / f"{save_prefix}_results.csv", index=False)
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(results_dir / f"{save_prefix}_metrics.csv", index=False)
    
    # Prepare for JSON serialization
    serializable_results = []
    for r in results:
        # Create a serializable version of each result
        serializable_r = {}
        for key, value in r.items():
            # Skip the response to keep file size manageable
            if key == "response":
                serializable_r[key] = str(value)[:1000] + "..." if len(str(value)) > 1000 else str(value)
            else:
                serializable_r[key] = value
        serializable_results.append(serializable_r)
    
    # Save to JSON
    with open(results_dir / f"{save_prefix}_full.json", "w") as f:
        json.dump(
            {
                "results": serializable_results,
                "metrics": metrics
            },
            f, 
            indent=2
        )
    
    # Print the metrics
    print("\nGrading Results:")
    print(f"Total questions: {metrics['total_questions']}")
    print(f"Correct answers: {metrics['correct_answers']} ({metrics['accuracy']:.2%})")
    print(f"Incorrect answers: {metrics['incorrect_answers']} ({metrics['incorrect_answers']/metrics['total_questions']:.2%})")
    print(f"Unsure answers: {metrics['unsure_answers']} ({metrics['unsure_rate']:.2%})")
    print(f"Precision: {metrics['precision']:.2%}")
    
    return metrics
