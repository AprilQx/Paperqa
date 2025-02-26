import os
import json
import time
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
import argparse
from autogen import AssistantAgent, UserProxyAgent



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
    Uses GPT-4o to grade PaperQA2 responses.
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        gpt4_model: str = "gpt-4o",
        temperature: float = 0.0,
        results_dir: str = "graded_results"
    ):
        """
        Initialize the grader.
        
        Args:
            openai_api_key: OpenAI API key (if None, will use from environment)
            gpt4_model: GPT-4 model to use (default: gpt-4o)
            temperature: Temperature for generation
            results_dir: Directory to save results
        """
        # Set API key
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        
        # Configure agents
        self.configure_agents(gpt4_model, temperature)
        
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
    
    def configure_agents(self, gpt4_model: str, temperature: float):
        """
        Configure AutoGen agents.
        
        Args:
            gpt4_model: GPT-4 model to use
            temperature: Temperature for generation
        """
        # Configure LLM
        config_list = [
            {
                "model": gpt4_model,
                "api_key": os.environ.get("OPENAI_API_KEY"),
                "temperature": temperature
            }
        ]
        
        # Create the grader agent
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
3. Return ONLY the letter of the selected option (A, B, C, D, etc.)
4. If the model is indicating it doesn't have enough information, select the "Insufficient information" option

Your output must be just the letter, nothing else.""",
            llm_config={"config_list": config_list}
        )
        
        # Create the judge agent that will compare with ground truth
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
3. Return ONLY the classification, nothing else.""",
            llm_config={"config_list": config_list}
        )
        
        # Create a human proxy agent
        self.human_proxy = UserProxyAgent(
            name="HumanProxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            is_termination_msg=lambda x: True if "TERMINATE" in x.get("content", "") else False
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

What option did PaperQA2 select? Return ONLY the letter.
"""
        
        # Get grader's response
        self.human_proxy.initiate_chat(
            self.grader_agent,
            message=grader_input
        )
        
        # Extract graded answer
        graded_answer = self.grader_agent.last_message()["content"].strip()
        
        # Clean up graded answer (in case it returned extra text)
        if len(graded_answer) > 1:
            # Try to extract just the letter
            letter_matches = [c for c in graded_answer if c.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
            if letter_matches:
                graded_answer = letter_matches[0].upper()
            else:
                # If no letter found, use the first character
                graded_answer = graded_answer[0].upper()
        
        # Compare with ground truth
        judge_input = f"""
Graded answer: {graded_answer}
Correct answer: {correct_answer}
Unsure option: {unsure_option}

Is this "correct", "incorrect", or "unsure"?
"""
        
        self.human_proxy.initiate_chat(
            self.judge_agent,
            message=judge_input
        )
        
        # Get judge's verdict
        verdict = self.judge_agent.last_message()["content"].strip().lower()
        
        # Clean up verdict
        if "correct" in verdict:
            grade = "correct"
        elif "unsure" in verdict:
            grade = "unsure"
        else:
            grade = "incorrect"
        
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
    
    def save_results(self, results: List[Dict[str, Any]], filename_prefix: str = "graded"):
        """
        Save the grading results to files.
        
        Args:
            results: List of grading results
            filename_prefix: Prefix for output filenames
        """
        # Create results directory if it doesn't exist
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
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
        
        df.to_csv(self.results_dir / f"{filename_prefix}_results.csv", index=False)
        
        # Calculate and save metrics
        metrics = self.calculate_metrics(results)
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(self.results_dir / f"{filename_prefix}_metrics.csv", index=False)
        
        # Save full results (including responses) to JSON
        with open(self.results_dir / f"{filename_prefix}_full.json", "w") as f:
            json.dump(
                {
                    "results": results,
                    "metrics": metrics
                },
                f, 
                indent=2
            )
        
        logger.info(f"Results saved to {self.results_dir}")
        
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
    # Grade the responses
    results = grader.grade_questions(response_data)
    
    # Save the results
    metrics = grader.save_results(results, filename_prefix=save_prefix)
    
    # Print the metrics
    print("\nGrading Results:")
    print(f"Total questions: {metrics['total_questions']}")
    print(f"Correct answers: {metrics['correct_answers']} ({metrics['accuracy']:.2%})")
    print(f"Incorrect answers: {metrics['incorrect_answers']} ({metrics['incorrect_answers']/metrics['total_questions']:.2%})")
    print(f"Unsure answers: {metrics['unsure_answers']} ({metrics['unsure_rate']:.2%})")
    print(f"Precision: {metrics['precision']:.2%}")
    
    return metrics
