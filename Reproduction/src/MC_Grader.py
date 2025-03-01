
import os
import json
import time
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging

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

    IMPORTANT RULES FOR NUMERICAL QUESTIONS:
- When the question involves numerical values, pay attention to significant figures
- If PaperQA2's response contains a numeric value with higher precision than the options (e.g., 45.67% vs 46%), 
  round to the same number of significant figures as in the options
- Match to the closest option after appropriate rounding
- If two options are equally close after rounding, choose the one that appears in the response
- If the response is significantly different from all options, select the "Insufficient information" option
    Do not include explanations, punctuation, or any other text.""",
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
        
    def extract_answer(
        self, 
        question: str, 
        choices: List[str], 
        response: str, 
        correct_answer: str,
        unsure_option: str
    ) -> Dict[str, Any]:
        """
        Extract the selected answer from PaperQA2's response.
        
        Args:
            question: The question text
            choices: List of options (A, B, C, etc.)
            response: PaperQA2's response
            correct_answer: The correct option letter
            unsure_option: The "insufficient information" option letter
            
        Returns:
            Dictionary with extraction results
        """
        logger.info(f"Processing question: {question[:50]}...")

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
        # Clear previous chat history
        self.human_proxy.reset()
        self.grader_agent.reset()

        # Get grader's response
        self.human_proxy.initiate_chat(
            self.grader_agent,
            message=grader_input,
            max_turns=1
        )
        
        # Extract selected answer - should be a single letter
        selected_answer = self.grader_agent.last_message()["content"].strip()
        
        return {
            "question": question,
            "choices": choices,
            "response": response,
            "correct_answer": correct_answer,
            "unsure_option": unsure_option,
            "selected_answer": selected_answer
        }
    
    def process_questions(self, questions_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a list of questions and extract selected answers.
        
        Args:
            questions_data: List of question dictionaries with responses
            
        Returns:
            List of extraction results
        """
        results = []
        
        for i, q in enumerate(questions_data):
            logger.info(f"Processing question {i+1}/{len(questions_data)}")
            
            # Extract the data needed
            question = q["question"]
            choices = q["choices"]
            response = q["response"]
            correct_answer = q["correct_answer"]
            unsure_option = q["unsure_option"]
            
            # Extract the selected answer
            result = self.extract_answer(
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
    
    def save_results(self, results: List[Dict[str, Any]], save_prefix: str = "paperqa2"):
        """
        Save the extraction results to files for manual verification.
        
        Args:
            results: List of extraction results
            save_prefix: Prefix for saved files
        """
        # Save the results
        results_dir = self.results_dir
        
        # Save detailed results to CSV for easy viewing
        df = pd.DataFrame([
            {
                "question": r["question"][:100] + "..." if len(r["question"]) > 100 else r["question"],
                "correct_answer": r["correct_answer"],
                "selected_answer": r["selected_answer"],
                "unsure_option": r["unsure_option"]
            }
            for r in results
        ])
        
        df.to_csv(results_dir / f"{save_prefix}_extracted_answers.csv", index=False)
        
        # Prepare for JSON serialization (complete data)
        serializable_results = []
        for r in results:
            # Create a serializable version of each result
            serializable_r = {}
            for key, value in r.items():
                # Truncate response to keep file size manageable
                if key == "response":
                    serializable_r[key] = str(value)[:1000] + "..." if len(str(value)) > 1000 else str(value)
                else:
                    serializable_r[key] = value
            serializable_results.append(serializable_r)
        
        # Save to JSON for manual verification and metrics calculation
        with open(results_dir / f"{save_prefix}_extraction_results.json", "w") as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to {results_dir}")
        print(f"\nResults saved to {results_dir}")
        print(f"- CSV: {save_prefix}_extracted_answers.csv")
        print(f"- JSON: {save_prefix}_extraction_results.json")
        print("\nYou can now manually verify the results and calculate metrics.")
    

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
    results = grader.process_questions(response_data)
    
     # Save the results
    json_path = grader.results_dir / f"{save_prefix}_extraction_results.json"
    grader.save_results(results, save_prefix)
    
    return str(json_path)


def calculate_metrics_from_file(file_path: str, save_to_csv: bool = True) -> Dict[str, float]:
    """
    Calculate metrics from the extraction results file.
    This can be run separately after manual verification.
    
    Args:
        file_path: Path to the JSON file with extraction results
        save_to_csv: Whether to save metrics to a CSV file
        
    Returns:
        Dictionary with metrics
    """
    with open(file_path, 'r') as f:
        results = json.load(f)
    
    total = len(results)
    correct = sum(1 for r in results if r["selected_answer"] == r["correct_answer"])
    unsure = sum(1 for r in results if r["selected_answer"] == r["unsure_option"])
    incorrect = total - correct - unsure
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
    
    # Print the metrics
    print("\nMetrics:")
    print(f"Total questions: {metrics['total_questions']}")
    print(f"Correct answers: {metrics['correct_answers']} ({metrics['accuracy']:.2%})")
    print(f"Incorrect answers: {metrics['incorrect_answers']} ({metrics['incorrect_answers']/metrics['total_questions']:.2%})")
    print(f"Unsure answers: {metrics['unsure_answers']} ({metrics['unsure_rate']:.2%})")
    print(f"Precision: {metrics['precision']:.2%}")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    
    # Save metrics to CSV if requested
    if save_to_csv:
        metrics_df = pd.DataFrame([metrics])
        csv_path = Path(file_path).parent / "metrics.csv"
        metrics_df.to_csv(csv_path, index=False)
        print(f"Metrics saved to {csv_path}")
    
    return metrics
