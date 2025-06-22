import os
import re
import json
import autogen
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
#from Reproduction.Pipeline_reproduction.config import OPENAI_API_KEY
from openai import OpenAI


class ScientificQAEvaluator:
    """
    A class to process scientific multiple-choice questions using AutoGen,
    extract answers, and evaluate performance.
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.2,
        results_dir: str = "evaluation_results_49MCs_gpt4o"
    ):
        """
        Initialize the evaluator.
        
        Args:
            model_name: The LLM model to use
            temperature: Temperature for generation
            results_dir: Directory to save results
        """
        # Set up results directory
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        # Model configuration
        self.model_name = model_name
        self.temperature = temperature
        
        # Initialize agents
        self.assistant, self.user_proxy = self._create_agents()
        
        # Container for results
        self.results = []
    
    def _create_agents(self):
        """Create AutoGen agents for answering questions."""
        config_list = [
            {
                "model": self.model_name,
                "api_key": os.environ.get("OPENAI_API_KEY"),
                "temperature": self.temperature
            }
        ]
        
        system_message = """You are a scientific research assistant that answers multiple-choice questions 
        based on your knowledge. You will receive scientific multiple-choice questions and you should:
        
        1. Analyze the question carefully
        2. Consider all the provided options
        3. Use your knowledge to identify the most likely correct answer
        4. If you truly don't have sufficient information to answer confidently, select the "Insufficient information" option
        5. Provide a clear, concise explanation for your choice
        6. End your response by clearly stating your final answer choice with EXACTLY this format: "Final Answer: X" where X is the letter of your chosen option
        
        Be precise and analytical in your approach.
        """
        
        assistant = autogen.AssistantAgent(
            name="ScientificMCAgent",
            system_message=system_message,
            llm_config={"config_list": config_list}
        )
        
        user_proxy = autogen.UserProxyAgent(
            name="UserProxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=0,
            code_execution_config=False
        )
        
        return assistant, user_proxy
    
    def _format_question(self, question: str, choices: List[str]) -> str:
        """Format a question with its choices for the agent."""
        formatted = f"Question: {question}; Options: "
        formatted += " ".join(choices)
        return formatted
    
    def _extract_answer(self, response: str) -> str:
        """Extract the answer letter from the agent's response."""
        # Try the main pattern: "Final Answer: X"
        match = re.search(r"Final Answer:\s*([A-I])", response, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        # Try alternative patterns
        secondary_match = re.search(r"\b([A-I])[\.:\)]", response, re.IGNORECASE)
        if secondary_match:
            return secondary_match.group(1).upper()
        
        option_words = re.search(r"(choose|select|pick|answer is)[\s:]*(option\s*)?([A-I])", 
                                 response, re.IGNORECASE)
        if option_words:
            return option_words.group(3).upper()
        
        print(f"WARNING: Could not extract answer from response: {response[:100]}...")
        return "MANUAL_REVIEW_REQUIRED"
    
    def process_single_question(self, question_data: Dict[str, Any], max_retries: int = 1) -> Dict[str, Any]:
        """
        Process a single question and get the agent's response.
        
        Args:
            question_data: Dictionary containing question, choices, correct_answer, unsure_option
            max_retries: Maximum number of retries if answer extraction fails
            
        Returns:
            Dictionary with question data and response
        """
        # Format the question
        formatted_question = self._format_question(
            question_data["question"], 
            question_data["choices"]
        )
        
        # Get response with retries if needed
        retries = 0
        while retries <= max_retries:
            # Reset agents for new conversation
            self.assistant.reset()
            self.user_proxy.reset()
            
            # Start the conversation with the formatted question
            message = f"Please answer this multiple-choice question: {formatted_question}"
            if retries > 0:
                message += "\n\nIMPORTANT: You MUST end your response with 'Final Answer: X' where X is the letter of your chosen option (A, B, C, or D)."
            
            self.user_proxy.initiate_chat(
                self.assistant,
                message=message
            )
            
            # Get the agent's response
            full_response = self.assistant.last_message()["content"]
            
            # Extract the final answer
            selected_answer = self._extract_answer(full_response)
            
            if selected_answer != "MANUAL_REVIEW_REQUIRED":
                break
            
            retries += 1
            print(f"Retrying question... (attempt {retries}/{max_retries})")
        
        # Prepare the result dictionary
        result = {
            "question": question_data["question"],
            "choices": question_data["choices"],
            "correct_answer": question_data["correct_answer"],
            "unsure_option": question_data["unsure_option"],
            "full_response": full_response,
            "selected_answer": selected_answer
        }
        
        # Determine result status
        if selected_answer == "MANUAL_REVIEW_REQUIRED":
            result["result_status"] = "MANUAL_REVIEW"
        elif selected_answer == result["correct_answer"]:
            result["result_status"] = "CORRECT"
        elif selected_answer == result["unsure_option"]:
            result["result_status"] = "UNSURE"
        else:
            result["result_status"] = "INCORRECT"
        
        print(f"Question processed. Status: {result['result_status']}")
        if selected_answer != "MANUAL_REVIEW_REQUIRED":
            print(f"Selected: {selected_answer}, Correct: {result['correct_answer']}")
            
        return result
    
    def process_questions(self, questions: List[Dict[str, Any]]) -> None:
        """
        Process multiple questions and store the results.
        
        Args:
            questions: List of question dictionaries
        """
        total_questions = len(questions)
        self.results = []
        
        for i, question in enumerate(questions):
            current_num = i + 1
            print(f"\nProcessing question {current_num}/{total_questions}")
            
            # Process the question
            result = self.process_single_question(question)
            
            # Store the result
            self.results.append(result)
            
            # Show progress
            print(f"Completed {current_num}/{total_questions}, {total_questions - current_num} remaining\n")
        
        print(f"All {total_questions} questions processed successfully")
    
    def calculate_metrics(self) -> Dict[str, Any]:
        """
        Calculate accuracy and precision metrics from processed results.
        
        Returns:
            Dictionary with calculated metrics
        """
        if not self.results:
            print("No results available to calculate metrics")
            return {}
        
        total = len(self.results)
        correct = sum(1 for r in self.results if r["result_status"] == "CORRECT")
        unsure = sum(1 for r in self.results if r["result_status"] == "UNSURE")
        manual_review = sum(1 for r in self.results if r["result_status"] == "MANUAL_REVIEW")
        incorrect = total - correct - unsure - manual_review
        
        # Calculate metrics
        answered = total - unsure - manual_review
        
        metrics = {
            "total_questions": total,
            "correct_answers": correct,
            "incorrect_answers": incorrect,
            "unsure_answers": unsure,
            "manual_review_required": manual_review,
            "answered_questions": answered,
            "accuracy": correct / total if total > 0 else 0,
            "precision": correct / answered if answered > 0 else 0,
            "unsure_rate": unsure / total if total > 0 else 0
        }
        
        return metrics
    
    def display_metrics(self, metrics: Optional[Dict[str, Any]] = None) -> None:
        """
        Display the calculated metrics.
        
        Args:
            metrics: Optional metrics dictionary (if None, will calculate)
        """
        if metrics is None:
            metrics = self.calculate_metrics()
        
        if not metrics:
            return
        
        print("\n===== EVALUATION METRICS =====")
        print(f"Total questions: {metrics['total_questions']}")
        print(f"Correct answers: {metrics['correct_answers']} ({metrics['accuracy']:.2%})")
        print(f"Incorrect answers: {metrics['incorrect_answers']} ({metrics['incorrect_answers']/metrics['total_questions']:.2%})")
        print(f"Unsure answers: {metrics['unsure_answers']} ({metrics['unsure_rate']:.2%})")
        print(f"Manual review needed: {metrics['manual_review_required']} ({metrics['manual_review_required']/metrics['total_questions']:.2%})")
        print(f"Precision: {metrics['precision']:.2%}")
        print(f"Accuracy: {metrics['accuracy']:.2%}")
    
    def save_results(self, experiment_name: Optional[str] = None) -> Dict[str, str]:
        """
        Save the results and metrics to files.
        
        Args:
            experiment_name: Optional name for the experiment (default: timestamp)
            
        Returns:
            Dictionary with paths to saved files
        """
        if not self.results:
            print("No results to save")
            return {}
        
        # Generate filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_prefix = f"{experiment_name}_{timestamp}" if experiment_name else timestamp
        
        results_file = self.results_dir / f"{name_prefix}_results.json"
        metrics_file = self.results_dir / f"{name_prefix}_metrics.json"
        
        # Calculate metrics
        metrics = self.calculate_metrics()
        
        # Save results
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)
        
        # Save metrics
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\nResults saved to {results_file}")
        print(f"Metrics saved to {metrics_file}")
        
        return {
            "results_file": str(results_file),
            "metrics_file": str(metrics_file)
        }
