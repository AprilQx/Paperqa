import pandas as pd
import time
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple, Union
from pydantic import BaseModel, Field
import autogen
from autogen import UserProxyAgent, AssistantAgent
import tiktoken
# Use the new Google Gen AI SDK
import google.generativeai as genai
from google.oauth2 import service_account

class ResponseEvaluation(BaseModel):
    """Structured evaluation of a scientific response"""
    accuracy_score: int = Field(
        description="Factual accuracy score (0-100), where 100 means perfectly matching the ideal answer", 
        ge=0, le=100
    )

    rationale: str = Field(
        description="Brief explanation of the evaluation scores and comparison with ideal answer"
    )

class AIEvaluator:
    """AI Evaluator for RAG system performance using AG2 (AutoGen 2)"""
    
    def __init__(self, model: str = "gpt-4o", temperature: float = 0.1):
        """Initialize the AI Evaluator"""
        self.model = model
        self.temperature = temperature
        self.ai_judge = self._create_ai_judge_agent()
        self.user_proxy = self._create_user_proxy_agent()
        
        # Disable Docker for AutoGen
        os.environ["AUTOGEN_USE_DOCKER"] = "False"
    
    def _create_ai_judge_agent(self) -> AssistantAgent:
        """Create and configure the AI judge agent"""
        system_message = """You are an expert scientific evaluator assessing the quality of scientific response against reference answers.

Your task is to evaluate responses using one critical criterion:

ACCURACY (0-100):
CRITICAL: Use ONLY these two scores for accuracy:
- 100: The answer contains the core correct factual content, concepts, and conclusions from the ideal answer
- 0: The answer is fundamentally wrong or contradicts the ideal answer

This is a BINARY evaluation - either the answer is essentially correct (100) or incorrect (0).
No partial credit or intermediate scores allowed.

EVALUATION GUIDELINES:
- Focus ONLY on whether the main scientific concepts and conclusions are correct
- Check that the core factual claims from the ideal answer are present in the generated answer
- Verify the overall conceptual direction and main conclusions align
- Additional correct information beyond the ideal answer is acceptable
- Only award 0 if the answer contradicts the ideal answer or gets the main concepts completely wrong
- Award 100 if the answer captures the essential correct scientific understanding

Provide your evaluation using the evaluate_response function with the numerical score and detailed rationale explaining why you chose 100 or 0."""
        
        return AssistantAgent(
            name="ai_judge",
            system_message=system_message,
            llm_config={
                "model": self.model,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "evaluate_response",
                            "description": "Evaluate a scientific response",
                            "parameters": ResponseEvaluation.model_json_schema()
                        }
                    }
                ]
            }
        )
    
    def _create_user_proxy_agent(self) -> UserProxyAgent:
        """Create and configure the user proxy agent"""
        return UserProxyAgent(
            name="user",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1,
            is_termination_msg=lambda x: True,
            code_execution_config={"use_docker": False}
        )
    
    def evaluate_single_response(
        self,
        question: str,
        generated_answer: str,
        ideal_answer: str,
        sources: Optional[List[str]] = None,
        system_name: str = "RAG System"
    ) -> Dict[str, Any]:
        """Evaluate a single RAG response"""
        
        # Format sources for citation evaluation
        # sources_text = "\n".join(sources) if sources else "No sources provided"
        
        evaluation_task = f"""
Please evaluate this system's response against the ideal answer:

QUESTION: {question}

GENERATED ANSWER:
{generated_answer}

IDEAL ANSWER:
{ideal_answer}


Evaluate based on:
Accuracy (0-100): How factually correct is the answer compared to the ideal?

Use the evaluate_response function to provide your structured evaluation with detailed rationale.
"""
        
        try:
            # Reset agents for fresh evaluation
            self.user_proxy.reset()
            self.ai_judge.reset()
            
            # Initiate the evaluation chat
            self.user_proxy.initiate_chat(
                self.ai_judge,
                message=evaluation_task,
                max_turns=1
            )
            
            # Extract evaluation results
            last_message = self.ai_judge.last_message()
            evaluation_result = None
            
            if last_message and "tool_calls" in last_message:
                tool_calls = last_message["tool_calls"]
                if tool_calls and len(tool_calls) > 0:
                    tool_call = tool_calls[0]
                    if tool_call.get("function", {}).get("name") == "evaluate_response":
                        try:
                            evaluation_result = json.loads(tool_call["function"].get("arguments", "{}"))
                        except json.JSONDecodeError as e:
                            print(f"Failed to parse evaluation: {e}")
                            return self._create_error_result(f"Parse error: {e}")
            
            # Fallback to old function_call format for compatibility
            elif last_message and "function_call" in last_message:
                function_call = last_message["function_call"]
                if function_call.get("name") == "evaluate_response":
                    try:
                        evaluation_result = json.loads(function_call.get("arguments", "{}"))
                    except json.JSONDecodeError as e:
                        print(f"Failed to parse evaluation: {e}")
                        return self._create_error_result(f"Parse error: {e}")
            
            if evaluation_result:
                # Calculate composite score
                accuracy_score = evaluation_result.get('accuracy_score', 0)
            
                
                return {
                    "eval_accuracy_score": accuracy_score,
                    "eval_rationale": evaluation_result.get('rationale', ''),
                    "eval_successful": True,
                    "eval_error": None
                }
            else:
                return self._create_error_result("No evaluation result obtained")
                
        except Exception as e:
            return self._create_error_result(f"Evaluation failed: {str(e)}")
    
    def _create_error_result(self, error_msg: str) -> Dict[str, Any]:
        """Create a standardized error result"""
        return {
            "eval_accuracy_score": None,
            "eval_rationale": None,
            "eval_successful": False,
            "eval_error": error_msg
        }

class GeminiEvaluator:
    """AI Evaluator for RAG system performance using the NEW Google Gen AI SDK"""
    
    def __init__(self, 
                 model: str = "gemini-2.5-pro-preview-06-05", 
                 temperature: float = 0.01,
                 max_retries: int = 3,
                 base_sleep_time: int = 60,
                 max_sleep_time: int = 300,
                 credentials_path: str = "gemini.json",
                 backoff_multiplier: float = 2.0,
                 max_tokens_per_minute: int = 2000000,
                 max_requests_per_minute: int = 150):
        """Initialize the Gemini Evaluator with rate limiting using NEW SDK"""
        self.model_name = model
        self.temperature = temperature
        self.credentials_path = credentials_path
        
        # Rate limiting configuration
        self.max_retries = max_retries
        self.base_sleep_time = base_sleep_time
        self.max_sleep_time = max_sleep_time
        self.backoff_multiplier = backoff_multiplier
        self.max_tokens_per_minute = max_tokens_per_minute
        self.max_requests_per_minute = max_requests_per_minute
        
        # Rate limiting state
        self._token_bucket = 0
        self._bucket_start_time = 0
        self._request_count = 0
        self._request_start_time = 0
        
        # Initialize Gemini with NEW SDK
        self._initialize_gemini()
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken encoding."""
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception as e:
            print(f"Warning: Could not count tokens: {e}")
            # Fallback: rough estimate of 1 token per 4 characters
            return len(text) // 4
    
    def _check_rate_limits(self, text: str):
        """Check and enforce comprehensive rate limiting"""
        tokens = self._count_tokens(text)
        now = time.time()
        
        # Token-based rate limiting
        if now - self._bucket_start_time > 60 or self._bucket_start_time == 0:
            self._token_bucket = 0
            self._bucket_start_time = now
        
        # Request-based rate limiting
        if now - self._request_start_time > 60 or self._request_start_time == 0:
            self._request_count = 0
            self._request_start_time = now
        
        # Check token quota
        if self._token_bucket + tokens > self.max_tokens_per_minute:
            sleep_time = 60 - (now - self._bucket_start_time)
            if sleep_time > 0:
                print(f"[RateLimit] Token quota reached ({self._token_bucket}/{self.max_tokens_per_minute} tokens). Sleeping for {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)
            self._token_bucket = 0
            self._bucket_start_time = time.time()
        
        # Check request quota
        if self._request_count >= self.max_requests_per_minute:
            sleep_time = 60 - (now - self._request_start_time)
            if sleep_time > 0:
                print(f"[RateLimit] Request quota reached ({self._request_count}/{self.max_requests_per_minute} requests). Sleeping for {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)
            self._request_count = 0
            self._request_start_time = time.time()
        
        # Update counters
        self._token_bucket += tokens
        self._request_count += 1
        
        # Small buffer between requests
        time.sleep(0.5)
    
    def _exponential_backoff_sleep(self, attempt: int, base_error: str = "") -> float:
        """Calculate exponential backoff sleep time"""
        sleep_time = min(
            self.base_sleep_time * (self.backoff_multiplier ** attempt),
            self.max_sleep_time
        )
        
        # Add jitter to prevent thundering herd
        import random
        jitter = random.uniform(0.8, 1.2)
        sleep_time *= jitter
        
        print(f"[Backoff] Attempt {attempt + 1}/{self.max_retries} failed. {base_error}")
        print(f"[Backoff] Sleeping for {sleep_time:.1f} seconds (base: {self.base_sleep_time}s, multiplier: {self.backoff_multiplier})")
        
        return sleep_time
    
    def _initialize_gemini(self):
        """Initialize Gemini client using service account credentials"""
        try:
            # Clear any existing API key configuration to avoid conflicts
            if "GOOGLE_API_KEY" in os.environ:
                print("Warning: GOOGLE_API_KEY found in environment. Removing to use service account.")
                del os.environ["GOOGLE_API_KEY"]
            # Load service account credentials
            if not os.path.exists(self.credentials_path):
                raise FileNotFoundError(f"Service account file not found: {self.credentials_path}")
            
            # Set up service account authentication
            credentials = service_account.Credentials.from_service_account_file(
                self.credentials_path,
                scopes=['https://www.googleapis.com/auth/generative-language']
            )
            
            # Configure genai with service account
            genai.configure(credentials=credentials)

            # Configure safety settings to be more permissive for scientific content
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH", 
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE"
                }
            ]
            
            # Create the model
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config={
                    'temperature': self.temperature,
                    'max_output_tokens': 2048,
                },
                safety_settings=safety_settings
            )
            
            print(f"[Init] Initialized {self.model_name} successfully with service account: {self.credentials_path}")
            
        except Exception as e:
            print(f"Failed to initialize Gemini client with service account: {e}")
            raise
    
    def _get_system_instruction(self) -> str:
        """Get the system instruction for the evaluator"""
        return """You are an expert scientific evaluator assessing the quality of scientific responses against reference answers.

Your task is to evaluate responses using one critical criterion:

ACCURACY (0-100):
CRITICAL: Use ONLY these two scores for accuracy:
- 100: The answer contains the core correct factual content, concepts, and conclusions from the ideal answer
- 0: The answer is fundamentally wrong or contradicts the ideal answer

This is a BINARY evaluation - either the answer is essentially correct (100) or fundamentally incorrect (0).
No partial credit or intermediate scores allowed.

EVALUATION GUIDELINES:
- Focus ONLY on whether the main scientific concepts and conclusions are correct
- Check that the core factual claims from the ideal answer are present in the generated answer
- Verify the overall conceptual direction and main conclusions align
- Additional correct information beyond the ideal answer is acceptable
- Only award 0 if the answer contradicts the ideal answer or gets the main concepts wrong
- Award 100 if the answer captures the essential correct scientific understanding

Provide your evaluation with the numerical score and detailed rationale explaining why you chose 100 or 0."""
    
    def evaluate_single_response(
        self,
        question: str,
        generated_answer: str,
        ideal_answer: str,
        sources: Optional[List[str]] = None,
        system_name: str = "RAG System"
    ) -> Dict[str, Any]:
        """Evaluate a single RAG response using NEW SDK"""
        
        evaluation_task = f"""
Please evaluate this system's response against the ideal answer:

QUESTION: {question}

GENERATED ANSWER:
{generated_answer}

IDEAL ANSWER:
{ideal_answer}

Evaluate based on:
Accuracy (0-100): How factually correct is the answer compared to the ideal?

Provide your structured evaluation with detailed rationale as JSON with fields:
- accuracy_score: integer from 0-100
- rationale: string explanation
"""
        
        # Check rate limits before making request
        self._check_rate_limits(evaluation_task)
        
        # Retry loop with exponential backoff
        for attempt in range(self.max_retries):
            try:
                full_prompt = f"{self._get_system_instruction()}\n\n{evaluation_task}"
                # Generate evaluation using service account authenticated model
                # evaluation_response_schema = {
                #     "type": "object",
                #     "properties": {
                #         "accuracy_score": {
                #             "type": "integer",
                #             "description": "Factual accuracy score from 0-100, where 100 means perfectly matching the ideal answer and 0 means fundamentally wrong"
                #         },
                #         "rationale": {
                #             "type": "string",
                #             "description": "Brief explanation of the evaluation scores and comparison with ideal answer. Should explain why the score was chosen."
                #         }
                #     },
                #     "required": ["accuracy_score", "rationale"]
                # }
                response = self.model.generate_content(
                    full_prompt,
                    generation_config={
                        'temperature': self.temperature,
                        'max_output_tokens': 2048,
                         "response_mime_type": "application/json"
                        
                    })
                # Check if response was generated
                if not response.text:
                    raise RuntimeError("Empty response received from Gemini")
                
                
                # Parse the JSON response
                try:
                    evaluation_result = json.loads(response.text)
                    accuracy_score = evaluation_result.get('accuracy_score', 0)
                    rationale = evaluation_result.get('rationale', '')
                    
                    return {
                        "eval_accuracy_score": accuracy_score,
                        "eval_rationale": rationale,
                        "eval_successful": True,
                        "eval_error": None
                    }
                    
                except json.JSONDecodeError as e:
                    print(f"Failed to parse Gemini JSON response: {e}")
                    print(f"Raw response: {response.text[:300]}...")
                    return self._create_error_result(f"JSON parse error: {e}")
                
            except Exception as e:
                error_str = str(e).lower()
                
                # Handle different types of errors with appropriate responses
                if "429" in error_str or "quota" in error_str or "rate limit" in error_str:
                    if attempt < self.max_retries - 1:
                        sleep_time = self._exponential_backoff_sleep(attempt, "Rate limit exceeded")
                        time.sleep(sleep_time)
                        # Reset rate limit counters after quota errors
                        self._token_bucket = 0
                        self._bucket_start_time = time.time()
                        self._request_count = 0
                        self._request_start_time = time.time()
                        continue
                    else:
                        return self._create_error_result(f"Rate limit exceeded after {self.max_retries} attempts")
                
                elif any(code in error_str for code in ["500", "502", "503", "504"]):
                    if attempt < self.max_retries - 1:
                        sleep_time = self._exponential_backoff_sleep(attempt, f"Server error: {e}")
                        time.sleep(sleep_time)
                        continue
                    else:
                        return self._create_error_result(f"Server error after {self.max_retries} attempts: {e}")
                
                elif any(code in error_str for code in ["400", "401", "403"]):
                    # Client errors - don't retry
                    return self._create_error_result(f"Client error: {e}")
                
                else:
                    # Unknown errors - retry with backoff
                    if attempt < self.max_retries - 1:
                        sleep_time = self._exponential_backoff_sleep(attempt, f"Unknown error: {e}")
                        time.sleep(sleep_time)
                        continue
                    else:
                        return self._create_error_result(f"Unknown error after {self.max_retries} attempts: {e}")
        
        # This should never be reached due to the return statements above
        return self._create_error_result(f"Failed to get evaluation after {self.max_retries} attempts")
    
    def _create_error_result(self, error_msg: str) -> Dict[str, Any]:
        """Create a standardized error result"""
        return {
            "eval_accuracy_score": None,
            "eval_rationale": None,
            "eval_successful": False,
            "eval_error": error_msg
        }
    
    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get current rate limit status for monitoring"""
        now = time.time()
        
        return {
            "model": self.model_name,
            "tokens_used_this_minute": self._token_bucket,
            "max_tokens_per_minute": self.max_tokens_per_minute,
            "requests_used_this_minute": self._request_count,
            "max_requests_per_minute": self.max_requests_per_minute,
            "seconds_until_token_reset": max(0, 60 - (now - self._bucket_start_time)),
            "seconds_until_request_reset": max(0, 60 - (now - self._request_start_time)),
            "rate_limit_config": {
                "max_retries": self.max_retries,
                "base_sleep_time": self.base_sleep_time,
                "max_sleep_time": self.max_sleep_time,
                "backoff_multiplier": self.backoff_multiplier
            }
        }

class SingleRAGEvaluationSystem:
    """Generic RAG Evaluation System for Individual DataFrames - ENHANCED with both backends"""
    
    def __init__(self, 
                 evaluator_model: str = "gemini-2.5-pro-preview-06-05", 
                 results_dir: str = "rag_evaluation_results_gemini",
                 evaluator_backend: str = "gemini", 
                 credentials=None,
                 **evaluator_kwargs):
        """
        Initialize the Single RAG Evaluation System
        
        Args:
            evaluator_model: Model to use for AI evaluation
            results_dir: Directory to store evaluation results
            evaluator_backend: "autogen" (default) or "gemini"
            credentials: Path to credentials file (ignored for new SDK, use GOOGLE_API_KEY env var)
            **evaluator_kwargs: Additional arguments for evaluator (e.g., rate limiting for Gemini)
        """
        self.evaluator_backend = evaluator_backend.lower()
        
        if self.evaluator_backend == "autogen":
            # Use original AutoGen implementation
            self.evaluator = AIEvaluator(model=evaluator_model, **evaluator_kwargs)
        elif self.evaluator_backend == "gemini":
            # Use new Gemini implementation
            # Override model if it's still the default GPT model
            evaluator_model = "gemini-2.5-pro-preview-06-05"
            
            # Note: credentials parameter is ignored for new SDK
            if credentials:
                print(f"Note: credentials parameter ({credentials}) is ignored. Please set GOOGLE_API_KEY environment variable.")
            
            self.evaluator = GeminiEvaluator(model=evaluator_model, **evaluator_kwargs)
        else:
            raise ValueError(f"Unknown evaluator_backend: {evaluator_backend}. Must be 'autogen' or 'gemini'")
        
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        print(f"Initialized evaluation system with {self.evaluator_backend} backend")
        if hasattr(self.evaluator, 'model_name'):
            print(f"Using model: {self.evaluator.model_name}")
        else:
            print(f"Using model: {self.evaluator.model}")
    
    def check_required_columns(self, df: pd.DataFrame) -> bool:
        """
        Check if the dataframe has the required columns for evaluation
        
        Args:
            df: DataFrame to check
            
        Returns:
            Boolean indicating if required columns are present
        """
        required_columns = ['question', 'ideal_solution']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"ERROR: Missing required columns: {missing_columns}")
            print(f"Available columns: {list(df.columns)}")
            return False
        
        print("All required columns found")
        return True
    
    def evaluate_single_dataframe(
        self, 
        df: pd.DataFrame, 
        system_name: str,
        max_evaluations: Optional[int] = None,
        save_results: bool = True
    ) -> pd.DataFrame:
        """
        Evaluate a single RAG system dataframe with known column structure
        
        Args:
            df: DataFrame containing RAG responses with columns:
                - question: The question text
                - answer: The generated answer
                - ideal_solution: The reference/ideal answer
                - sources: Source citations (optional)
                - success: Success flag (optional)
            system_name: Name of the RAG system for identification
            max_evaluations: Maximum number of evaluations to perform (None for all)
            save_results: Whether to save results to file
            
        Returns:
            DataFrame with evaluation results added
        """
        print(f"\n{'='*60}")
        print(f"EVALUATING: {system_name} (using {self.evaluator_backend} backend)")
        print(f"{'='*60}")
        
        # Check required columns
        if not self.check_required_columns(df):
            return df
        
        print(f"Available columns: {list(df.columns)}")
        
        # Create a copy of the dataframe
        eval_df = df.copy()
        
        # Filter to successful responses if success column exists
        if 'success' in df.columns:
            evaluable_rows = eval_df[eval_df['success'] == True]
            print(f"Filtering by success column: {len(evaluable_rows)} successful out of {len(eval_df)} total")
        else:
            # Check for non-null answers and ideals
            if 'answer' in df.columns:
                evaluable_rows = eval_df[
                    eval_df['ideal_solution'].notna() & 
                    eval_df['answer'].notna()
                ]
            else:
                evaluable_rows = eval_df[eval_df['ideal_solution'].notna()]
            print(f"No success column found, filtering by non-null values: {len(evaluable_rows)} evaluable rows")
        
        # Limit evaluations if requested
        if max_evaluations:
            evaluable_rows = evaluable_rows.head(max_evaluations)
            print(f"Limited to: {len(evaluable_rows)} evaluations")
        
        if len(evaluable_rows) == 0:
            print("No evaluable rows found. Returning original dataframe.")
            return eval_df
        
        # Initialize evaluation columns
        eval_columns = [
            "eval_accuracy_score", "eval_rationale", "eval_successful", "eval_error",
            "eval_processing_time"
        ]
        for col in eval_columns:
            eval_df[col] = None
        
        # Show rate limit status for Gemini
        if self.evaluator_backend == "gemini":
            status = self.evaluator.get_rate_limit_status()
            print(f"Rate limit status: {status['requests_used_this_minute']}/{status['max_requests_per_minute']} requests, {status['tokens_used_this_minute']}/{status['max_tokens_per_minute']} tokens")
        
        # Perform evaluations
        start_time = time.time()
        successful_evaluations = 0
        
        for idx, (_, row) in enumerate(evaluable_rows.iterrows(), 1):
            print(f"\nEvaluating {idx}/{len(evaluable_rows)} - Question ID: {row.get('question_id', idx)}")
            
            # Extract data directly using known column names
            question = row['question']
            ideal = row['ideal_solution']
            
            # Get answer - prefer 'answer' column, fallback to 'response'
            if 'answer' in df.columns and pd.notna(row['answer']):
                answer = row['answer']
            elif 'response' in df.columns and pd.notna(row['response']):
                answer = row['response']
            else:
                answer = ""
                print(f"    Warning: No answer found for question {idx}")
            
            # Skip if no answer available
            if not answer or answer.strip() == "":
                print(f"    ✗ Skipping - No answer available")
                continue
            
            # Perform evaluation
            eval_start_time = time.time()
            evaluation_result = self.evaluator.evaluate_single_response(
                question=question,
                generated_answer=answer,
                ideal_answer=ideal,
                system_name=system_name
            )
            eval_time = time.time() - eval_start_time
            evaluation_result['eval_processing_time'] = eval_time
            
            # Update dataframe with results
            row_idx = row.name  # Get the original index
            for key, value in evaluation_result.items():
                eval_df.at[row_idx, key] = value
            
            # Print progress
            if evaluation_result.get('eval_successful'):
                successful_evaluations += 1
                print( f"Accuracy:{evaluation_result['eval_accuracy_score']}")
                print(f"  Time: {eval_time:.2f}s")
            else:
                print(f"  ✗ Failed: {evaluation_result.get('eval_error', 'Unknown error')}")
        
        total_time = time.time() - start_time
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"EVALUATION COMPLETE: {system_name}")
        print(f"{'='*60}")
        print(f"Backend: {self.evaluator_backend}")
        print(f"Total evaluation time: {total_time:.2f} seconds")
        print(f"Successful evaluations: {successful_evaluations}/{len(evaluable_rows)}")
        print(f"Success rate: {successful_evaluations/len(evaluable_rows)*100:.1f}%")
        
        if successful_evaluations > 0:
            successful_evals = eval_df[eval_df['eval_successful'] == True]
            print(f"\nAverage Scores:")
            print(f"  Accuracy: {successful_evals['eval_accuracy_score'].mean():.2f}")
        
        # Show final rate limit status for Gemini
        if self.evaluator_backend == "gemini":
            final_status = self.evaluator.get_rate_limit_status()
            print(f"\nFinal rate limit usage: {final_status['requests_used_this_minute']}/{final_status['max_requests_per_minute']} requests, {final_status['tokens_used_this_minute']}/{final_status['max_tokens_per_minute']} tokens")
        
        # Save results if requested
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{system_name.replace(' ', '_').lower()}_evaluated_{timestamp}.csv"
            filepath = os.path.join(self.results_dir, filename)
            eval_df.to_csv(filepath, index=False)
            print(f"\nResults saved to: {filepath}")
        
        return eval_df
    
    def _save_individual_results(self, df: pd.DataFrame, system_name: str):
        """Save individual system results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{system_name.replace(' ', '_').lower()}_evaluated_{timestamp}.csv"
        filepath = os.path.join(self.results_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"Saved results: {filepath}")