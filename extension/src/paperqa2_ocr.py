import csv
import re
import os
import json
import nest_asyncio
from shortuuid import uuid
from typing import Any, Dict, List, Optional, Tuple, Union
import datetime
import logging

from inspect_ai import Task, task, Epochs
from inspect_ai.dataset import Dataset, Sample
from inspect_ai.solver import generate, solver
from inspect_ai.scorer import exact 
from inspect_ai.log import transcript
from inspect_ai.agent import bridge
from inspect_ai.dataset import Sample, json_dataset
from inspect_ai.model import ModelOutput, StopReason

from paperqa import ask
from paperqa.agents.main import agent_query
from paperqa.agents.search import get_directory_index
from paperqa.settings import Settings, AgentSettings
from paperqa.settings import AnswerSettings

# Import AutoGen components
import autogen
from autogen import UserProxyAgent, AssistantAgent
from pydantic import BaseModel, Field
from summary_scorer import summary_quality_scorer
# Set environment variable to disable Docker
os.environ["AUTOGEN_USE_DOCKER"] = "False"

# Path configurations
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..','..'))
OCR_OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'ocr_output')
QUESTIONS_FILE = os.path.join(PROJECT_ROOT, 'data', 'formatted_summary_questions', 'summary_questions.jsonl')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'extension', 'results', 'summary_evaluations_paperqa2_ocr')

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PaperQA2_OCR")

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Define structured output models for summary evaluation
class SummaryEvaluation(BaseModel):
    """Structured evaluation of a scientific summary"""
    conciseness_score: int = Field(
        description="Conciseness score (1-5), where 5 is optimally concise without missing important information", 
        ge=1, le=5
    )
    accuracy_score: int = Field(
        description="Factual accuracy score (1-90), where 90 means perfectly matching the ideal answer", 
        ge=1, le=90
    )
    citation_score: int = Field(
        description="Citation quality score (1-5), where 5 means all claims are properly cited to reliable sources", 
        ge=1, le=5
    )
    rationale: str = Field(
        description="Brief explanation of the evaluation scores and comparison with ideal answer"
    )

def record_to_sample(record: dict[str, Any]) -> Sample:
    """Convert a question record to an Inspect AI Sample"""
    return Sample(
        input=f"Question: {record['question']}",
        target=record["ideal"],  # The ideal answer is the target
        metadata={
            "citations": record.get("citations", ""),
            "ideal": record["ideal"],
            "key_passage": record.get("key_passage", None),
            "source_file": record.get("source_file", None)
        }
    )

def paperqa2_ocr_summary_agent():
    """Agent function for PaperQA2 that generates and evaluates summaries using OCR-processed documents"""
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
        },
        summary_llm="gpt-4o-mini",
        summary_llm_config={
            "rate_limit": {"gpt-4o-mini": "30000 per 1 minute"},
        },
        answer=AnswerSettings(
            evidence_k=30,  # Number of evidence items to consider
            answer_max_sources=15,  # Maximum sources to include
            evidence_skip_summary=False  # Don't skip summarization
        ),
        agent=AgentSettings(
            agent_llm="gpt-4o-mini",
            agent_llm_config={
                "rate_limit": {"gpt-4o-mini": "30000 per 1 minute"},
            }
        ),
        embedding="text-embedding-3-small",
        temperature=0.5,  # Keep deterministic
        paper_directory=OCR_OUTPUT_DIR  # Using OCR output directory instead of original papers
    )
    
    index_built = False
    
    async def build_index():
        """Build PaperQA2 document index for OCR-processed files"""
        nonlocal index_built
        if not index_built:
            transcript().info("Building PaperQA2 document index for OCR-processed files...")
            
            index_settings = Settings(
                paper_directory=OCR_OUTPUT_DIR,
                agent={"index": {
                    "sync_with_paper_directory": True,
                    "recurse_subdirectories": True
                }}
            )
            
            if not os.path.exists(OCR_OUTPUT_DIR):
                transcript().error(f"OCR output directory not found: {OCR_OUTPUT_DIR}")
                raise FileNotFoundError(f"OCR output directory not found: {OCR_OUTPUT_DIR}")
                
            # List all txt files in OCR output directory
            txt_files = [f for f in os.listdir(OCR_OUTPUT_DIR) if f.endswith('.txt')]
            transcript().info(f"Found {len(txt_files)} text files in {OCR_OUTPUT_DIR}")
            if txt_files:
                transcript().info(f"Sample files: {txt_files[:5]}")
            
            built_index = await get_directory_index(settings=index_settings)
            transcript().info(f"Using index: {index_settings.get_index_name()}")
            index_files = await built_index.index_files
            transcript().info(f"Indexed {len(index_files)} files")
            index_built = True
        return built_index
    
    async def run(sample: dict) -> dict:
        """Process a sample and return evaluation results"""
        # Build index if not already built
        if not index_built:
            await build_index()

        # Extract question from the input messages
        user_message = next((msg["content"] for msg in sample["messages"] if msg["role"] == "user"), "")
            
        # Parse the question
        question_match = re.search(r"Question:\s*(.*?)(?:\Z)", user_message, re.DOTALL)
        question = question_match.group(1).strip() if question_match else user_message
        
        # Get target (ideal answer) and metadata
        target = ""
        if "target" in sample and sample["target"]:
            target = sample["target"]
            transcript().info(f"Found target in sample.target: {target[:50]}")
        elif "metadata" in sample and "ideal" in sample["metadata"] and sample["metadata"]["ideal"]:
            target = sample["metadata"]["ideal"]
            transcript().info(f"Found target in sample.metadata.ideal: {target[:50]}")
        else:
            # Check raw structure for debugging
            transcript().info(f"Sample keys: {list(sample.keys())}")
            if "metadata" in sample:
                transcript().info(f"Metadata keys: {list(sample['metadata'].keys())}")
            transcript().warning("No target/ideal answer found for this question!")
        citations = sample.get("metadata", {}).get("citations", "")
        key_passage = sample.get("metadata", {}).get("key_passage", None)
        source_file = sample.get("metadata", {}).get("source_file", None)
        transcript().info(f"Processing question: {question[:50]}...")
        
        try:
            # # Create AI agents for evaluation
            # ai_judge = AssistantAgent(
            #     name="ai_judge",
            #     system_message="""You are an expert scientific evaluator assessing the quality of scientific summaries against reference answers.

            #     Your task is to evaluate responses using three critical criteria on a numerical scale:

            #     1. CONCISENESS (0-5):
            #     - 5: Perfectly concise with complete key information
            #     - 2-4: Generally good but contains some unnecessary details or slightly lacking
            #     - 0-2: Extremely wordy or missing most critical information
                
            #     2. ACCURACY (0-90):
            #     -70-90: Excellent accuracy (80-90: minor issues, 70-79: virtually perfect)
            #     -50-69: Moderate accuracy with notable errors or omissions
            #     -20-49: Significant factual problems or misunderstandings
            #     -0-19: Fundamentally flawed or contradicts reference answer
            #     3. CITATION QUALITY (0-5):
            #     - 5: Exemplary citation practice (specific, relevant, comprehensive)
            #     - 2-5: Adequate citation with room for improvement
            #     - 0-2: Severely lacking or inappropriate citations

            #     EVALUATION GUIDELINES:
            #     - Focus on factual alignment rather than exact wording or phrasing
            #     - For mathematical content, verify formula correctness including LaTeX syntax and notation
            #     - Citation formats may vary (e.g., "p6; sec2.2.1" or "Smith et al., 2023, p.45")
            #     - Consider partial credit when citations are present but incomplete
            #     - When key passages are specified, ensure the evaluation emphasizes coverage of that material
            #     - Balance technical accuracy with accessibility of explanation

            #     For each criterion, provide:
            #     1. A numerical score within the specified range
            #     2. A brief rationale (2-3 sentences) explaining your reasoning
            #     3. At least one specific example from the text supporting your assessment.

            #     """,
            #     llm_config={
            #         "model": "gpt-4o",
            #         "temperature": 0.1,
            #         "functions": [
            #             {"name": "evaluate_summary", "parameters": SummaryEvaluation.model_json_schema()}
            #         ],
            #         "function_call": {"name": "evaluate_summary"}
            #     }
            # )
            
            # user_proxy = UserProxyAgent(
            #     name="user",
            #     human_input_mode="NEVER",
            #     max_consecutive_auto_reply=10,
            #     is_termination_msg=lambda x: True if "TERMINATE" in x.get("content", "") else False,
            #     code_execution_config={"use_docker": False}
            # )
            
            async def query_paperqa(query: str) -> str:
                """Query PaperQA2 for scientific evidence using OCR-processed documents"""
                transcript().info(f"Searching OCR-processed papers for: {query[:50]}...")
                nest_asyncio.apply()
                response = ask(query, settings=settings)
                return response.dict()['session']['answer']
            
            # Get PaperQA2 answer using OCR-processed documents
            transcript().info("Getting answer from PaperQA2 using OCR documents...")
            # ai_judge.register_function(function_map={"query_paperqa": query_paperqa})
            evidence_query = f"Just answer from scientific papers about: {question}, and be concise. No other information is needed to provide except the brief evidence from the paper. Try to return the chapter and section number of the paper if possible."
            answer = await query_paperqa(evidence_query)
    
            transcript().info(f"PaperQA2 answer received ({len(answer)} chars)")
            
            # # Now have the AI judge evaluate the answer
            # transcript().info("AI judge evaluating answer quality...")
            
            # # Format the evaluation task
            # key_passage_text = f"KEY PASSAGE FROM SOURCE:\n{key_passage}" if key_passage else "No specific key passage provided."

            # # Then use it in your f-string:
            # evaluation_task = f"""
            # Please evaluate this scientific answer against the ideal answer:

            # QUESTION: {question}

            # PAPERQA2 OCR-BASED ANSWER:
            # {answer}

            # IDEAL ANSWER:
            # {target}

            # EXPECTED CITATIONS:
            # {citations}

            # {key_passage_text}

            # Evaluate based on:
            # 1. Conciseness (1-5)
            # 2. Accuracy compared to ideal answer (1-90)
            # 3. Citation quality (1-5)

            # Provide detailed rationale for your scores."""
            
            # # Reset agents for evaluation
            # user_proxy.reset()
            # ai_judge.reset()
            
            # # Get AI judge evaluation
            # user_proxy.initiate_chat(
            #     ai_judge,
            #     message=evaluation_task,
            #     max_turns=1
            # )
            # transcript().info(f"Chat completed successfully")
            # # Extract evaluation results
            # last_message = ai_judge.last_message()
            
            # evaluation_result = None
            # if "function_call" in last_message:
            #     function_call = last_message["function_call"]
            #     if function_call.get("name") == "evaluate_summary":
            #         try:
            #             evaluation_result = json.loads(function_call.get("arguments", "{}"))
            #             transcript().info(f"Evaluation scores: Conciseness={evaluation_result.get('conciseness_score')}, "
            #                              f"Accuracy={evaluation_result.get('accuracy_score')}, "
            #                              f"Citations={evaluation_result.get('citation_score')}")
            #         except json.JSONDecodeError:
            #             transcript().info("Failed to parse evaluation function arguments")
            
            # Save results to file
            result_id = f"ocr_summary_{uuid()[:8]}"
            result_file = os.path.join(RESULTS_DIR, f"{result_id}.json")
            
            result_data = {
                "question": question,
                "generated_answer": answer,
                "ideal_answer": target,
                "expected_citations": citations,
                "source_file": source_file,
                "key_passage": key_passage
                # "evaluation": evaluation_result,
                #"timestamp": datetime.datetime.now().isoformat()
            }
            
            with open(result_file, 'w') as f:
                json.dump(result_data, f, indent=2)
                
            transcript().info(f"Saved evaluation results to {result_file}")

            # # Return results for Inspect-AI
            # if evaluation_result:
            #     accuracy_score = evaluation_result.get('accuracy_score', 0)
            #     conciseness_score = evaluation_result.get('conciseness_score', 0)
            #     citation_score = evaluation_result.get('citation_score', 0) 
            #     # Normalize to 0-1 range for compatibility with Inspect-AI scoring
            #     normalized_score = (accuracy_score + conciseness_score + citation_score) / 100.0
                
            #     return {
            #         "output": answer,
            #         "evaluation": evaluation_result,
            #         "metadata": {
            #             "accuracy_score": accuracy_score,
            #             "conciseness_score": conciseness_score,
            #             "citation_score": citation_score,
            #             "normalized_score": normalized_score
            #         }
            #     }
            # else:
            return {"output": answer}
        
                
        except Exception as e:
            transcript().error(f"Error in PaperQA2 OCR summary evaluation: {str(e)}")
            import traceback
            transcript().error(f"Traceback: {traceback.format_exc()}")
            return {"output": str(e), "score": 0.0}
    
    return run

@task
def evaluate_ocr_summary_answers(
    model: str = "gpt-4o-mini",
    evidence_k: int = 30,
    max_sources: int = 15,
    skip_summary: bool = False,
    accuracy_threshold: float = 0.5
):
    """Task to evaluate PaperQA2's cosmology summaries using OCR-processed documents against ideal answers"""
    dataset = json_dataset(QUESTIONS_FILE, record_to_sample)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"paperqa2_ocr_summary_eval_{model}_evK{evidence_k}_maxS{max_sources}_{timestamp}"
    
    return Task(
        dataset=dataset,
        solver=bridge(paperqa2_ocr_summary_agent()),
        scorer=summary_quality_scorer(accuracy_threshold=accuracy_threshold),
        epochs=Epochs(1, "mode"),
        metadata={"experiment_name": experiment_name}
    )

if __name__ == "__main__":
    logger.info(f"Starting PaperQA2 OCR evaluation script")
    logger.info(f"OCR output directory: {OCR_OUTPUT_DIR}")
    
    # Check if OCR output directory exists and count text files
    if not os.path.exists(OCR_OUTPUT_DIR):
        logger.error(f"OCR output directory not found: {OCR_OUTPUT_DIR}")
        exit(1)
    
    txt_files = [f for f in os.listdir(OCR_OUTPUT_DIR) if f.endswith('.txt')]
    logger.info(f"Found {len(txt_files)} text files in OCR output directory")
    
    # Run the evaluation task
    logger.info("Running OCR-based summary evaluation task...")
    task_instance = evaluate_ocr_summary_answers()
    # Note: The actual execution would be handled by the Inspect-AI framework
