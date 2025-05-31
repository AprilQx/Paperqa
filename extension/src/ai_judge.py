import os
import json
import datetime
from typing import List, Dict, Any
from autogen import AssistantAgent, UserProxyAgent
from pydantic import BaseModel, Field
from shortuuid import uuid

# Directory to save results
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'ai_judge_evaluations')
os.makedirs(RESULTS_DIR, exist_ok=True)

class SummaryEvaluation(BaseModel):
    conciseness_score: int = Field(ge=0, le=5)
    accuracy_score: int = Field(ge=0, le=90)
    citation_score: int = Field(ge=0, le=5)
    rationale: str

def create_ai_judge_agent():
    system_message = """You are an expert scientific evaluator assessing the quality of scientific summaries against reference answers.\n\nYour task is to evaluate responses using three critical criteria on a numerical scale:\n\n1. CONCISENESS (0-5):\n- 5: Perfectly concise with complete key information\n- 2-4: Generally good but contains some unnecessary details or slightly lacking\n- 0-2: Extremely wordy or missing most critical information\n\n2. ACCURACY (0-90):\n-70-90: Excellent accuracy (80-90: minor issues, 70-79: virtually perfect)\n-50-69: Moderate accuracy with notable errors or omissions\n-20-49: Significant factual problems or misunderstandings\n-0-19: Fundamentally flawed or contradicts reference answer\n\n3. CITATION QUALITY (0-5):\n- 5: Exemplary citation practice (specific, relevant, comprehensive)\n- 2-5: Adequate citation with room for improvement\n- 0-2: Severely lacking or inappropriate citations\n\nEVALUATION GUIDELINES:\n- Focus on factual alignment rather than exact wording or phrasing\n- For mathematical content, verify formula correctness including LaTeX syntax and notation\n- Citation formats may vary (e.g., \"p6; sec2.2.1\" or \"Smith et al., 2023, p.45\")\n"""
    return AssistantAgent(
        name="ai_judge",
        system_message=system_message,
        llm_config={
            "model": "gpt-4o",
            "temperature": 0.1,
            "functions": [
                {"name": "evaluate_summary", "parameters": SummaryEvaluation.model_json_schema()}
            ],
            "function_call": {"name": "evaluate_summary"}
        }
    )

def evaluate_answers(samples: List[Dict[str, Any]], save_dir: str = RESULTS_DIR, return_results: bool = False, verbose: bool = True):
    """
    Evaluate a list of AI-generated answers using the AG2 judge.
    Args:
        samples: List of dicts with keys: question, ai_answer, ideal_answer, expected_citations, source_file, key_passage
        save_dir: Directory to save evaluation result JSON files
        return_results: If True, return a list of all result_data dicts
        verbose: If True, print/log progress and scores
    Returns:
        If return_results is True, returns a list of result_data dicts. Otherwise, returns None.
    """
    os.makedirs(save_dir, exist_ok=True)
    ai_judge = create_ai_judge_agent()
    user_proxy = UserProxyAgent(
        name="user",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        is_termination_msg=lambda x: True if "TERMINATE" in x.get("content", "") else False,
        code_execution_config={"use_docker": False}
    )
    all_results = []
    for idx, sample in enumerate(samples):
        question = sample["question"]
        ai_answer = sample["ai_answer"]
        ideal_answer = sample["ideal_answer"]
        expected_citations = sample.get("expected_citations", "")
        source_file = sample.get("source_file", "")
        key_passage = sample.get("key_passage", "")
        key_passage_text = f"KEY PASSAGE FROM SOURCE:\n{key_passage}" if key_passage else "No specific key passage provided."
        evaluation_task = f"""
        Please evaluate this scientific answer against the ideal answer:

        QUESTION: {question}

        AI ANSWER:
        {ai_answer}

        IDEAL ANSWER:
        {ideal_answer}

        EXPECTED CITATIONS:
        {expected_citations}

        {key_passage_text}

        Evaluate based on:
        1. Conciseness (0-5)
        2. Accuracy compared to ideal answer (0-90)
        3. Citation quality (0-5)

        Provide detailed rationale for your scores."""
        user_proxy.reset()
        ai_judge.reset()
        user_proxy.initiate_chat(
            ai_judge,
            message=evaluation_task,
            max_turns=1
        )
        last_message = ai_judge.last_message()
        evaluation_result = None
        if "function_call" in last_message:
            function_call = last_message["function_call"]
            if function_call.get("name") == "evaluate_summary":
                try:
                    evaluation_result = json.loads(function_call.get("arguments", "{}"))
                except json.JSONDecodeError:
                    if verbose:
                        print("Failed to parse evaluation function arguments")
        result_id = f"summary_{uuid()[:8]}"
        result_file = os.path.join(save_dir, f"{result_id}.json")
        result_data = {
            "question": question,
            "ai_answer": ai_answer,
            "ideal_answer": ideal_answer,
            "expected_citations": expected_citations,
            "source_file": source_file,
            "key_passage": key_passage,
            "evaluation": evaluation_result,
            "timestamp": datetime.datetime.now().isoformat()
        }
        with open(result_file, 'w') as f:
            json.dump(result_data, f, indent=2)
        if verbose:
            print(f"Saved evaluation results to {result_file}")
            if evaluation_result:
                print(f"Scores: Conciseness={evaluation_result.get('conciseness_score')}, Accuracy={evaluation_result.get('accuracy_score')}, Citations={evaluation_result.get('citation_score')}")
        all_results.append(result_data)
    if return_results:
        return all_results

if __name__ == "__main__":
    # Example usage: load samples from a JSON file or define them inline
    # samples = json.load(open('your_samples.json'))
    samples = [
        {
            "question": "How much does the ACT DR6 power spectra improve white noise levels over previous results?",
            "ai_answer": "I cannot answer.",
            "ideal_answer": "ACT DR6 power spectra white noise levels improve over those of Planck by roughly a factor of 3 with polarization and a factor of two in temperature.",
            "expected_citations": "p4; sec2.1",
            "source_file": "https://arxiv.org/abs/2503.14454",
            "key_passage": "These power spectra have white noise levels that improve over those of Planck by roughly a factor of 3 with polarization and a factor of two in temperature."
        }
    ]
    evaluate_answers(samples, save_dir=RESULTS_DIR, return_results=True, verbose=True) 