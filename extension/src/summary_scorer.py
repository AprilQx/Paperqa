from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    NOANSWER,
    Score,
    Scorer,
    Target,
    accuracy,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState

@scorer(metrics=[accuracy(), stderr()])
def summary_quality_scorer(accuracy_threshold: float = 0.7) -> Scorer:
    """
    Custom scorer for evaluating summary quality based on AI judge evaluations.
    
    Args:
        accuracy_threshold: Minimum threshold for a summary to be considered correct
                          (normalized score, between 0 and 1)
    """
    async def score(state: TaskState, target: Target) -> Score:
        # Get the model's output summary
        summary = state.output.completion.strip()
        
        # Extract evaluation from state metadata if available
        evaluation = state.metadata.get("evaluation", {})
        
        # If we have structured evaluation scores
        if evaluation:
            # Extract individual scores
            accuracy_score = float(evaluation.get("accuracy_score", 0))
            conciseness_score = float(evaluation.get("conciseness_score", 0))
            citation_score = float(evaluation.get("citation_score", 0))
            rationale = evaluation.get("rationale", "No evaluation rationale provided")
            
            # Normalize scores to 0-1 range
            norm_accuracy = accuracy_score / 5.0
            norm_conciseness = conciseness_score / 5.0
            norm_citation = citation_score / 5.0
            
            # Calculate a weighted combined score
            # You can adjust the weights based on your priorities
            weighted_score = (
                0.5 * norm_accuracy +      # Accuracy has highest weight
                0.3 * norm_citation +      # Citation quality second
                0.2 * norm_conciseness   # Conciseness third         
            )
            
            # Determine if the summary meets the quality threshold
            is_quality_summary = weighted_score >= accuracy_threshold
            
            # Create explanation from scores and rationale
            explanation = (
                f"Summary Evaluation:\n"
                f"- Accuracy: {accuracy_score}/5\n"
                f"- Conciseness: {conciseness_score}/5\n"
                f"- Citation Quality: {citation_score}/5\n"
                f"Weighted Score: {weighted_score:.2f}\n"
                f"Evaluation: {rationale}"
            )
            
            return Score(
                value=CORRECT if is_quality_summary else INCORRECT,
                answer=summary[:100] + "..." if len(summary) > 100 else summary,  # Truncate for display
                explanation=explanation,
                metadata={
                    "accuracy_score": accuracy_score,
                    "conciseness_score": conciseness_score, 
                    "citation_score": citation_score,
                    "weighted_score": weighted_score
                }
            )
        else:
            # If no evaluation is available, mark as incorrect with explanation
            return Score(
                value=INCORRECT,
                answer=summary[:100] + "..." if len(summary) > 100 else summary,
                explanation="No structured evaluation was produced for this summary.",
            )
    
    return score

