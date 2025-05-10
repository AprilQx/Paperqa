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

from metrics import coverage, precision, accuracy

# Define constant for the "Insufficient information" answer choice
UNCERTAIN_ANSWER_CHOICE = "Insufficient information to answer the question"

@scorer(metrics=[accuracy(), precision(), coverage(), stderr()])
def precision_choice(no_answer: str | None = UNCERTAIN_ANSWER_CHOICE) -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        choices = state.choices
        explanation = state.output.completion

        model_output = state.output.completion.strip()

        # compute the target positions
        target_positions = [
            ord(target_character.upper()) - ord("A") for target_character in target.text
        ]
        generated_selected_choices = []
        
        # compute the answers
        correct_choices = [i for i, choice in enumerate(choices) if getattr(choice, "correct", False) is True]
        if correct_choices:
            generated_selected_choices = correct_choices
        elif len(model_output) == 1 and 'A' <= model_output.upper() <= 'Z':
            letter_index = ord(model_output.upper()) - ord('A')
            if 0 <= letter_index < len(choices):
                generated_selected_choices = [letter_index]
        elif hasattr(state, "metadata") and "choice_correct" in state.metadata:
            index = state.metadata["choice_correct"]
            if isinstance(index, int) and 0 <= index < len(choices):
                generated_selected_choices = [index]
    
        answers = [chr(ord("A") + choice) for choice in generated_selected_choices]
        
        # Check if the model selected a choice containing "Insufficient information"
        if no_answer is not None and generated_selected_choices:
            for choice_idx in generated_selected_choices:
                if choice_idx < len(choices) and no_answer in choices[choice_idx].value:
                    return Score(
                        value=NOANSWER,  # Mark as NOANSWER so it's excluded from precision calculation
                        answer=", ".join(answers) if answers else no_answer,
                        explanation=explanation,
                    )
        
        target_matches_choices = set(generated_selected_choices) == set(target_positions)
        return Score(
            value=CORRECT if target_matches_choices else INCORRECT,
            answer=", ".join(answers) if answers else model_output,
            explanation=explanation,
        )

    return score