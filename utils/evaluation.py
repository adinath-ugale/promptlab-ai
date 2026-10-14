"""PromptLab AI - Response Quality Evaluation (RQS) Module.

Implements the Response Quality Score (RQS) mathematical framework:
    RQS = (Accuracy + Relevance + Consistency + Fluency) / 4

Provides automated LLM-as-a-judge scoring, input validation, consistency testing workflows,
and Pandas-based record tracking and CSV export.
"""

import json
import re
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, asdict
import pandas as pd


@dataclass
class RQSResult:
    """Dataclass holding RQS evaluation scores, rationales, and metadata."""
    accuracy: float
    relevance: float
    consistency: float
    fluency: float
    rqs_score: float
    accuracy_rationale: str
    relevance_rationale: str
    consistency_rationale: str
    fluency_rationale: str
    task: str
    strategy: str
    model: str
    input_text: str
    generated_response: str
    reference_answer: Optional[str] = None
    timestamp: str = ""
    is_simulated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return data


def validate_score(score: Any, dimension_name: str = "Dimension") -> float:
    """Validate that an evaluation score is a numeric float between 1.0 and 5.0.
    
    Raises:
        ValueError: if score cannot be cast to float or falls outside [1.0, 5.0].
    """
    try:
        val = float(score)
    except (TypeError, ValueError) as err:
        raise ValueError(f"Invalid {dimension_name} score: '{score}' is not a numeric value.") from err

    if val < 1.0 or val > 5.0:
        raise ValueError(
            f"Invalid {dimension_name} score: {val} is outside valid range [1.0, 5.0]."
        )
    return round(val, 2)


def calculate_rqs(accuracy: float, relevance: float, consistency: float, fluency: float) -> float:
    """Calculate Response Quality Score using the arithmetic mean:
    
        RQS = (Accuracy + Relevance + Consistency + Fluency) / 4
        
    All parameters must be between 1.0 and 5.0.
    
    Returns:
        float: RQS score rounded to 2 decimal places.
    """
    acc = validate_score(accuracy, "Accuracy")
    rel = validate_score(relevance, "Relevance")
    con = validate_score(consistency, "Consistency")
    flu = validate_score(fluency, "Fluency")

    rqs = (acc + rel + con + flu) / 4.0
    return round(rqs, 2)


def extract_json_from_text(raw_text: str) -> Optional[Dict[str, Any]]:
    """Attempt to parse JSON from raw text or markdown code blocks."""
    if not raw_text or not raw_text.strip():
        return None

    # Try 1: Direct JSON parsing
    try:
        return json.loads(raw_text.strip())
    except json.JSONDecodeError:
        pass

    # Try 2: Extract from ```json ... ``` blocks
    json_block = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_text)
    if json_block:
        try:
            return json.loads(json_block.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try 3: Find first { and last }
    first_brace = raw_text.find("{")
    last_brace = raw_text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        snippet = raw_text[first_brace : last_brace + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            pass

    return None


def evaluate_response_with_llm(
    client: Any,
    model: str,
    task: str,
    strategy: str,
    input_text: str,
    generated_response: str,
    reference_answer: Optional[str] = None,
) -> Tuple[Optional[RQSResult], Optional[str]]:
    """Evaluate an AI generated response using an LLM-as-a-judge prompt.
    
    Returns:
        (RQSResult, error_message)
    """
    if client is None:
        return None, "Gemini client is not initialized. Please verify API key."

    if not generated_response or not generated_response.strip():
        return None, "Generated response is empty; cannot evaluate quality."

    ref_clause = (
        f"\nStandard / Reference Answer for Grounding:\n\"\"\"{reference_answer.strip()}\"\"\"\n"
        if reference_answer and reference_answer.strip()
        else "\nStandard / Reference Answer: None provided (evaluate based on intrinsic technical accuracy and truthfulness).\n"
    )

    eval_prompt = f"""You are an expert NLP evaluator and academic judge.
Evaluate the following AI-generated response on four distinct quality dimensions on a 1.0 to 5.0 scale (with decimal precision if appropriate).

Evaluation Criteria:
1. Accuracy (1.0 to 5.0): Factual correctness, absence of hallucinations, adherence to domain facts.
2. Relevance (1.0 to 5.0): How directly and completely the response addresses the user input without digressions.
3. Consistency (1.0 to 5.0): Internal logical cohesion, non-contradiction across claims, and alignment with constraints.
4. Fluency (1.0 to 5.0): Grammar, readability, clarity of expression, and vocabulary precision.

Context Information:
- NLP Task: {task}
- Prompting Strategy: {strategy}
- User Input:
\"\"\"
{input_text}
\"\"\"
{ref_clause}
- AI-Generated Response to Evaluate:
\"\"\"
{generated_response}
\"\"\"

Respond STRICTLY with valid JSON using this exact schema, without additional conversational text:
{{
  "accuracy": <float 1.0-5.0>,
  "accuracy_rationale": "<1-2 sentences explaining the accuracy score>",
  "relevance": <float 1.0-5.0>,
  "relevance_rationale": "<1-2 sentences explaining the relevance score>",
  "consistency": <float 1.0-5.0>,
  "consistency_rationale": "<1-2 sentences explaining internal consistency>",
  "fluency": <float 1.0-5.0>,
  "fluency_rationale": "<1-2 sentences explaining fluency and grammar>"
}}
"""

    try:
        from google.genai import types

        config = types.GenerateContentConfig(
            temperature=0.1,  # Low temperature for objective evaluation
            max_output_tokens=600,
        )

        response = client.models.generate_content(
            model=model,
            contents=eval_prompt,
            config=config,
        )

        if not response or not response.text:
            return None, "Evaluator LLM returned an empty response."

        data = extract_json_from_text(response.text)
        if not data:
            return None, f"Failed to parse structured JSON from judge output. Raw output:\n{response.text[:200]}"

        # Validate fields and bounds
        acc = validate_score(data.get("accuracy"), "Accuracy")
        rel = validate_score(data.get("relevance"), "Relevance")
        con = validate_score(data.get("consistency"), "Consistency")
        flu = validate_score(data.get("fluency"), "Fluency")
        rqs = calculate_rqs(acc, rel, con, flu)

        result = RQSResult(
            accuracy=acc,
            relevance=rel,
            consistency=con,
            fluency=flu,
            rqs_score=rqs,
            accuracy_rationale=str(data.get("accuracy_rationale", "No rationale provided.")).strip(),
            relevance_rationale=str(data.get("relevance_rationale", "No rationale provided.")).strip(),
            consistency_rationale=str(data.get("consistency_rationale", "No rationale provided.")).strip(),
            fluency_rationale=str(data.get("fluency_rationale", "No rationale provided.")).strip(),
            task=task,
            strategy=strategy,
            model=model,
            input_text=input_text,
            generated_response=generated_response,
            reference_answer=reference_answer,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            is_simulated=False,
        )
        return result, None

    except ValueError as val_err:
        return None, f"Evaluation score validation failed: {val_err}"
    except Exception as exc:
        return None, f"Evaluator call failed: {exc}"


def create_evaluation_record(
    task: str,
    strategy: str,
    model: str,
    input_text: str,
    generated_response: str,
    accuracy: float,
    relevance: float,
    consistency: float,
    fluency: float,
    acc_rationale: str = "",
    rel_rationale: str = "",
    con_rationale: str = "",
    flu_rationale: str = "",
    reference_answer: Optional[str] = None,
) -> RQSResult:
    """Manually construct a validated RQSResult record."""
    acc = validate_score(accuracy, "Accuracy")
    rel = validate_score(relevance, "Relevance")
    con = validate_score(consistency, "Consistency")
    flu = validate_score(fluency, "Fluency")
    rqs = calculate_rqs(acc, rel, con, flu)

    return RQSResult(
        accuracy=acc,
        relevance=rel,
        consistency=con,
        fluency=flu,
        rqs_score=rqs,
        accuracy_rationale=acc_rationale or "User-provided or heuristic evaluation",
        relevance_rationale=rel_rationale or "User-provided or heuristic evaluation",
        consistency_rationale=con_rationale or "User-provided or heuristic evaluation",
        fluency_rationale=flu_rationale or "User-provided or heuristic evaluation",
        task=task,
        strategy=strategy,
        model=model,
        input_text=input_text,
        generated_response=generated_response,
        reference_answer=reference_answer,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        is_simulated=False,
    )


def records_to_dataframe(records: List[RQSResult]) -> pd.DataFrame:
    """Convert a list of RQSResult objects into a cleaned Pandas DataFrame."""
    if not records:
        return pd.DataFrame(
            columns=[
                "Timestamp",
                "Task",
                "Strategy",
                "Model",
                "Accuracy",
                "Relevance",
                "Consistency",
                "Fluency",
                "RQS Score",
                "Input Preview",
                "Response Preview",
            ]
        )

    rows = []
    for r in records:
        rows.append(
            {
                "Timestamp": r.timestamp,
                "Task": r.task,
                "Strategy": r.strategy,
                "Model": r.model,
                "Accuracy": r.accuracy,
                "Relevance": r.relevance,
                "Consistency": r.consistency,
                "Fluency": r.fluency,
                "RQS Score": r.rqs_score,
                "Input Preview": (r.input_text[:60] + "...") if len(r.input_text) > 60 else r.input_text,
                "Response Preview": (r.generated_response[:80] + "...") if len(r.generated_response) > 80 else r.generated_response,
                "Accuracy Rationale": r.accuracy_rationale,
                "Relevance Rationale": r.relevance_rationale,
                "Consistency Rationale": r.consistency_rationale,
                "Fluency Rationale": r.fluency_rationale,
            }
        )
    return pd.DataFrame(rows)


def export_evaluations_to_csv(records: List[RQSResult]) -> str:
    """Serialize evaluation records to a CSV formatted string for download."""
    df = records_to_dataframe(records)
    return df.to_csv(index=False)
