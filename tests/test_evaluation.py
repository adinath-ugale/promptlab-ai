"""Unit tests for utils/evaluation.py.

Verifies:
- Mathematical RQS calculation: RQS = (Accuracy + Relevance + Consistency + Fluency) / 4
- Boundary testing (1.0 to 5.0 scale)
- Rejection of invalid ranges (< 1.0 or > 5.0)
- Rejection of non-numeric score formats
- JSON response extraction and parsing (plain JSON, markdown blocks, malformed strings)
- DataFrame conversion and CSV export
"""

import pytest
import pandas as pd
from utils.evaluation import (
    calculate_rqs,
    validate_score,
    extract_json_from_text,
    create_evaluation_record,
    records_to_dataframe,
    export_evaluations_to_csv,
    RQSResult,
)


def test_calculate_rqs_standard():
    """Verify arithmetic mean calculation with standard scores."""
    # (4.0 + 4.0 + 4.0 + 4.0) / 4 = 4.0
    assert calculate_rqs(4.0, 4.0, 4.0, 4.0) == 4.0

    # (5.0 + 4.0 + 3.0 + 4.0) / 4 = 16.0 / 4 = 4.0
    assert calculate_rqs(5.0, 4.0, 3.0, 4.0) == 4.0

    # (4.5 + 4.2 + 4.8 + 4.1) / 4 = 17.6 / 4 = 4.4
    assert calculate_rqs(4.5, 4.2, 4.8, 4.1) == 4.4


def test_calculate_rqs_boundary_values():
    """Verify minimum (1.0) and maximum (5.0) boundaries."""
    assert calculate_rqs(1.0, 1.0, 1.0, 1.0) == 1.0
    assert calculate_rqs(5.0, 5.0, 5.0, 5.0) == 5.0


def test_validate_score_valid():
    """Verify valid score casting and boundary checks."""
    assert validate_score(3.5, "Accuracy") == 3.5
    assert validate_score("4.2", "Relevance") == 4.2
    assert validate_score(1, "Consistency") == 1.0
    assert validate_score(5, "Fluency") == 5.0


def test_validate_score_out_of_bounds():
    """Verify scores outside [1.0, 5.0] raise ValueError."""
    with pytest.raises(ValueError, match="outside valid range"):
        validate_score(0.9, "Accuracy")

    with pytest.raises(ValueError, match="outside valid range"):
        validate_score(5.1, "Relevance")

    with pytest.raises(ValueError, match="outside valid range"):
        validate_score(-1.0, "Consistency")

    with pytest.raises(ValueError, match="outside valid range"):
        validate_score(10.0, "Fluency")


def test_validate_score_non_numeric():
    """Verify non-numeric values raise ValueError."""
    with pytest.raises(ValueError, match="not a numeric value"):
        validate_score("high", "Accuracy")

    with pytest.raises(ValueError, match="not a numeric value"):
        validate_score(None, "Relevance")


def test_extract_json_from_text():
    """Verify JSON extraction from various text formats."""
    # 1. Plain clean JSON
    plain = '{"accuracy": 4.5, "relevance": 5.0, "consistency": 4.0, "fluency": 4.5}'
    data = extract_json_from_text(plain)
    assert data is not None
    assert data["accuracy"] == 4.5

    # 2. Markdown fenced code block
    fenced = 'Here is the score:\n```json\n{"accuracy": 4.0, "relevance": 4.0}\n```\nDone.'
    data2 = extract_json_from_text(fenced)
    assert data2 is not None
    assert data2["accuracy"] == 4.0

    # 3. JSON with surrounding prose
    surrounded = 'Lead text {"accuracy": 5.0} trailing text'
    data3 = extract_json_from_text(surrounded)
    assert data3 is not None
    assert data3["accuracy"] == 5.0

    # 4. Malformed / empty text
    assert extract_json_from_text("") is None
    assert extract_json_from_text("Invalid text with no braces") is None


def test_create_and_export_evaluation_records():
    """Verify RQSResult creation, conversion to DataFrame, and CSV export."""
    record = create_evaluation_record(
        task="Question Answering",
        strategy="Role Prompting",
        model="gemini-2.5-flash",
        input_text="What is NLP?",
        generated_response="NLP is Natural Language Processing.",
        accuracy=5.0,
        relevance=5.0,
        consistency=4.5,
        fluency=4.5,
        acc_rationale="Factual and precise",
        rel_rationale="Directly answers",
    )
    assert record.rqs_score == 4.75

    df = records_to_dataframe([record])
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert "RQS Score" in df.columns
    assert df["RQS Score"].iloc[0] == 4.75

    csv_out = export_evaluations_to_csv([record])
    assert "Question Answering" in csv_out
    assert "Role Prompting" in csv_out
    assert "4.75" in csv_out
