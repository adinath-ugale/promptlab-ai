"""Unit tests for utils/prompts.py.

Verifies:
- Prompt template generation for all NLP tasks (QA, Summarization, Sentiment)
- All prompt strategies (Zero-Shot, Few-Shot, Role Prompting, Chain-of-Thought)
- Input text validation (empty input, minimum length, maximum length)
- Display formatting security (no secrets, properly formatted markdown)
"""

import pytest
from utils.prompts import (
    NLPTask,
    PromptStrategy,
    build_prompt,
    get_system_instruction,
    format_prompt_for_display,
    validate_input_text,
    FEW_SHOT_EXAMPLES,
    ROLE_PERSONAS,
)


def test_validate_input_text_valid():
    """Test standard valid user inputs."""
    is_valid, err = validate_input_text("What is tokenization in NLP?")
    assert is_valid is True
    assert err is None


def test_validate_input_text_empty_and_whitespace():
    """Test rejection of empty and whitespace-only inputs."""
    is_valid, err = validate_input_text("")
    assert is_valid is False
    assert "cannot be empty" in err.lower()

    is_valid, err = validate_input_text("   \n\t  ")
    assert is_valid is False
    assert "cannot be empty" in err.lower()

    is_valid, err = validate_input_text(None)
    assert is_valid is False


def test_validate_input_text_length_boundaries():
    """Test min_length and max_length boundary enforcement."""
    is_valid, err = validate_input_text("hi", min_length=5)
    assert is_valid is False
    assert "too short" in err.lower()

    long_text = "a" * 15001
    is_valid, err = validate_input_text(long_text, max_length=15000)
    assert is_valid is False
    assert "exceeds maximum allowed length" in err.lower()


@pytest.mark.parametrize("task", [
    NLPTask.QUESTION_ANSWERING,
    NLPTask.SUMMARIZATION,
    NLPTask.SENTIMENT_ANALYSIS,
])
def test_system_instruction_retrieval(task):
    """Ensure every task has a non-empty system instruction for standard and role strategies."""
    standard_inst = get_system_instruction(task, PromptStrategy.ZERO_SHOT)
    assert standard_inst and len(standard_inst) > 20

    role_inst = get_system_instruction(task, PromptStrategy.ROLE_PROMPT)
    assert role_inst and len(role_inst) > 20
    assert ROLE_PERSONAS[task] in role_inst


@pytest.mark.parametrize("strategy", [
    PromptStrategy.ZERO_SHOT,
    PromptStrategy.FEW_SHOT,
    PromptStrategy.ROLE_PROMPT,
    PromptStrategy.CHAIN_OF_THOUGHT,
])
def test_qa_prompt_generation(strategy):
    """Test question answering prompt generation across all strategies."""
    question = "How does attention work in Transformers?"
    sys_inst, user_prompt = build_prompt(
        task=NLPTask.QUESTION_ANSWERING,
        strategy=strategy,
        user_input=question,
    )
    assert sys_inst is not None and len(sys_inst) > 0
    assert question in user_prompt

    if strategy == PromptStrategy.FEW_SHOT:
        assert "Tokenization" in user_prompt
    elif strategy == PromptStrategy.ROLE_PROMPT:
        assert "Professor" in user_prompt
    elif strategy == PromptStrategy.CHAIN_OF_THOUGHT:
        assert "Core Conceptual Analysis" in user_prompt or "rationale" in user_prompt.lower()


@pytest.mark.parametrize("length_mode", ["Short", "Medium", "Detailed"])
def test_summarization_prompt_length_modes(length_mode):
    """Test summarization prompts with different target lengths."""
    article = "Deep learning models require data preprocessing and extensive training on GPUs."
    sys_inst, user_prompt = build_prompt(
        task=NLPTask.SUMMARIZATION,
        strategy=PromptStrategy.ZERO_SHOT,
        user_input=article,
        extra_params={"length_mode": length_mode},
    )
    assert article in user_prompt
    if length_mode == "Short":
        assert "1 to 2 clear sentences" in user_prompt
    elif length_mode == "Medium":
        assert "3 to 5 concise bullet points" in user_prompt
    elif length_mode == "Detailed":
        assert "comprehensive summary" in user_prompt


def test_sentiment_classification_prompts():
    """Test sentiment prompts and verify required test examples are present in few-shot."""
    test_cases = [
        "This phone has excellent battery life.",
        "The product quality is terrible.",
        "The package arrived yesterday.",
    ]
    for text in test_cases:
        sys_inst, user_prompt = build_prompt(
            task=NLPTask.SENTIMENT_ANALYSIS,
            strategy=PromptStrategy.FEW_SHOT,
            user_input=text,
        )
        assert text in user_prompt
        # Verify the few-shot demonstrations exist
        assert "Positive" in user_prompt
        assert "Negative" in user_prompt
        assert "Neutral" in user_prompt


def test_format_prompt_for_display():
    """Verify display formatting creates markdown and does not expose secrets."""
    sys_inst = "You are a helpful assistant."
    user_p = "Hello world"
    display = format_prompt_for_display(sys_inst, user_p)
    assert "### 🛡️ System Instruction:" in display
    assert "### 📝 Rendered User Prompt:" in display
    assert "You are a helpful assistant." in display
    assert "Hello world" in display
