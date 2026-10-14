"""PromptLab AI - Prompt Engineering & Template Management Module.

Provides prompt strategies (Zero-Shot, Few-Shot, Role Prompting, Chain-of-Thought Rationale)
and templates for core NLP tasks: Question Answering, Text Summarization, and Sentiment Classification.
"""

from enum import Enum
from typing import Optional, Dict, Any, Tuple


class NLPTask(str, Enum):
    """Supported NLP Tasks."""
    QUESTION_ANSWERING = "Question Answering"
    SUMMARIZATION = "Text Summarization"
    SENTIMENT_ANALYSIS = "Sentiment Classification"


class PromptStrategy(str, Enum):
    """Prompt Engineering Strategies."""
    ZERO_SHOT = "Zero-Shot Prompting"
    FEW_SHOT = "Few-Shot Prompting"
    ROLE_PROMPT = "Role Prompting"
    CHAIN_OF_THOUGHT = "Chain-of-Thought (Rationale)"


# ---------------------------------------------------------------------------
# Task-Specific System Instructions (Grounding & Guardrails)
# ---------------------------------------------------------------------------
SYSTEM_INSTRUCTIONS: Dict[NLPTask, str] = {
    NLPTask.QUESTION_ANSWERING: (
        "You are an expert AI tutor and technical assistant specializing in Natural Language Processing and Computer Science. "
        "Provide structured, accurate, and beginner-friendly explanations. "
        "If you are uncertain of a fact or the answer is not known, explicitly state your uncertainty "
        "instead of inventing or hallucinating facts."
    ),
    NLPTask.SUMMARIZATION: (
        "You are a precise technical summarizer. Extract and preserve the core facts, "
        "key metrics, and crucial takeaways from the provided text. "
        "Strictly avoid hallucinating or introducing external assumptions not grounded in the source text."
    ),
    NLPTask.SENTIMENT_ANALYSIS: (
        "You are an NLP sentiment analysis classifier. Classify the user-supplied text into "
        "exactly one of three categories: Positive, Negative, or Neutral. "
        "Provide a concise rationale explaining the linguistic cues that led to your decision."
    ),
}

# ---------------------------------------------------------------------------
# Role Personas
# ---------------------------------------------------------------------------
ROLE_PERSONAS: Dict[NLPTask, str] = {
    NLPTask.QUESTION_ANSWERING: (
        "You are Professor Alan, an award-winning university professor and Senior NLP Researcher. "
        "You explain complex computing concepts using intuitive real-world analogies, step-by-step clarity, "
        "and academic rigor while remaining encouraging and accessible to students."
    ),
    NLPTask.SUMMARIZATION: (
        "You are a Chief Editor and Executive Intelligence Analyst. Your expertise is distilling "
        "dense, complicated reports into crystal-clear executive briefings without losing critical nuances or data."
    ),
    NLPTask.SENTIMENT_ANALYSIS: (
        "You are an elite Computational Linguist and Sentiment Analyst. You analyze semantic valence, "
        "tonal polarity, sentiment markers, and context with clinical precision."
    ),
}

# ---------------------------------------------------------------------------
# Few-Shot Demonstrations
# ---------------------------------------------------------------------------
FEW_SHOT_EXAMPLES: Dict[NLPTask, str] = {
    NLPTask.QUESTION_ANSWERING: (
        "Example 1:\n"
        "Question: What is Tokenization in NLP?\n"
        "Answer: Tokenization is the process of breaking down a stream of text into smaller units called tokens "
        "(such as words, subwords, or punctuation symbols). For example, the sentence 'NLP is amazing!' "
        "can be tokenized into ['NLP', 'is', 'amazing', '!']. It is the foundational first step for any text processing pipeline.\n\n"
        "Example 2:\n"
        "Question: How does Word2Vec represent word meanings?\n"
        "Answer: Word2Vec represents words as dense continuous vectors in an N-dimensional space based on the Distributional Hypothesis: "
        "'words that appear in similar contexts share similar meanings'. For instance, vectors for 'king' - 'man' + 'woman' approximate 'queen'."
    ),
    NLPTask.SUMMARIZATION: (
        "Example 1:\n"
        "Source Text: Solar energy is the conversion of energy from sunlight into electricity, either directly using photovoltaics "
        "or indirectly using concentrated solar power. Photovoltaic cells convert light into an electric current using the photovoltaic effect. "
        "Solar power is a rapidly growing renewable resource that significantly reduces greenhouse gas emissions.\n"
        "Summary: Solar energy converts sunlight into electricity via photovoltaics or concentrated thermal power, providing a rapidly expanding "
        "clean renewable alternative that cuts greenhouse emissions.\n\n"
        "Example 2:\n"
        "Source Text: Machine learning models require significant preprocessing of input data. Missing values must be imputed or removed, "
        "categorical variables need encoding, and numerical features often require normalization or standardization to ensure smooth gradient descent.\n"
        "Summary: Effective machine learning relies on data preprocessing, including handling missing values, encoding categories, and scaling numeric features."
    ),
    NLPTask.SENTIMENT_ANALYSIS: (
        "Example 1:\n"
        "Input: 'This phone has excellent battery life and charges remarkably fast.'\n"
        "Label: Positive\n"
        "Explanation: The words 'excellent' and 'remarkably fast' convey strong user satisfaction and high product performance.\n\n"
        "Example 2:\n"
        "Input: 'The product quality is terrible, and customer support refused to help.'\n"
        "Label: Negative\n"
        "Explanation: 'Terrible' quality and 'refused to help' indicate dissatisfaction, negative emotion, and unresolved grievance.\n\n"
        "Example 3:\n"
        "Input: 'The package arrived yesterday afternoon at 3 PM.'\n"
        "Label: Neutral\n"
        "Explanation: The statement conveys an objective, factual timeline of delivery without emotional polarity or opinion."
    ),
}


# ---------------------------------------------------------------------------
# Input Validation
# ---------------------------------------------------------------------------
def validate_input_text(text: Optional[str], min_length: int = 1, max_length: int = 15000) -> Tuple[bool, Optional[str]]:
    """Validate user input text for emptiness and length boundaries.
    
    Returns:
        (is_valid, error_message)
    """
    if text is None or not text.strip():
        return False, "Input cannot be empty. Please enter text to proceed."
    
    cleaned = text.strip()
    if len(cleaned) < min_length:
        return False, f"Input is too short (minimum {min_length} characters required)."
        
    if len(cleaned) > max_length:
        return False, f"Input exceeds maximum allowed length ({len(cleaned)}/{max_length} characters). Please shorten your input."
        
    return True, None


# ---------------------------------------------------------------------------
# Prompt Building Functions
# ---------------------------------------------------------------------------
def get_system_instruction(task: NLPTask, strategy: PromptStrategy) -> str:
    """Retrieve the base system instruction or role persona based on strategy."""
    if strategy == PromptStrategy.ROLE_PROMPT:
        persona = ROLE_PERSONAS.get(task, SYSTEM_INSTRUCTIONS.get(task, ""))
        return f"{persona}\nAdhere strictly to factual grounding and answer truthfully."
    return SYSTEM_INSTRUCTIONS.get(task, "You are a helpful and accurate AI assistant.")


def build_prompt(
    task: NLPTask,
    strategy: PromptStrategy,
    user_input: str,
    extra_params: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str]:
    """Build the final system instruction and user prompt string.
    
    Args:
        task: NLPTask enum value
        strategy: PromptStrategy enum value
        user_input: The input text from user
        extra_params: Additional task options (e.g. summary length: short/medium/detailed)
        
    Returns:
        (system_instruction, user_prompt)
    """
    params = extra_params or {}
    system_inst = get_system_instruction(task, strategy)
    cleaned_input = user_input.strip()

    if task == NLPTask.QUESTION_ANSWERING:
        user_prompt = _build_qa_prompt(strategy, cleaned_input)
    elif task == NLPTask.SUMMARIZATION:
        length_mode = params.get("length_mode", "Medium")
        user_prompt = _build_summarization_prompt(strategy, cleaned_input, length_mode)
    elif task == NLPTask.SENTIMENT_ANALYSIS:
        user_prompt = _build_sentiment_prompt(strategy, cleaned_input)
    else:
        user_prompt = cleaned_input

    return system_inst, user_prompt


def _build_qa_prompt(strategy: PromptStrategy, user_input: str) -> str:
    """Build prompt for Question Answering."""
    guidance = (
        "Guidelines:\n"
        "- Explain concepts clearly with beginner-friendly language and intuitive examples.\n"
        "- Structure your answer with clear headings or bullet points where beneficial.\n"
        "- If uncertain or if the premise is factually flawed, state so honestly.\n"
    )
    
    if strategy == PromptStrategy.ZERO_SHOT:
        return f"{guidance}\nQuestion:\n{user_input}\n\nAnswer:"
        
    elif strategy == PromptStrategy.FEW_SHOT:
        examples = FEW_SHOT_EXAMPLES[NLPTask.QUESTION_ANSWERING]
        return (
            f"Here are demonstrations of high quality, grounded technical answers:\n\n"
            f"{examples}\n\n"
            f"Now answer the following question following the same standards:\n"
            f"{guidance}\n"
            f"Question:\n{user_input}\n\nAnswer:"
        )
        
    elif strategy == PromptStrategy.ROLE_PROMPT:
        return (
            f"As an experienced Professor and AI Educator, answer the following student question:\n"
            f"{guidance}\n"
            f"Question:\n{user_input}\n\n"
            f"Professor's Explanation:"
        )
        
    elif strategy == PromptStrategy.CHAIN_OF_THOUGHT:
        return (
            f"{guidance}\n"
            f"Question:\n{user_input}\n\n"
            f"Please structure your response in two clear sections:\n"
            f"1. Core Conceptual Analysis (concise reasoning rationale)\n"
            f"2. Final Direct Answer\n\n"
            f"Response:"
        )
    return user_input


def _build_summarization_prompt(strategy: PromptStrategy, user_input: str, length_mode: str) -> str:
    """Build prompt for Text Summarization."""
    length_specs = {
        "Short": "a concise summary in 1 to 2 clear sentences capturing the core conclusion.",
        "Medium": "a well-structured summary of 3 to 5 concise bullet points highlighting key insights.",
        "Detailed": "a comprehensive summary organized with an overview, key points, and strategic takeaways.",
    }
    target_spec = length_specs.get(length_mode, length_specs["Medium"])

    base_instruction = (
        f"Task: Summarize the following text.\n"
        f"Target format: Provide {target_spec}\n"
        f"Constraint: Preserve facts, quantities, and dates faithfully. Do not add outside assumptions.\n\n"
    )

    if strategy == PromptStrategy.ZERO_SHOT:
        return f"{base_instruction}Source Text:\n\"\"\"\n{user_input}\n\"\"\"\n\nSummary:"

    elif strategy == PromptStrategy.FEW_SHOT:
        examples = FEW_SHOT_EXAMPLES[NLPTask.SUMMARIZATION]
        return (
            f"Review these reference summarization demonstrations:\n\n"
            f"{examples}\n\n"
            f"Now summarize this source text following the same factual precision:\n"
            f"{base_instruction}"
            f"Source Text:\n\"\"\"\n{user_input}\n\"\"\"\n\nSummary:"
        )

    elif strategy == PromptStrategy.ROLE_PROMPT:
        return (
            f"In your capacity as an Executive Intelligence Editor, prepare an authoritative brief:\n"
            f"{base_instruction}"
            f"Source Text:\n\"\"\"\n{user_input}\n\"\"\"\n\nExecutive Briefing:"
        )

    elif strategy == PromptStrategy.CHAIN_OF_THOUGHT:
        return (
            f"{base_instruction}"
            f"Source Text:\n\"\"\"\n{user_input}\n\"\"\"\n\n"
            f"First, briefly list the 3 most crucial factual anchors from the text.\n"
            f"Then, present the cohesive final summary.\n\n"
            f"Summary:"
        )
    return user_input


def _build_sentiment_prompt(strategy: PromptStrategy, user_input: str) -> str:
    """Build prompt for Sentiment Classification."""
    instructions = (
        "Task: Classify the sentiment of the provided text into exactly one category: Positive, Negative, or Neutral.\n"
        "Output format requirements:\n"
        "Label: <Positive | Negative | Neutral>\n"
        "Explanation: <One to two sentences explaining the semantic sentiment markers>\n"
    )

    if strategy == PromptStrategy.ZERO_SHOT:
        return f"{instructions}\nInput Text:\n\"{user_input}\"\n\nLabel:"

    elif strategy == PromptStrategy.FEW_SHOT:
        examples = FEW_SHOT_EXAMPLES[NLPTask.SENTIMENT_ANALYSIS]
        return (
            f"Study the following sentiment classification benchmarks:\n\n"
            f"{examples}\n\n"
            f"Now classify the new text using the exact same format:\n"
            f"{instructions}\n"
            f"Input Text:\n\"{user_input}\"\n\n"
            f"Classification:"
        )

    elif strategy == PromptStrategy.ROLE_PROMPT:
        return (
            f"Operating as a Computational Linguist and Sentiment Analyst, evaluate the polarity of this text:\n"
            f"{instructions}\n"
            f"Input Text:\n\"{user_input}\"\n\n"
            f"Linguistic Evaluation:"
        )

    elif strategy == PromptStrategy.CHAIN_OF_THOUGHT:
        return (
            f"Evaluate the sentiment of the following text with explicit rationale:\n"
            f"Input Text:\n\"{user_input}\"\n\n"
            f"1. Polarity Analysis: (Identify emotional cues, tone, or neutral statements)\n"
            f"2. Label: <Positive | Negative | Neutral>\n"
            f"3. Explanation: (Concise justification)\n"
        )
    return user_input


def format_prompt_for_display(system_instruction: str, user_prompt: str) -> str:
    """Format the full prompt cleanly for display in UI without leaking secrets."""
    return (
        f"### 🛡️ System Instruction:\n"
        f"```text\n{system_instruction}\n```\n\n"
        f"### 📝 Rendered User Prompt:\n"
        f"```text\n{user_prompt}\n```"
    )
