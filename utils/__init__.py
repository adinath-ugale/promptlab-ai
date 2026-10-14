"""PromptLab AI - Utility Package for LLM Chatbot and Prompt Engineering."""

from .prompts import (
    PromptStrategy,
    NLPTask,
    build_prompt,
    get_system_instruction,
    format_prompt_for_display,
    validate_input_text,
)
from .gemini_client import (
    get_gemini_client,
    get_api_key,
    verify_connectivity,
    generate_chat_response,
    generate_single_response,
    DEFAULT_MODEL,
    SUPPORTED_MODELS,
)
from .evaluation import (
    calculate_rqs,
    validate_score,
    evaluate_response_with_llm,
    create_evaluation_record,
    records_to_dataframe,
    export_evaluations_to_csv,
    RQSResult,
)
from .history import (
    init_session_state,
    add_chat_message,
    clear_chat_history,
    get_chat_history,
    record_evaluation,
    clear_evaluations,
    get_session_stats,
)

__all__ = [
    "PromptStrategy",
    "NLPTask",
    "build_prompt",
    "get_system_instruction",
    "format_prompt_for_display",
    "validate_input_text",
    "get_gemini_client",
    "get_api_key",
    "verify_connectivity",
    "generate_chat_response",
    "generate_single_response",
    "DEFAULT_MODEL",
    "SUPPORTED_MODELS",
    "calculate_rqs",
    "validate_score",
    "evaluate_response_with_llm",
    "create_evaluation_record",
    "records_to_dataframe",
    "export_evaluations_to_csv",
    "RQSResult",
    "init_session_state",
    "add_chat_message",
    "clear_chat_history",
    "get_chat_history",
    "record_evaluation",
    "clear_evaluations",
    "get_session_stats",
]
