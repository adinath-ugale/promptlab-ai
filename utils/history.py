"""PromptLab AI - Session History and State Management Module.

Manages conversational message history, evaluation logs, and session metrics in Streamlit
session state. Ensures privacy by strictly excluding credentials or sensitive tokens from state.
"""

from typing import List, Dict, Any, Optional
import streamlit as st


def init_session_state() -> None:
    """Initialize default keys in Streamlit session state if not already set."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "evaluation_logs" not in st.session_state:
        st.session_state.evaluation_logs = []

    if "queries_count" not in st.session_state:
        st.session_state.queries_count = 0

    if "connection_status" not in st.session_state:
        st.session_state.connection_status = None

    if "custom_api_key" not in st.session_state:
        st.session_state.custom_api_key = ""

    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "gemini-2.5-flash"

    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.7


def add_chat_message(role: str, content: str) -> None:
    """Append a message to the active chat history in session state.
    
    Args:
        role: "user" or "assistant"
        content: message text
    """
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.session_state.chat_history.append({"role": role, "content": content})
    if role == "user":
        st.session_state.queries_count = st.session_state.get("queries_count", 0) + 1


def get_chat_history() -> List[Dict[str, str]]:
    """Return the current session's chat history list."""
    return st.session_state.get("chat_history", [])


def clear_chat_history() -> None:
    """Reset the chat history for the active session."""
    st.session_state.chat_history = []


def record_evaluation(result: Any) -> None:
    """Store an RQSResult in session evaluation logs."""
    if "evaluation_logs" not in st.session_state:
        st.session_state.evaluation_logs = []
    st.session_state.evaluation_logs.append(result)


def get_evaluations() -> List[Any]:
    """Retrieve all evaluation records in the current session."""
    return st.session_state.get("evaluation_logs", [])


def clear_evaluations() -> None:
    """Clear all recorded evaluation records."""
    st.session_state.evaluation_logs = []


def get_session_stats() -> Dict[str, Any]:
    """Calculate summary statistics for display in sidebar or dashboard."""
    chat_count = len(st.session_state.get("chat_history", []))
    eval_count = len(st.session_state.get("evaluation_logs", []))
    query_count = st.session_state.get("queries_count", 0)

    eval_logs = st.session_state.get("evaluation_logs", [])
    avg_rqs = (
        round(sum(e.rqs_score for e in eval_logs) / eval_count, 2)
        if eval_count > 0
        else 0.0
    )

    return {
        "chat_messages": chat_count,
        "evaluations_run": eval_count,
        "queries_executed": query_count,
        "average_rqs": avg_rqs,
    }
