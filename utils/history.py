"""PromptLab AI - Session History and State Management Module.

Manages conversational message history, recent chat threads, evaluation logs, and session metrics
in Streamlit session state. Ensures privacy by strictly excluding credentials or sensitive tokens.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import streamlit as st


def init_session_state() -> None:
    """Initialize default keys in Streamlit session state if not already set."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "recent_chats" not in st.session_state:
        # Initial starter chat threads for elegant visual presentation
        st.session_state.recent_chats = [
            {
                "id": "starter_1",
                "title": "Transformer Attention Mechanics",
                "preview": "Explain Scaled Dot-Product Attention...",
                "messages": [],
            },
            {
                "id": "starter_2",
                "title": "Sentiment Polarity Analysis",
                "preview": "Classify text into Positive/Negative...",
                "messages": [],
            },
            {
                "id": "starter_3",
                "title": "Zero vs Few-Shot Prompting",
                "preview": "In-context learning benchmarks...",
                "messages": [],
            },
        ]

    if "active_chat_id" not in st.session_state:
        st.session_state.active_chat_id = "current"

    if "evaluation_logs" not in st.session_state:
        st.session_state.evaluation_logs = []

    if "queries_count" not in st.session_state:
        st.session_state.queries_count = 0

    if "connection_status" not in st.session_state:
        st.session_state.connection_status = None

    if "custom_api_key" not in st.session_state:
        st.session_state.custom_api_key = ""

    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "gemini-3.5-flash-lite"

    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.7

    if "theme_mode" not in st.session_state:
        st.session_state.theme_mode = "dark"

    if "active_nav" not in st.session_state:
        st.session_state.active_nav = "💬 Chat"

    if "active_view" not in st.session_state:
        st.session_state.active_view = "chat"

    if "prompt_strategy" not in st.session_state:
        st.session_state.prompt_strategy = "Zero-Shot"



def add_chat_message(
    role: str,
    content: str,
    latency: Optional[float] = None,
    model: Optional[str] = None,
) -> None:
    """Append a message to the active chat history in session state.
    
    Args:
        role: "user" or "assistant"
        content: message text
        latency: optional execution time in milliseconds
        model: optional model identifier
    """
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    msg = {
        "role": role,
        "content": content,
        "timestamp": datetime.now().strftime("%I:%M %p"),
    }
    if latency is not None:
        msg["latency"] = latency
    if model:
        msg["model"] = model

    st.session_state.chat_history.append(msg)

    # If first user message, update recent chat title
    if role == "user":
        st.session_state.queries_count = st.session_state.get("queries_count", 0) + 1
        title = content.strip().replace("\n", " ")
        if len(title) > 30:
            title = title[:27] + "..."

        # Update or add to recent chats
        current_id = st.session_state.get("active_chat_id", "current")
        updated = False
        if "recent_chats" in st.session_state:
            for item in st.session_state.recent_chats:
                if item["id"] == current_id:
                    item["title"] = title
                    item["messages"] = list(st.session_state.chat_history)
                    updated = True
                    break
            if not updated:
                st.session_state.recent_chats.insert(
                    0,
                    {
                        "id": current_id,
                        "title": title,
                        "preview": content[:40],
                        "messages": list(st.session_state.chat_history),
                    },
                )


def get_chat_history() -> List[Dict[str, str]]:
    """Return the current session's chat history list."""
    return st.session_state.get("chat_history", [])


def clear_chat_history() -> None:
    """Reset the chat history for the active session."""
    st.session_state.chat_history = []
    st.session_state.active_chat_id = f"chat_{int(datetime.now().timestamp())}"


def load_chat_thread(thread_id: str) -> None:
    """Load a specific chat thread from recent_chats into active chat_history."""
    if "recent_chats" in st.session_state:
        for t in st.session_state.recent_chats:
            if t["id"] == thread_id:
                st.session_state.active_chat_id = thread_id
                st.session_state.chat_history = list(t.get("messages", []))
                break


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
