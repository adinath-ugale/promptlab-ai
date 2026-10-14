"""PromptLab AI - Google Gemini API Client Module.

Integrates with the official `google-genai` Python SDK to communicate with Gemini models.
Provides resilient error handling, quota limits reporting, latency tracking, and connectivity checks.
"""

import os
import time
from typing import Optional, Tuple, List, Dict, Any
from dotenv import load_dotenv

# Load local environment variables from .env file if present
load_dotenv()

# Supported models list with recommended default
DEFAULT_MODEL = "gemini-2.5-flash"
SUPPORTED_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]


def get_api_key() -> Optional[str]:
    """Retrieve Gemini API key with priority:
    1. Streamlit session state (if user provided in UI)
    2. Streamlit secrets (for Streamlit Community Cloud hosting)
    3. Environment variable `GEMINI_API_KEY` (from .env or OS environment)
    """
    # 1. UI session state override
    try:
        import streamlit as st
        if "custom_api_key" in st.session_state and st.session_state.custom_api_key:
            return st.session_state.custom_api_key.strip()
    except Exception:
        pass

    # 2. Streamlit Cloud Secrets
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
            key = st.secrets["GEMINI_API_KEY"]
            if key and key.strip():
                return key.strip()
    except Exception:
        pass

    # 3. Environment Variable
    env_key = os.environ.get("GEMINI_API_KEY")
    if env_key and env_key.strip() and env_key.strip() != "your_actual_gemini_api_key_here":
        return env_key.strip()

    return None


def get_default_model() -> str:
    """Retrieve default model from environment or Streamlit secrets."""
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "GEMINI_MODEL" in st.secrets:
            return st.secrets["GEMINI_MODEL"].strip()
    except Exception:
        pass

    env_model = os.environ.get("GEMINI_MODEL")
    if env_model and env_model.strip():
        return env_model.strip()

    return DEFAULT_MODEL


def get_gemini_client(api_key: Optional[str] = None):
    """Instantiate and return the official google-genai Client.
    
    Args:
        api_key: Optional explicit API key. If omitted, resolved automatically.
        
    Returns:
        genai.Client instance or None if API key is missing.
    """
    resolved_key = api_key or get_api_key()
    if not resolved_key:
        return None

    try:
        from google import genai
        return genai.Client(api_key=resolved_key)
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize Gemini Client: {exc}") from exc


def format_api_error(exc: Exception) -> str:
    """Translate raw API exceptions into actionable, user-friendly error messages."""
    err_str = str(exc)
    err_lower = err_str.lower()

    if "resource_exhausted" in err_lower or "429" in err_lower or "quota" in err_lower:
        return (
            "⚠️ **API Quota Exceeded (HTTP 429)**: Your Gemini API key has temporarily reached its rate or quota limit. "
            "Please wait 30–60 seconds before retrying, or check your quota in Google AI Studio."
        )
    if "api_key_invalid" in err_lower or "unauthenticated" in err_lower or "401" in err_lower or "403" in err_lower:
        return (
            "🔒 **Invalid API Key (HTTP 403/401)**: The configured Gemini API key was rejected. "
            "Please verify your key in `.env`, Streamlit Secrets, or the sidebar settings."
        )
    if "not_found" in err_lower or "404" in err_lower:
        return (
            "🔍 **Model Not Found (HTTP 404)**: The requested Gemini model ID is not accessible with this API key. "
            "Try switching to `gemini-2.5-flash` or `gemini-1.5-flash` in settings."
        )
    if "connection" in err_lower or "timeout" in err_lower or "name resolution" in err_lower:
        return (
            "🌐 **Network / Timeout Error**: Unable to reach Google AI servers. "
            "Please check your internet connection and proxy settings."
        )
    return f"❌ **Gemini API Error**: {err_str}"


def verify_connectivity(client: Any, model: str) -> Tuple[bool, str, float]:
    """Perform a live, lightweight API ping to verify actual Gemini connectivity.
    
    Returns:
        (is_connected, message, latency_ms)
    """
    if client is None:
        return False, "No Gemini API key configured. Provide an API key to connect.", 0.0

    start_time = time.perf_counter()
    try:
        from google.genai import types
        # Ultra-lightweight ping request
        response = client.models.generate_content(
            model=model,
            contents="Ping test: Respond strictly with 'PONG'.",
            config=types.GenerateContentConfig(
                max_output_tokens=10,
                temperature=0.0,
            ),
        )
        latency = (time.perf_counter() - start_time) * 1000

        if response and response.text:
            return True, f"Connection verified successfully with '{model}'!", round(latency, 1)
        return False, "Received empty response from Gemini API during ping test.", round(latency, 1)

    except Exception as exc:
        latency = (time.perf_counter() - start_time) * 1000
        return False, format_api_error(exc), round(latency, 1)


def generate_single_response(
    client: Any,
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    system_instruction: Optional[str] = None,
    max_output_tokens: Optional[int] = None,
) -> Tuple[Optional[str], Optional[str], float]:
    """Execute a single-turn generation with Gemini.
    
    Args:
        client: genai.Client instance
        prompt: User input prompt
        model: Gemini model ID
        temperature: Sampling temperature (0.0 to 1.0)
        system_instruction: Optional persona / system instruction
        max_output_tokens: Optional output token cap
        
    Returns:
        (generated_text, error_message, latency_ms)
    """
    if client is None:
        return None, "API Key is missing. Please configure GEMINI_API_KEY.", 0.0

    start_time = time.perf_counter()
    try:
        from google.genai import types

        config_kwargs: Dict[str, Any] = {"temperature": float(temperature)}
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if max_output_tokens:
            config_kwargs["max_output_tokens"] = int(max_output_tokens)

        config = types.GenerateContentConfig(**config_kwargs)

        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )

        latency = (time.perf_counter() - start_time) * 1000

        if response and response.text:
            return response.text.strip(), None, round(latency, 1)

        # Handle blocked response / safety filter
        if hasattr(response, "candidates") and response.candidates:
            finish_reason = getattr(response.candidates[0], "finish_reason", "UNKNOWN")
            return None, f"Response generation halted by safety filter: {finish_reason}", round(latency, 1)

        return None, "Gemini returned an empty response.", round(latency, 1)

    except Exception as exc:
        latency = (time.perf_counter() - start_time) * 1000
        return None, format_api_error(exc), round(latency, 1)


def generate_chat_response(
    client: Any,
    history: List[Dict[str, str]],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    system_instruction: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str], float]:
    """Generate a response in an ongoing multi-turn chat conversation.
    
    Args:
        client: genai.Client instance
        history: List of dicts with 'role' ('user' or 'assistant'/'model') and 'content'
        model: Gemini model ID
        temperature: Sampling temperature
        system_instruction: System prompt
        
    Returns:
        (assistant_reply, error_message, latency_ms)
    """
    if client is None:
        return None, "API Key is missing. Please configure GEMINI_API_KEY.", 0.0

    if not history:
        return None, "No messages in conversation history.", 0.0

    start_time = time.perf_counter()
    try:
        from google.genai import types

        contents = []
        for msg in history:
            role = msg.get("role", "user")
            # Map 'assistant' to 'model' for Gemini standard role naming
            mapped_role = "model" if role in ("assistant", "model") else "user"
            text_content = msg.get("content", "")
            contents.append(
                types.Content(
                    role=mapped_role,
                    parts=[types.Part.from_text(text=text_content)],
                )
            )

        config_kwargs: Dict[str, Any] = {"temperature": float(temperature)}
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        config = types.GenerateContentConfig(**config_kwargs)

        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )

        latency = (time.perf_counter() - start_time) * 1000

        if response and response.text:
            return response.text.strip(), None, round(latency, 1)

        if hasattr(response, "candidates") and response.candidates:
            finish_reason = getattr(response.candidates[0], "finish_reason", "UNKNOWN")
            return None, f"Response generation halted by safety filter: {finish_reason}", round(latency, 1)

        return None, "Gemini returned an empty response.", round(latency, 1)

    except Exception as exc:
        latency = (time.perf_counter() - start_time) * 1000
        return None, format_api_error(exc), round(latency, 1)
