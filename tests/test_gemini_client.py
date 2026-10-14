"""Unit tests for utils/gemini_client.py (Mocked Gemini API Integration).

Verifies:
- Error translation (429 Quota, 401/403 Invalid Key, 404 Model Not Found, Network Timeouts)
- API key resolution precedence and placeholder rejection
- Safe client invocation and latency calculation without calling live Google endpoints
"""

import os
from unittest.mock import MagicMock, patch
import pytest
from utils.gemini_client import (
    format_api_error,
    get_api_key,
    verify_connectivity,
    generate_single_response,
    generate_chat_response,
    DEFAULT_MODEL,
)


def test_format_api_error():
    """Verify friendly error messages are returned for known API failure modes."""
    # 1. 429 Quota Exhaustion
    err_429 = Exception("ResourceExhausted: 429 Quota exceeded for quota group")
    msg_429 = format_api_error(err_429)
    assert "Quota Exceeded" in msg_429
    assert "429" in msg_429

    # 2. 403 / 401 Invalid Key
    err_403 = Exception("PermissionDenied: 403 API_KEY_INVALID")
    msg_403 = format_api_error(err_403)
    assert "Invalid API Key" in msg_403

    # 3. 404 Model Not Found
    err_404 = Exception("NotFound: 404 Model gemini-xyz not found")
    msg_404 = format_api_error(err_404)
    assert "Model Not Found" in msg_404

    # 4. Connection / Network Timeout
    err_net = Exception("TimeoutError: Name resolution or connection failed")
    msg_net = format_api_error(err_net)
    assert "Network / Timeout Error" in msg_net


def test_get_api_key_with_placeholder():
    """Verify template placeholder is rejected as None."""
    with patch.dict(os.environ, {"GEMINI_API_KEY": "your_actual_gemini_api_key_here"}):
        key = get_api_key()
        assert key is None


def test_get_api_key_valid_env():
    """Verify valid environment key is retrieved."""
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test_secret_key_12345"}):
        key = get_api_key()
        assert key == "test_secret_key_12345"


def test_verify_connectivity_none_client():
    """Verify connectivity returns False gracefully when client is None."""
    is_ok, msg, lat = verify_connectivity(None, DEFAULT_MODEL)
    assert is_ok is False
    assert "No Gemini API key" in msg


def test_verify_connectivity_mock_success():
    """Verify ping success with a mocked client."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "PONG"
    mock_client.models.generate_content.return_value = mock_response

    is_ok, msg, lat = verify_connectivity(mock_client, DEFAULT_MODEL)
    assert is_ok is True
    assert "verified successfully" in msg
    assert lat >= 0.0


def test_generate_single_response_mock_success():
    """Verify single turn generation with a mocked client."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "This is a high quality AI answer."
    mock_client.models.generate_content.return_value = mock_response

    text, err, lat = generate_single_response(
        client=mock_client,
        prompt="Explain tokenization",
        model=DEFAULT_MODEL,
    )
    assert text == "This is a high quality AI answer."
    assert err is None
    assert lat >= 0.0


def test_generate_single_response_mock_error():
    """Verify graceful handling when client raises an exception."""
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("429 ResourceExhausted")

    text, err, lat = generate_single_response(
        client=mock_client,
        prompt="Explain tokenization",
    )
    assert text is None
    assert err is not None
    assert "Quota Exceeded" in err


def test_generate_chat_response_mock_success():
    """Verify multi-turn chat handling and role mapping."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Hello! I am ready to help you with NLP."
    mock_client.models.generate_content.return_value = mock_response

    history = [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "What is an embedding?"},
    ]

    reply, err, lat = generate_chat_response(
        client=mock_client,
        history=history,
        model=DEFAULT_MODEL,
    )
    assert reply == "Hello! I am ready to help you with NLP."
    assert err is None
    assert mock_client.models.generate_content.called
