"""Integration test calling the actual Gemini API when GEMINI_API_KEY is configured.

Skipped automatically in CI or unit test environments if no live key is present.
"""

import os
import pytest
from utils.gemini_client import get_api_key, get_gemini_client, verify_connectivity, get_default_model

LIVE_KEY = get_api_key()


@pytest.mark.skipif(not LIVE_KEY, reason="No live GEMINI_API_KEY configured; skipping live API ping.")
def test_live_gemini_connectivity():
    """Verify live connectivity and latency against the real Google Gemini API."""
    client = get_gemini_client(LIVE_KEY)
    assert client is not None
    model = get_default_model()
    is_ok, msg, lat = verify_connectivity(client, model)
    if not is_ok and ("getaddrinfo failed" in msg.lower() or "name resolution" in msg.lower()):
        pytest.skip(f"Outbound network unreachable in test sandbox: {msg}")
    assert is_ok is True, f"Ping failed: {msg}"
    assert lat > 0.0



