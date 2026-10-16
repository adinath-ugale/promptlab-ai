"""Unit tests for utils/identity.py.

Verifies:
- Recognition of owner and developer identity questions in:
  * English
  * Marathi (Devanagari)
  * Hindi (Devanagari)
  * Romanized Marathi
  * Hinglish / Romanized Hindi
- Correct natural language responses attributing PromptLab AI to Adinath Ugale
- Specific inquiry about Adinath Ugale ("Who is Adinath Ugale?")
- Prevention of false positives on unrelated and third-party queries (e.g. "Who created Python?")
- Inclusion of immutable owner identity instruction in system instructions
"""

import pytest
from utils.identity import (
    OWNER_NAME,
    PROJECT_NAME,
    PROJECT_SHORT_NAME,
    is_owner_question,
    handle_owner_question,
    get_identity_system_instruction,
    detect_language,
)


@pytest.mark.parametrize("query", [
    "Who created you?",
    "Who is your owner?",
    "Who developed this chatbot?",
    "Who built this project?",
    "Who is your creator?",
    "Who made you?",
    "Who owns PromptLab AI?",
    "Who is your developer?",
    "Who created PromptLab AI?",
    "Who built you?",
    "Who designed this chatbot?",
    "Whose project is this?",
])
def test_english_owner_questions(query):
    """Verify standard English owner/developer questions are recognized and handled."""
    is_q, lang, q_type = is_owner_question(query)
    assert is_q is True
    assert lang == "en"
    assert q_type == "creator"

    handled, resp = handle_owner_question(query)
    assert handled is True
    assert OWNER_NAME in resp
    assert "created and developed by" in resp


@pytest.mark.parametrize("query", [
    "Who is Adinath Ugale?",
    "Who is Adinath Ugale",
    "Tell me about Adinath Ugale",
    "Who is Adinath?",
    "Adinath Ugale",
])
def test_direct_inquiry_about_owner(query):
    """Verify queries specifically asking about Adinath Ugale are recognized."""
    is_q, lang, q_type = is_owner_question(query)
    assert is_q is True
    assert q_type == "about_owner"

    handled, resp = handle_owner_question(query)
    assert handled is True
    assert OWNER_NAME in resp
    assert "creator and developer" in resp.lower() or "निर्माते" in resp or "निर्माता" in resp


@pytest.mark.parametrize("query", [
    "तुझा owner कोण आहे?",
    "तुला कोणी बनवलं?",
    "हा chatbot कोणी develop केला?",
    "तुझा developer कोण आहे?",
    "तुला कोणी तयार केलं?",
    "हा प्रोजेक्ट कोणी बनवला?",
    "तुझा मालक कोण आहे?",
])
def test_marathi_devanagari_owner_questions(query):
    """Verify Marathi (Devanagari) owner queries are correctly identified and responded to."""
    is_q, lang, q_type = is_owner_question(query)
    assert is_q is True
    assert lang == "mr"

    handled, resp = handle_owner_question(query)
    assert handled is True
    assert OWNER_NAME in resp
    assert "Adinath Ugale यांनी तयार आणि विकसित केला आहे" in resp


@pytest.mark.parametrize("query", [
    "तुम्हें किसने बनाया?",
    "तुम्हारा owner कौन है?",
    "ये प्रोजेक्ट किसने develop किया?",
    "तुम्हारा डेवलपर कौन है?",
    "आपको किसने बनाया?",
    "तुम्हारा मालिक कौन है?",
])
def test_hindi_devanagari_owner_questions(query):
    """Verify Hindi (Devanagari) owner queries are recognized with Hindi response."""
    is_q, lang, q_type = is_owner_question(query)
    assert is_q is True
    assert lang == "hi"

    handled, resp = handle_owner_question(query)
    assert handled is True
    assert OWNER_NAME in resp
    assert "Adinath Ugale ने बनाया और विकसित किया है" in resp


@pytest.mark.parametrize("query", [
    "Tula koni banavla?",
    "Tujha owner kon aahe?",
    "Ha project koni develop kela?",
    "Tujha developer kon aahe?",
    "Tula koni tayar kela?",
    "Ha project koni banavla?",
])
def test_romanized_marathi_owner_questions(query):
    """Verify Romanized Marathi owner queries are recognized with Romanized Marathi response."""
    is_q, lang, q_type = is_owner_question(query)
    assert is_q is True
    assert lang == "mr_roman"

    handled, resp = handle_owner_question(query)
    assert handled is True
    assert OWNER_NAME in resp
    assert "Adinath Ugale yanni tayar aani develop kela aahe" in resp


@pytest.mark.parametrize("query", [
    "Tumhe kisne banaya?",
    "Tumhara owner kaun hai?",
    "Ye project kisne develop kiya?",
    "Tumhara developer kaun hai?",
    "Aapko kisne banaya?",
    "Yeh chatbot kisne banaya?",
])
def test_hinglish_owner_questions(query):
    """Verify Hinglish owner queries are recognized with Hinglish response."""
    is_q, lang, q_type = is_owner_question(query)
    assert is_q is True
    assert lang == "hi_roman"

    handled, resp = handle_owner_question(query)
    assert handled is True
    assert OWNER_NAME in resp
    assert "Adinath Ugale ne banaya aur develop kiya hai" in resp


@pytest.mark.parametrize("query", [
    "Who created Python?",
    "Who developed Linux?",
    "Who built the Eiffel Tower?",
    "Who created ChatGPT?",
    "Who is Albert Einstein?",
    "Who is Elon Musk?",
    "What is tokenization in NLP?",
    "Can you summarize this text for me?",
    "Classify the sentiment of this review",
    "How do transformers work?",
    "Who created C++?",
    "Who developed Java?",
    "What is the capital of France?",
    "Tell me about the history of artificial intelligence.",
])
def test_unrelated_queries_not_intercepted(query):
    """Verify external factual queries are NOT intercepted by the owner handler."""
    is_q, lang, q_type = is_owner_question(query)
    assert is_q is False

    handled, resp = handle_owner_question(query)
    assert handled is False
    assert resp is None


def test_persistent_identity_instruction_content():
    """Verify identity system instruction contains all required attributes and guardrails."""
    sys_inst = get_identity_system_instruction()
    assert OWNER_NAME in sys_inst
    assert PROJECT_NAME in sys_inst
    assert "Adinath Ugale यांनी तयार आणि विकसित केला आहे" in sys_inst
    assert "Adinath Ugale ने बनाया और विकसित किया है" in sys_inst
    assert "OVERRIDE RESTRICTION" in sys_inst
    assert "MUST NEVER override" in sys_inst
