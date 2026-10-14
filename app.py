"""PromptLab AI - LLM Chatbot with Prompt Engineering.

Main Streamlit Application for College NLP Practical Project.
Integrates Google Gemini API, Prompt Engineering Strategies, and RQS Evaluation Framework.
"""

import time
import streamlit as st
import pandas as pd

from utils.prompts import (
    NLPTask,
    PromptStrategy,
    build_prompt,
    format_prompt_for_display,
    validate_input_text,
    SYSTEM_INSTRUCTIONS,
)
from utils.gemini_client import (
    get_gemini_client,
    get_api_key,
    get_default_model,
    verify_connectivity,
    generate_chat_response,
    generate_single_response,
    SUPPORTED_MODELS,
    DEFAULT_MODEL,
)
from utils.evaluation import (
    calculate_rqs,
    evaluate_response_with_llm,
    create_evaluation_record,
    records_to_dataframe,
    export_evaluations_to_csv,
    RQSResult,
)
from utils.history import (
    init_session_state,
    add_chat_message,
    get_chat_history,
    clear_chat_history,
    record_evaluation,
    get_evaluations,
    clear_evaluations,
    get_session_stats,
)

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PromptLab AI — LLM Chatbot",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Session State
init_session_state()

# ---------------------------------------------------------------------------
# Custom CSS for Modern, Accessible, Polished UI
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Main container styling */
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366F1 0%, #A855F7 50%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
        margin-bottom: 0.8rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #818CF8;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-positive {
        background-color: rgba(34, 197, 94, 0.2);
        color: #4ADE80;
        border: 1px solid #22C55E;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
    }
    .badge-negative {
        background-color: rgba(239, 68, 68, 0.2);
        color: #F87171;
        border: 1px solid #EF4444;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
    }
    .badge-neutral {
        background-color: rgba(59, 130, 246, 0.2);
        color: #60A5FA;
        border: 1px solid #3B82F6;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
    }
    .rqs-formula-box {
        background: rgba(99, 102, 241, 0.1);
        border: 1px solid #6366F1;
        border-radius: 8px;
        padding: 0.8rem 1.2rem;
        font-family: monospace;
        color: #C7D2FE;
        margin: 1rem 0;
    }
    .card-box {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    .prompt-chip {
        display: inline-block;
        background-color: #334155;
        color: #E2E8F0;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.82rem;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar: Navigation, API Configuration & System Health
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🧪 **PromptLab AI**")
    st.caption("LLM Chatbot with Prompt Engineering")
    st.divider()

    # Navigation Section
    st.markdown("### 🧭 **Navigation**")
    selected_page = st.radio(
        "Select Module:",
        options=[
            "💬 AI Chat",
            "❓ Question Answering",
            "📝 Text Summarization",
            "🎭 Sentiment Classification",
            "🔬 Prompt Engineering Lab",
            "📊 Quality Evaluation (RQS)",
            "📖 About Project & Viva",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    # Gemini API Configuration Section
    st.markdown("### ⚙️ **API Configuration**")
    configured_key = get_api_key()

    # Manual key entry if user hasn't set environment variable
    custom_key = st.text_input(
        "Custom Gemini API Key:",
        type="password",
        value=st.session_state.custom_api_key,
        help="Optional override if GEMINI_API_KEY is not in .env or secrets.",
    )
    if custom_key != st.session_state.custom_api_key:
        st.session_state.custom_api_key = custom_key
        st.session_state.connection_status = None
        st.rerun()

    # Model Selection
    default_m = get_default_model()
    default_idx = SUPPORTED_MODELS.index(default_m) if default_m in SUPPORTED_MODELS else 0
    selected_model = st.selectbox(
        "Gemini Model ID:",
        options=SUPPORTED_MODELS,
        index=default_idx,
        help="Select a supported Gemini model available to your API key.",
    )
    st.session_state.selected_model = selected_model

    # Temperature Slider
    temperature = st.slider(
        "Temperature:",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.temperature,
        step=0.05,
        help="0.0 = Deterministic & focused; 1.0 = Creative & varied.",
    )
    st.session_state.temperature = temperature

    # Real API Connectivity Test
    st.markdown("#### 🔌 **API Status**")
    active_key = custom_key.strip() if custom_key.strip() else configured_key

    if not active_key:
        st.warning("⚠️ No API Key found.\nAdd key to `.env` or paste above.")
    else:
        # Masked key indicator (safe)
        masked_key = f"...{active_key[-4:]}" if len(active_key) >= 4 else "configured"
        st.caption(f"🔑 Key configured: `{masked_key}`")

        col_ping, col_status = st.columns([1, 1])
        with col_ping:
            if st.button("Test Connection", use_container_width=True):
                with st.spinner("Pinging Gemini..."):
                    client = get_gemini_client(active_key)
                    is_ok, msg, lat = verify_connectivity(client, selected_model)
                    st.session_state.connection_status = {
                        "is_ok": is_ok,
                        "msg": msg,
                        "lat": lat,
                    }

        if st.session_state.connection_status:
            stat = st.session_state.connection_status
            if stat["is_ok"]:
                st.success(f"🟢 **Connected** ({stat['lat']} ms)\n\n{stat['msg']}")
            else:
                st.error(f"🔴 **Connection Failed**\n\n{stat['msg']}")

    st.divider()

    # Session Statistics
    stats = get_session_stats()
    st.markdown("### 📈 **Session Stats**")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Queries", stats["queries_executed"])
    with c2:
        st.metric("Avg RQS", f"{stats['average_rqs']}/5" if stats["average_rqs"] > 0 else "N/A")

    if st.button("🧹 Clear All Session Data", use_container_width=True):
        clear_chat_history()
        clear_evaluations()
        st.session_state.queries_count = 0
        st.success("Session reset.")
        st.rerun()


# ---------------------------------------------------------------------------
# Helper: Client Resolution
# ---------------------------------------------------------------------------
def resolve_client():
    """Retrieve Gemini client using active key."""
    key = st.session_state.custom_api_key.strip() or get_api_key()
    if not key:
        st.error(
            "🛑 **Gemini API Key Required**\n\n"
            "Please provide your Google Gemini API key by either:\n"
            "1. Creating a `.env` file from `.env.example` with `GEMINI_API_KEY=your_key`\n"
            "2. Adding it to `.streamlit/secrets.toml` or Streamlit Cloud Secrets\n"
            "3. Pasting it into the **Custom Gemini API Key** box in the left sidebar."
        )
        return None
    return get_gemini_client(key)


# ===========================================================================
# PAGE 1: AI CHAT (Multi-Turn Conversational Interface)
# ===========================================================================
if selected_page == "💬 AI Chat":
    st.markdown('<div class="main-header">💬 AI Chat</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Multi-turn conversational assistant with prompt engineering roles and context memory.</div>',
        unsafe_allow_html=True,
    )

    # Chat Persona / Strategy configuration
    top_col1, top_col2, top_col3 = st.columns([2, 1, 1])
    with top_col1:
        persona = st.selectbox(
            "Assistant Role / Persona:",
            options=[
                "NLP & Computer Science Tutor",
                "Technical Software Architect",
                "Concise & Direct Academic Helper",
                "Socratic Inquirer (Guides with questions)",
            ],
            index=0,
        )
    with top_col2:
        st.caption("Active Model:")
        st.code(st.session_state.selected_model, language="text")
    with top_col3:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            clear_chat_history()
            st.rerun()

    persona_prompts = {
        "NLP & Computer Science Tutor": SYSTEM_INSTRUCTIONS[NLPTask.QUESTION_ANSWERING],
        "Technical Software Architect": (
            "You are a Principal Software Architect. Give rigorous, production-grade technical advice, "
            "highlighting scalability, edge cases, and algorithmic complexity."
        ),
        "Concise & Direct Academic Helper": (
            "You are a direct, concise academic assistant. Give high-density, accurate answers without fluff."
        ),
        "Socratic Inquirer (Guides with questions)": (
            "You are a Socratic tutor. Help the user discover solutions by providing concise hints, "
            "clarifying questions, and encouraging logical reasoning."
        ),
    }
    active_sys_inst = persona_prompts[persona]

    # Sample prompt chips if chat is empty
    chat_history = get_chat_history()
    if not chat_history:
        st.info("💡 **Welcome to PromptLab AI!** Click any starter prompt below or type your question at the bottom:")
        sample_cols = st.columns(3)
        sample_prompts = [
            "Explain how the Transformer attention mechanism works.",
            "Compare Word2Vec vs BERT embeddings in NLP.",
            "What is the difference between zero-shot and few-shot prompting?",
        ]
        for idx, prompt_text in enumerate(sample_prompts):
            with sample_cols[idx]:
                if st.button(f"📌 {prompt_text}", key=f"sample_{idx}"):
                    add_chat_message("user", prompt_text)
                    st.rerun()

    # Render conversation history
    for msg in chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat Input Box (Fixed at bottom)
    user_input = st.chat_input("Ask a question or explore an NLP concept...")
    if user_input:
        add_chat_message("user", user_input)
        with st.chat_message("user"):
            st.markdown(user_input)

        client = resolve_client()
        if client:
            with st.chat_message("assistant"):
                with st.spinner("PromptLab AI is thinking..."):
                    reply, err_msg, lat = generate_chat_response(
                        client=client,
                        history=get_chat_history(),
                        model=st.session_state.selected_model,
                        temperature=st.session_state.temperature,
                        system_instruction=active_sys_inst,
                    )
                    if err_msg:
                        st.error(err_msg)
                    elif reply:
                        st.markdown(reply)
                        st.caption(f"⏱️ Response generated in {lat} ms using `{st.session_state.selected_model}`")
                        add_chat_message("assistant", reply)


# ===========================================================================
# PAGE 2: QUESTION ANSWERING (Task 1)
# ===========================================================================
elif selected_page == "❓ Question Answering":
    st.markdown('<div class="main-header">❓ Question Answering</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Structured technical explanations with uncertainty acknowledgment and prompt strategy tuning.</div>',
        unsafe_allow_html=True,
    )

    col_strategy, col_presets = st.columns([1, 1])
    with col_strategy:
        qa_strategy = st.selectbox(
            "Select Prompt Strategy:",
            options=[
                PromptStrategy.ZERO_SHOT,
                PromptStrategy.FEW_SHOT,
                PromptStrategy.ROLE_PROMPT,
                PromptStrategy.CHAIN_OF_THOUGHT,
            ],
            index=2,  # Default to Role Prompting
        )
    with col_presets:
        preset_q = st.selectbox(
            "Load Sample NLP Question:",
            options=[
                "-- Custom Question --",
                "What is Self-Attention and how does it compute query, key, and value vectors?",
                "How does Temperature affect LLM decoding probability distributions?",
                "Explain the difference between Extractive and Abstractive text summarization.",
                "Can an LLM experience hallucinations? If so, why do they occur?",
            ],
        )

    default_text = preset_q if preset_q != "-- Custom Question --" else ""
    qa_input = st.text_area(
        "Enter your question:",
        value=default_text,
        height=110,
        placeholder="e.g. Explain how Backpropagation through time works in RNNs...",
    )

    col_btn1, col_btn2 = st.columns([1, 3])
    with col_btn1:
        submit_qa = st.button("🚀 Generate Answer", type="primary", use_container_width=True)

    if submit_qa:
        is_valid, validation_err = validate_input_text(qa_input, min_length=3)
        if not is_valid:
            st.warning(validation_err)
        else:
            client = resolve_client()
            if client:
                sys_inst, formatted_prompt = build_prompt(
                    task=NLPTask.QUESTION_ANSWERING,
                    strategy=qa_strategy,
                    user_input=qa_input,
                )

                with st.spinner("Generating structured answer..."):
                    reply, err_msg, lat = generate_single_response(
                        client=client,
                        prompt=formatted_prompt,
                        model=st.session_state.selected_model,
                        temperature=st.session_state.temperature,
                        system_instruction=sys_inst,
                    )

                if err_msg:
                    st.error(err_msg)
                elif reply:
                    st.session_state.queries_count = st.session_state.get("queries_count", 0) + 1
                    st.success(f"Generated via **{qa_strategy.value}** ({lat} ms)")
                    st.markdown("### 💡 **Answer**")
                    st.markdown(reply)

                    # Template inspect expander
                    with st.expander("🔍 View Prompt Template Used (Secrets Excluded)"):
                        st.markdown(format_prompt_for_display(sys_inst, formatted_prompt))

                    # Quick RQS evaluation trigger
                    st.divider()
                    st.markdown("#### ⚖️ **Evaluate Quality of this Response**")
                    if st.button("Evaluate with RQS Evaluator", key="eval_qa_btn"):
                        with st.spinner("Evaluating response quality via LLM judge..."):
                            eval_result, eval_err = evaluate_response_with_llm(
                                client=client,
                                model=st.session_state.selected_model,
                                task=NLPTask.QUESTION_ANSWERING.value,
                                strategy=qa_strategy.value,
                                input_text=qa_input,
                                generated_response=reply,
                            )
                        if eval_err:
                            st.error(eval_err)
                        elif eval_result:
                            record_evaluation(eval_result)
                            st.success(f"**RQS Score: {eval_result.rqs_score} / 5.0**")
                            q1, q2, q3, q4 = st.columns(4)
                            q1.metric("Accuracy", f"{eval_result.accuracy}/5")
                            q2.metric("Relevance", f"{eval_result.relevance}/5")
                            q3.metric("Consistency", f"{eval_result.consistency}/5")
                            q4.metric("Fluency", f"{eval_result.fluency}/5")


# ===========================================================================
# PAGE 3: TEXT SUMMARIZATION (Task 2)
# ===========================================================================
elif selected_page == "📝 Text Summarization":
    st.markdown('<div class="main-header">📝 Text Summarization</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Condense dense articles and notes with factual grounding and adjustable length modes.</div>',
        unsafe_allow_html=True,
    )

    col_len, col_strat = st.columns(2)
    with col_len:
        length_mode = st.radio(
            "Summary Target Length:",
            options=["Short", "Medium", "Detailed"],
            index=1,
            horizontal=True,
            help="Short: 1-2 sentences. Medium: 3-5 bullet points. Detailed: Executive multi-section overview.",
        )
    with col_strat:
        sum_strategy = st.selectbox(
            "Prompting Strategy:",
            options=[
                PromptStrategy.ZERO_SHOT,
                PromptStrategy.FEW_SHOT,
                PromptStrategy.ROLE_PROMPT,
                PromptStrategy.CHAIN_OF_THOUGHT,
            ],
            index=2,
        )

    # Sample articles buttons
    st.markdown("##### 📄 Quick Sample Texts:")
    s_col1, s_col2 = st.columns(2)
    sample_article_1 = (
        "The Attention mechanism in deep learning was introduced to address the information bottleneck in Sequence-to-Sequence models. "
        "Traditional RNNs compress all source information into a single fixed-length context vector, which struggles with long sequences. "
        "Bahdanau attention resolved this by computing dynamic alignment scores between decoder hidden states and all encoder outputs. "
        "In 2017, the Transformer architecture entirely replaced recurrent layers with Self-Attention, computing Query, Key, and Value projections. "
        "This allowed massive parallelization during training and enabled foundational LLMs like BERT, GPT, and Gemini."
    )
    sample_article_2 = (
        "Prompt engineering has emerged as a disciplined paradigm for steering foundation models without weight fine-tuning. "
        "Techniques span Zero-Shot (direct task specification), Few-Shot (demonstrating exemplars in context), "
        "Role Prompting (conditioning model tone and expertise via persona), and Chain-of-Thought (soliciting intermediate reasoning steps). "
        "Empirical research demonstrates that well-structured prompt instructions reduce hallucinations and improve domain accuracy significantly."
    )

    if "sum_input_val" not in st.session_state:
        st.session_state.sum_input_val = sample_article_1

    with s_col1:
        if st.button("Load Transformer Architecture Note", use_container_width=True):
            st.session_state.sum_input_val = sample_article_1
            st.rerun()
    with s_col2:
        if st.button("Load Prompt Engineering Note", use_container_width=True):
            st.session_state.sum_input_val = sample_article_2
            st.rerun()

    sum_input = st.text_area(
        "Paste text to summarize:",
        value=st.session_state.sum_input_val,
        height=160,
    )

    chars_count = len(sum_input)
    words_count = len(sum_input.split())
    st.caption(f"📊 Input Statistics: {words_count} words | {chars_count} characters")

    if st.button("✨ Summarize Text", type="primary"):
        is_valid, validation_err = validate_input_text(sum_input, min_length=20, max_length=15000)
        if not is_valid:
            st.warning(validation_err)
        else:
            client = resolve_client()
            if client:
                sys_inst, formatted_prompt = build_prompt(
                    task=NLPTask.SUMMARIZATION,
                    strategy=sum_strategy,
                    user_input=sum_input,
                    extra_params={"length_mode": length_mode},
                )

                with st.spinner("Generating factual summary..."):
                    summary, err_msg, lat = generate_single_response(
                        client=client,
                        prompt=formatted_prompt,
                        model=st.session_state.selected_model,
                        temperature=0.3,  # Lower temperature for summarization precision
                        system_instruction=sys_inst,
                    )

                if err_msg:
                    st.error(err_msg)
                elif summary:
                    st.session_state.queries_count = st.session_state.get("queries_count", 0) + 1
                    sum_words = len(summary.split())
                    compression_ratio = round((1 - (sum_words / max(words_count, 1))) * 100, 1)

                    st.success(f"Summary Generated ({lat} ms) — Compressed by {compression_ratio}%")
                    st.markdown("### 📋 **Generated Summary**")
                    st.markdown(summary)

                    with st.expander("🔍 View Prompt Template Used"):
                        st.markdown(format_prompt_for_display(sys_inst, formatted_prompt))

                    st.divider()
                    if st.button("Evaluate Summary with RQS", key="eval_sum_btn"):
                        with st.spinner("Evaluating summary quality via LLM judge..."):
                            eval_result, eval_err = evaluate_response_with_llm(
                                client=client,
                                model=st.session_state.selected_model,
                                task=NLPTask.SUMMARIZATION.value,
                                strategy=sum_strategy.value,
                                input_text=sum_input,
                                generated_response=summary,
                            )
                        if eval_err:
                            st.error(eval_err)
                        elif eval_result:
                            record_evaluation(eval_result)
                            st.success(f"**RQS Score: {eval_result.rqs_score} / 5.0**")
                            q1, q2, q3, q4 = st.columns(4)
                            q1.metric("Accuracy", f"{eval_result.accuracy}/5")
                            q2.metric("Relevance", f"{eval_result.relevance}/5")
                            q3.metric("Consistency", f"{eval_result.consistency}/5")
                            q4.metric("Fluency", f"{eval_result.fluency}/5")


# ===========================================================================
# PAGE 4: SENTIMENT CLASSIFICATION (Task 3)
# ===========================================================================
elif selected_page == "🎭 Sentiment Classification":
    st.markdown('<div class="main-header">🎭 Sentiment Classification</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Real LLM-driven sentiment categorization into Positive, Negative, or Neutral with linguistic rationale.</div>',
        unsafe_allow_html=True,
    )

    strat_col, _ = st.columns([1, 1])
    with strat_col:
        sent_strategy = st.selectbox(
            "Prompting Strategy:",
            options=[
                PromptStrategy.ZERO_SHOT,
                PromptStrategy.FEW_SHOT,
                PromptStrategy.ROLE_PROMPT,
                PromptStrategy.CHAIN_OF_THOUGHT,
            ],
            index=1,  # Default to Few-Shot for classification stability
        )

    # Required Test Examples
    st.markdown("##### 🧪 Required Practical Test Examples (Click to Load):")
    t1, t2, t3 = st.columns(3)
    if "sent_input_val" not in st.session_state:
        st.session_state.sent_input_val = "This phone has excellent battery life."

    with t1:
        if st.button("📱 'This phone has excellent battery life.'", use_container_width=True):
            st.session_state.sent_input_val = "This phone has excellent battery life."
            st.rerun()
    with t2:
        if st.button("⚠️ 'The product quality is terrible.'", use_container_width=True):
            st.session_state.sent_input_val = "The product quality is terrible."
            st.rerun()
    with t3:
        if st.button("📦 'The package arrived yesterday.'", use_container_width=True):
            st.session_state.sent_input_val = "The package arrived yesterday."
            st.rerun()

    sent_input = st.text_input(
        "Text to Classify:",
        value=st.session_state.sent_input_val,
        placeholder="Enter a review, statement, or tweet...",
    )

    if st.button("🎯 Classify Sentiment", type="primary"):
        is_valid, validation_err = validate_input_text(sent_input, min_length=2)
        if not is_valid:
            st.warning(validation_err)
        else:
            client = resolve_client()
            if client:
                sys_inst, formatted_prompt = build_prompt(
                    task=NLPTask.SENTIMENT_ANALYSIS,
                    strategy=sent_strategy,
                    user_input=sent_input,
                )

                with st.spinner("Analyzing sentiment via Gemini..."):
                    prediction, err_msg, lat = generate_single_response(
                        client=client,
                        prompt=formatted_prompt,
                        model=st.session_state.selected_model,
                        temperature=0.0,  # Zero temperature for deterministic classification
                        system_instruction=sys_inst,
                    )

                if err_msg:
                    st.error(err_msg)
                elif prediction:
                    st.session_state.queries_count = st.session_state.get("queries_count", 0) + 1

                    # Parse label
                    pred_upper = prediction.upper()
                    if "POSITIVE" in pred_upper:
                        badge_html = '<span class="badge-positive">🟢 POSITIVE</span>'
                        norm_label = "Positive"
                    elif "NEGATIVE" in pred_upper:
                        badge_html = '<span class="badge-negative">🔴 NEGATIVE</span>'
                        norm_label = "Negative"
                    else:
                        badge_html = '<span class="badge-neutral">🔵 NEUTRAL</span>'
                        norm_label = "Neutral"

                    st.markdown("### 🏷️ **Classification Result**")
                    c_badge, c_lat = st.columns([1, 2])
                    with c_badge:
                        st.markdown(badge_html, unsafe_allow_html=True)
                    with c_lat:
                        st.caption(f"⚡ Classification latency: {lat} ms | Model: `{st.session_state.selected_model}`")

                    st.markdown('<div class="card-box">', unsafe_allow_html=True)
                    st.markdown("#### 🔬 **Model Analysis & Explanation:**")
                    st.markdown(prediction)
                    st.markdown("</div>", unsafe_allow_html=True)

                    with st.expander("🔍 View Prompt Template Used"):
                        st.markdown(format_prompt_for_display(sys_inst, formatted_prompt))

                    st.divider()
                    if st.button("Evaluate Classification with RQS", key="eval_sent_btn"):
                        with st.spinner("Evaluating classification quality via LLM judge..."):
                            eval_result, eval_err = evaluate_response_with_llm(
                                client=client,
                                model=st.session_state.selected_model,
                                task=NLPTask.SENTIMENT_ANALYSIS.value,
                                strategy=sent_strategy.value,
                                input_text=sent_input,
                                generated_response=prediction,
                            )
                        if eval_err:
                            st.error(eval_err)
                        elif eval_result:
                            record_evaluation(eval_result)
                            st.success(f"**RQS Score: {eval_result.rqs_score} / 5.0**")
                            q1, q2, q3, q4 = st.columns(4)
                            q1.metric("Accuracy", f"{eval_result.accuracy}/5")
                            q2.metric("Relevance", f"{eval_result.relevance}/5")
                            q3.metric("Consistency", f"{eval_result.consistency}/5")
                            q4.metric("Fluency", f"{eval_result.fluency}/5")


# ===========================================================================
# PAGE 5: PROMPT ENGINEERING LAB
# ===========================================================================
elif selected_page == "🔬 Prompt Engineering Lab":
    st.markdown('<div class="main-header">🔬 Prompt Engineering Lab</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Interactive comparison workbench: Evaluate Zero-Shot vs Few-Shot vs Role vs CoT on identical inputs.</div>',
        unsafe_allow_html=True,
    )

    lab_task = st.selectbox(
        "Select NLP Task for Experimentation:",
        options=[
            NLPTask.QUESTION_ANSWERING.value,
            NLPTask.SUMMARIZATION.value,
            NLPTask.SENTIMENT_ANALYSIS.value,
        ],
    )

    default_inputs = {
        NLPTask.QUESTION_ANSWERING.value: "How does the Transformer model overcome the vanishing gradient problem in deep networks?",
        NLPTask.SUMMARIZATION.value: (
            "Modern Large Language Models are pre-trained on massive internet datasets using self-supervised learning objectives, "
            "such as masked token prediction or next token autoregression. Post-pretraining, these models undergo Supervised Fine-Tuning (SFT) "
            "on high-quality question-answering pairs, followed by Reinforcement Learning from Human Feedback (RLHF) or Direct Preference "
            "Optimization (DPO) to align their behavior with human values, safety criteria, and helpfulness."
        ),
        NLPTask.SENTIMENT_ANALYSIS.value: "The software UI is beautifully designed, but the unexpected server crashes make it unusable for production.",
    }

    lab_input = st.text_area(
        "Test Input for Comparative Evaluation:",
        value=default_inputs[lab_task],
        height=110,
    )

    strategies_to_test = st.multiselect(
        "Select Strategies to Compare Concurrently:",
        options=[
            PromptStrategy.ZERO_SHOT.value,
            PromptStrategy.FEW_SHOT.value,
            PromptStrategy.ROLE_PROMPT.value,
            PromptStrategy.CHAIN_OF_THOUGHT.value,
        ],
        default=[
            PromptStrategy.ZERO_SHOT.value,
            PromptStrategy.FEW_SHOT.value,
            PromptStrategy.ROLE_PROMPT.value,
        ],
    )

    if st.button("⚡ Run Comparative Experiment", type="primary"):
        is_valid, validation_err = validate_input_text(lab_input, min_length=3)
        if not is_valid:
            st.warning(validation_err)
        elif not strategies_to_test:
            st.warning("Please select at least one strategy to evaluate.")
        else:
            client = resolve_client()
            if client:
                nlp_task_enum = NLPTask(lab_task)
                results_cols = st.columns(len(strategies_to_test))
                comparison_metrics = []

                progress_bar = st.progress(0, text="Executing comparative prompts across Gemini...")
                for idx, strat_str in enumerate(strategies_to_test):
                    strat_enum = PromptStrategy(strat_str)
                    sys_inst, formatted_prompt = build_prompt(
                        task=nlp_task_enum,
                        strategy=strat_enum,
                        user_input=lab_input,
                    )

                    with results_cols[idx]:
                        st.markdown(f"#### 🎯 **{strat_str}**")
                        with st.spinner(f"Generating ({strat_str})..."):
                            reply, err_msg, lat = generate_single_response(
                                client=client,
                                prompt=formatted_prompt,
                                model=st.session_state.selected_model,
                                temperature=st.session_state.temperature,
                                system_instruction=sys_inst,
                            )

                        if err_msg:
                            st.error(err_msg)
                        elif reply:
                            st.session_state.queries_count = st.session_state.get("queries_count", 0) + 1
                            st.caption(f"⚡ Latency: **{lat} ms**")

                            with st.expander("🔍 View Prompt Template"):
                                st.markdown(format_prompt_for_display(sys_inst, formatted_prompt))

                            st.markdown('<div class="card-box">', unsafe_allow_html=True)
                            st.markdown(reply)
                            st.markdown("</div>", unsafe_allow_html=True)

                            # Evaluate automatically
                            with st.spinner("Scoring quality..."):
                                eval_result, _ = evaluate_response_with_llm(
                                    client=client,
                                    model=st.session_state.selected_model,
                                    task=lab_task,
                                    strategy=strat_str,
                                    input_text=lab_input,
                                    generated_response=reply,
                                )

                            if eval_result:
                                record_evaluation(eval_result)
                                st.metric("RQS Score", f"{eval_result.rqs_score}/5.0")
                                comparison_metrics.append(
                                    {
                                        "Strategy": strat_str,
                                        "Latency (ms)": lat,
                                        "RQS Score": eval_result.rqs_score,
                                        "Accuracy": eval_result.accuracy,
                                        "Relevance": eval_result.relevance,
                                        "Consistency": eval_result.consistency,
                                        "Fluency": eval_result.fluency,
                                    }
                                )

                    progress_bar.progress((idx + 1) / len(strategies_to_test))

                if comparison_metrics:
                    st.divider()
                    st.markdown("### 📊 **Comparative Analysis Summary**")
                    comp_df = pd.DataFrame(comparison_metrics)
                    st.dataframe(comp_df, use_container_width=True)

                    chart_col1, chart_col2 = st.columns(2)
                    with chart_col1:
                        st.markdown("##### 🏆 RQS Score Comparison")
                        st.bar_chart(comp_df.set_index("Strategy")["RQS Score"])
                    with chart_col2:
                        st.markdown("##### ⏱️ Latency Comparison (ms)")
                        st.bar_chart(comp_df.set_index("Strategy")["Latency (ms)"])


# ===========================================================================
# PAGE 6: RESPONSE QUALITY EVALUATION DASHBOARD (RQS)
# ===========================================================================
elif selected_page == "📊 Quality Evaluation (RQS)":
    st.markdown('<div class="main-header">📊 Response Quality Evaluation Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Mathematical evaluation framework measuring Accuracy, Relevance, Consistency, and Fluency.</div>',
        unsafe_allow_html=True,
    )

    # Mathematical Model Display
    st.markdown(
        """
        <div class="rqs-formula-box">
            <b>Mathematical Formulation:</b><br>
            $$\\text{RQS} = \\frac{\\text{Accuracy} + \\text{Relevance} + \\text{Consistency} + \\text{Fluency}}{4}$$
            <small>Where each dimension is scored on an interval scale: $[1.0, 5.0]$, yielding $\\text{RQS} \\in [1.0, 5.0]$.</small>
        </div>
        """,
        unsafe_allow_html=True,
    )

    eval_logs = get_evaluations()

    # Metrics Summary Row
    if eval_logs:
        df_logs = records_to_dataframe(eval_logs)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Evaluations", len(eval_logs))
        m2.metric("Mean RQS", f"{round(df_logs['RQS Score'].mean(), 2)} / 5.0")
        m3.metric("Mean Accuracy", f"{round(df_logs['Accuracy'].mean(), 2)} / 5.0")
        m4.metric("Mean Relevance", f"{round(df_logs['Relevance'].mean(), 2)} / 5.0")
        m5.metric("Mean Consistency", f"{round(df_logs['Consistency'].mean(), 2)} / 5.0")
    else:
        st.info("ℹ️ No evaluations recorded yet. Run queries in Chat, QA, Summarization, or use the interactive evaluator below.")

    st.divider()

    # Interactive Standalone Evaluator
    st.markdown("### 🧪 **Standalone Response Evaluator**")
    st.caption("Evaluate any input-response pair against the 4 RQS dimensions using an LLM-as-a-judge.")

    with st.expander("Click to Open Standalone Evaluator", expanded=not bool(eval_logs)):
        eval_task = st.selectbox("Task Category:", [t.value for t in NLPTask], key="st_task")
        eval_strat = st.selectbox("Strategy Applied:", [s.value for s in PromptStrategy], key="st_strat")
        eval_inp = st.text_area("User Input Text:", key="st_inp", height=70)
        eval_out = st.text_area("AI Generated Response to Evaluate:", key="st_out", height=100)
        eval_ref = st.text_area("Reference / Benchmark Answer (Optional Grounding):", key="st_ref", height=60)

        if st.button("⚖️ Run Evaluator Judgment", type="primary"):
            if not eval_inp.strip() or not eval_out.strip():
                st.warning("Please provide both User Input and Generated Response to evaluate.")
            else:
                client = resolve_client()
                if client:
                    with st.spinner("Scoring response against RQS dimensions..."):
                        eval_res, eval_err = evaluate_response_with_llm(
                            client=client,
                            model=st.session_state.selected_model,
                            task=eval_task,
                            strategy=eval_strat,
                            input_text=eval_inp,
                            generated_response=eval_out,
                            reference_answer=eval_ref,
                        )
                    if eval_err:
                        st.error(eval_err)
                    elif eval_res:
                        record_evaluation(eval_res)
                        st.success(f"**Calculated RQS: {eval_res.rqs_score} / 5.0**")
                        res_c1, res_c2, res_c3, res_c4 = st.columns(4)
                        res_c1.metric("Accuracy", f"{eval_res.accuracy}/5.0")
                        res_c2.metric("Relevance", f"{eval_res.relevance}/5.0")
                        res_c3.metric("Consistency", f"{eval_res.consistency}/5.0")
                        res_c4.metric("Fluency", f"{eval_res.fluency}/5.0")
                        st.rerun()

    # Evaluation Records Table & CSV Download
    if eval_logs:
        st.markdown("### 📋 **Evaluation History Records**")
        df_display = records_to_dataframe(eval_logs)
        st.dataframe(df_display, use_container_width=True)

        col_csv, col_clear = st.columns([1, 1])
        with col_csv:
            csv_data = export_evaluations_to_csv(eval_logs)
            st.download_button(
                label="📥 Download Evaluation Report (CSV)",
                data=csv_data,
                file_name=f"promptlab_rqs_report_{int(time.time())}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col_clear:
            if st.button("🗑️ Clear Evaluation History", use_container_width=True):
                clear_evaluations()
                st.rerun()

    # Academic Disclaimer & Limitations Box
    st.markdown("---")
    st.markdown(
        """
        > **⚠️ Academic Methodology & Limitations Note:**
        > - **LLM-as-a-Judge Nature**: Automated scores are computed using prompt-based LLM judgment and represent *heuristic estimates*, not absolute mathematical ground truth.
        > - **Consistency Measurement**: Scientific consistency cannot be definitively established from a single evaluation sample. True consistency assessment requires multi-sample variance testing across varied sampling temperatures.
        > - **Human-in-the-Loop**: In academic research, LLM evaluations should be verified against human-annotated benchmark datasets.
        """
    )


# ===========================================================================
# PAGE 7: ABOUT PROJECT & VIVA PREPARATION
# ===========================================================================
elif selected_page == "📖 About Project & Viva":
    st.markdown('<div class="main-header">📖 About Project & Viva Guide</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Academic practical report, system architecture, theoretical foundations, and viva preparation.</div>',
        unsafe_allow_html=True,
    )

    tab_overview, tab_theory, tab_arch, tab_viva = st.tabs(
        ["📌 Project Overview", "🧠 NLP Theory & Math", "🏗️ System Architecture", "🎓 Viva Questions & Answers"]
    )

    with tab_overview:
        st.markdown(
            """
            ### 🎯 Project Title:
            **PromptLab AI — Chatbot using LLM with Prompt Engineering**

            #### 1. Aim
            To design, implement, and evaluate a comprehensive Natural Language Processing (NLP) web application that harnesses 
            Large Language Models (LLMs) via prompt engineering methodologies (Zero-Shot, Few-Shot, Role Prompting, Chain-of-Thought) 
            and validates output quality through a structured mathematical framework (**Response Quality Score — RQS**).

            #### 2. Key Objectives
            - **Core NLP Tasks**: Demonstrate practical solutions for Question Answering, Text Summarization, and Sentiment Classification.
            - **Prompt Engineering Paradigms**: Compare Zero-Shot, Few-Shot, and Role Prompting techniques side-by-side.
            - **Real Generative AI Integration**: Connect directly to Google's official Gemini API (`google-genai` SDK) without fake responses or simulations.
            - **RQS Quality Framework**: Formulate and compute $RQS = \\frac{Accuracy + Relevance + Consistency + Fluency}{4}$.
            - **Cloud Deployment**: Package and deploy the application as a publicly accessible service on Streamlit Community Cloud.

            #### 3. Software & Hardware Requirements
            - **Programming Language**: Python 3.10+
            - **Framework**: Streamlit
            - **LLM Engine**: Google Gemini API (`gemini-2.5-flash`, `gemini-1.5-flash`)
            - **SDK**: `google-genai` (Official Google Generative AI Python SDK)
            - **Data Manipulation**: Pandas
            - **Automated Testing**: Pytest
            - **Hardware**: Any modern standard CPU laptop or desktop with internet connectivity (inference handled via Google Cloud).
            """
        )

    with tab_theory:
        st.markdown(
            """
            ### 🧠 Theory of LLMs and Prompt Engineering

            #### 1. What are Large Language Models (LLMs)?
            LLMs are autoregressive deep neural networks based on the **Transformer architecture** (Vaswani et al., 2017). 
            They are trained on hundreds of billions of text tokens to predict the next token given a context sequence:
            $$P(w_1, w_2, \\dots, w_T) = \\prod_{t=1}^T P(w_t \\mid w_1, \\dots, w_{t-1})$$

            #### 2. Prompt Engineering Taxonomy
            Instead of updating model weights through computationally expensive fine-tuning, **Prompt Engineering** guides 
            in-context learning via inference-time instructions:
            - **Zero-Shot Prompting**: Presenting the model with a direct natural language task instruction without demonstration examples.
            - **Few-Shot Prompting**: Providing $k$ exemplar input-output demonstrations in context before presenting the query. This guides formatting and domain boundaries.
            - **Role Prompting**: Explicitly conditioning the prior probabilities of the model by assigning an authoritative persona (e.g. *Senior Professor*, *Chief Linguist*).
            - **Chain-of-Thought (CoT)**: Instructing the model to generate intermediate reasoning rationale steps prior to emitting the terminal response.

            #### 3. Mathematical Model: Response Quality Score (RQS)
            To systematically benchmark the generative output quality, we define:
            $$\\text{RQS} = \\frac{\\text{Accuracy} + \\text{Relevance} + \\text{Consistency} + \\text{Fluency}}{4}$$
            Where each dimension is evaluated on an interval $[1.0, 5.0]$:
            1. **Accuracy (A)**: Truthfulness and factual alignment with domain knowledge.
            2. **Relevance (R)**: Pertinence to user query without conversational drift.
            3. **Consistency (C)**: Internal logical coherence and non-contradiction across claims.
            4. **Fluency (F)**: Grammatical precision, syntax, and readability.
            """
        )

    with tab_arch:
        st.markdown(
            """
            ### 🏗️ System Architecture & Workflow

            ```mermaid
            flowchart TD
                User([👤 User / Student]) -->|Interacts via Browser| UI[🖥️ Streamlit Web Interface]
                
                subgraph AppLogic [PromptLab AI Application Logic]
                    UI --> Nav{Navigation Router}
                    Nav -->|1| ChatModule[AI Chat multi-turn]
                    Nav -->|2| QAModule[Question Answering]
                    Nav -->|3| SumModule[Text Summarization]
                    Nav -->|4| SentModule[Sentiment Classifier]
                    Nav -->|5| LabModule[Prompt Engineering Lab]
                    Nav -->|6| EvalModule[RQS Dashboard]
                    
                    PromptBuilder[utils/prompts.py: Prompt Engine]
                    PromptBuilder -->|Injects Strategy: Zero/Few/Role/CoT| Client[utils/gemini_client.py]
                end

                subgraph CloudAPI [Google AI Cloud Infrastructure]
                    Client -->|HTTPS API Request google-genai| GeminiAPI[Google Gemini Models\ngemini-2.5-flash]
                    GeminiAPI -->|Inference Output| Client
                end

                subgraph EvalEngine [Quality Benchmark Engine]
                    Client -->|Response| Evaluator[utils/evaluation.py]
                    Evaluator -->|LLM Judge & Math Formula| RQS[Compute RQS Score]
                    RQS -->|Persist Records| SessionHistory[(utils/history.py\nSession State)]
                    SessionHistory -->|Export| CSV[Pandas CSV Download]
                end

                SessionHistory -->|Render Metrics & Charts| UI
            ```
            """
        )

    with tab_viva:
        st.markdown(
            """
            ### 🎓 College Viva & Practical Defense Q&A

            **Q1: Why did you choose Prompt Engineering instead of Fine-Tuning?**  
            *Answer:* Fine-tuning requires massive computational resources (GPUs), large labeled domain datasets, and risks catastrophic forgetting. Prompt engineering leverages in-context learning inherent in pretrained foundation models (Gemini), enabling instant adaptation, zero weight modification, and flexible strategy benchmarking.

            **Q2: What is the difference between Zero-Shot and Few-Shot prompting?**  
            *Answer:* In Zero-Shot, the model receives only task instructions. In Few-Shot prompting, we supply 2–3 input-output examples directly in the prompt context. Few-shot drastically improves formatting compliance, stylistic alignment, and structured classification accuracy.

            **Q3: How does the Temperature parameter affect output generation?**  
            *Answer:* Temperature controls the sharpness of the softmax probability distribution over vocabulary tokens during autoregressive generation. At $T=0.0$, the model performs greedy decoding (highly deterministic and reproducible), ideal for classification. At higher temperatures ($T=0.7 - 1.0$), the probability mass is flattened, yielding more diverse, creative answers.

            **Q4: How does your RQS metric work?**  
            *Answer:* RQS stands for Response Quality Score, computed as the arithmetic mean of four normalized dimensions: Accuracy, Relevance, Consistency, and Fluency. We use an LLM-as-a-judge prompting routine to evaluate the generated output on a 1.0–5.0 scale, followed by automated mathematical validation.

            **Q5: How are API keys secured in this project?**  
            *Answer:* API keys are never hardcoded in source code, committed to Git repositories, or exposed in the frontend. Locally, they are managed via `.env` with `python-dotenv`. For cloud deployment, Streamlit Cloud Secrets management is used.
            """
        )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "🔬 **PromptLab AI** — NLP Practical Project | Developed with Python, Streamlit, and Google Gemini API."
)
