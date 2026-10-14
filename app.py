"""PromptLab AI — LLM Chatbot with Prompt Engineering.

A modern, minimal, and elegant AI Chatbot Streamlit Application.
Features a clean New Chat screen with centered greeting and heading,
and exactly three expandable sidebar sections:
1. Prompt Engineering Lab (QA, Summarization, Sentiment, RQS)
2. Settings (API Key, Model, Theme)
3. Help & Viva (Project Info, How to Use, Viva Q&A)
"""

import time
from datetime import datetime
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
# Streamlit Page Setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PromptLab AI — LLM Chatbot with Prompt Engineering",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Session State
init_session_state()

if "active_view" not in st.session_state:
    st.session_state.active_view = "chat"

# ---------------------------------------------------------------------------
# Theme System & CSS (Minimal, Dark Navy with Purple Accents)
# ---------------------------------------------------------------------------
is_dark = st.session_state.get("theme_mode", "dark") == "dark"

if is_dark:
    theme_vars = """
        --bg-app: #080B12;
        --bg-sidebar: #0B0E17;
        --bg-main-panel: #0E1322;
        --bg-card: rgba(18, 24, 41, 0.75);
        --bg-card-hover: rgba(28, 38, 65, 0.9);
        --border-subtle: rgba(255, 255, 255, 0.08);
        --border-purple: rgba(139, 92, 246, 0.35);
        --border-purple-hover: rgba(168, 85, 247, 0.65);
        --text-primary: #F8FAFC;
        --text-secondary: #94A3B8;
        --text-muted: #64748B;
        --accent-purple: #8B5CF6;
        --accent-purple-gradient: linear-gradient(135deg, #8B5CF6 0%, #6D28D9 100%);
        --accent-purple-hover: linear-gradient(135deg, #9333EA 0%, #7C3AED 100%);
        --chat-user-bg: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        --chat-assistant-bg: rgba(16, 22, 38, 0.9);
        --shadow-glow: 0 0 25px rgba(139, 92, 246, 0.25);
    """
else:
    theme_vars = """
        --bg-app: #F4F6FB;
        --bg-sidebar: #FFFFFF;
        --bg-main-panel: #FFFFFF;
        --bg-card: #F8FAFC;
        --bg-card-hover: #F1F5F9;
        --border-subtle: #E2E8F0;
        --border-purple: rgba(124, 58, 237, 0.3);
        --border-purple-hover: rgba(124, 58, 237, 0.6);
        --text-primary: #0F172A;
        --text-secondary: #475569;
        --text-muted: #94A3B8;
        --accent-purple: #7C3AED;
        --accent-purple-gradient: linear-gradient(135deg, #7C3AED 0%, #6D28D9 100%);
        --accent-purple-hover: linear-gradient(135deg, #8B5CF6 0%, #7C3AED 100%);
        --chat-user-bg: linear-gradient(135deg, #6366F1 0%, #7C3AED 100%);
        --chat-assistant-bg: #F8FAFC;
        --shadow-glow: 0 0 20px rgba(124, 58, 237, 0.15);
    """

st.markdown(
    f"""
    <style>
    :root {{
        {theme_vars}
    }}

    html, body, [class*="st-"] {{
        font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, sans-serif;
    }}

    .stApp {{
        background: var(--bg-app);
        color: var(--text-primary);
    }}

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {{
        background-color: var(--bg-sidebar) !important;
        border-right: 1px solid var(--border-subtle);
    }}

    .sidebar-brand {{
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.5rem 0.2rem 1rem 0.2rem;
    }}
    .sidebar-logo-icon {{
        width: 36px;
        height: 36px;
        border-radius: 10px;
        background: var(--accent-purple-gradient);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        box-shadow: 0 0 15px rgba(139, 92, 246, 0.45);
    }}
    .sidebar-title {{
        font-size: 1.15rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: var(--text-primary);
        line-height: 1.1;
    }}
    .sidebar-tagline {{
        font-size: 0.72rem;
        color: var(--text-secondary);
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }}

    /* Expandable Sidebar Sections */
    .st-emotion-cache-p5msec, details {{
        background-color: transparent !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 10px !important;
        margin-bottom: 0.6rem !important;
    }}

    /* Clean Centered New Chat Hero */
    .hero-center-box {{
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        min-height: 52vh;
        text-align: center;
        padding: 2rem 1rem;
    }}
    .hero-greeting {{
        font-size: 2.35rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: var(--text-primary);
        margin-bottom: 0.6rem;
    }}
    .hero-heading {{
        font-size: 1.45rem;
        font-weight: 600;
        color: var(--text-secondary);
        letter-spacing: -0.01em;
    }}

    /* Chat Messages */
    .chat-bubble-user {{
        background: var(--chat-user-bg);
        color: #FFFFFF;
        padding: 0.9rem 1.3rem;
        border-radius: 16px 16px 4px 16px;
        margin: 0.6rem 0;
        max-width: 82%;
        margin-left: auto;
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.25);
        font-size: 0.95rem;
        line-height: 1.55;
    }}
    .chat-bubble-assistant {{
        background: var(--chat-assistant-bg);
        color: var(--text-primary);
        border: 1px solid var(--border-subtle);
        padding: 1.2rem 1.4rem;
        border-radius: 16px 16px 16px 4px;
        margin: 0.6rem 0;
        max-width: 94%;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        font-size: 0.95rem;
        line-height: 1.6;
    }}
    .meta-tag {{
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        font-size: 0.75rem;
        color: var(--text-muted);
        margin-top: 0.4rem;
    }}

    /* Badges */
    .badge-positive {{
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid #10B981;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }}
    .badge-negative {{
        background: rgba(244, 63, 94, 0.15);
        color: #FB7185;
        border: 1px solid #F43F5E;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }}
    .badge-neutral {{
        background: rgba(6, 182, 212, 0.15);
        color: #38BDF8;
        border: 1px solid #06B6D4;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }}

    /* Card Panels */
    .tool-card {{
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 14px;
        padding: 1.4rem;
        margin-bottom: 1.2rem;
    }}

    /* Streamlit Chat Input Purple Button Styling */
    div[data-testid="stChatInput"] button {{
        background: var(--accent-purple-gradient) !important;
        color: white !important;
        border: none !important;
        border-radius: 50% !important;
    }}
    div[data-testid="stChatInput"] button:hover {{
        background: var(--accent-purple-hover) !important;
        box-shadow: 0 0 15px rgba(139, 92, 246, 0.6) !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar: Logo, + New Chat, and EXACTLY THREE Expandable Sections
# ---------------------------------------------------------------------------
with st.sidebar:
    # PromptLab AI Brand Header
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-logo-icon">⚛️</div>
            <div>
                <div class="sidebar-title">PromptLab AI</div>
                <div class="sidebar-tagline">LLM Prompt Studio</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # + New Chat Button
    if st.button("➕  New Chat", key="btn_new_chat", use_container_width=True, type="primary"):
        clear_chat_history()
        st.session_state.active_view = "chat"
        st.rerun()

    st.markdown("<div style='height: 0.4rem;'></div>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # SECTION 1: Prompt Engineering Lab (Expandable)
    # -----------------------------------------------------------------------
    exp_lab = st.expander(
        "🔬 Prompt Engineering Lab",
        expanded=(st.session_state.active_view in ["chat", "summarization", "sentiment", "rqs"]),
    )
    with exp_lab:
        if st.button("❓ Question Answering", use_container_width=True, key="nav_qa"):
            st.session_state.active_view = "chat"
            st.rerun()
        if st.button("📝 Text Summarization", use_container_width=True, key="nav_sum"):
            st.session_state.active_view = "summarization"
            st.rerun()
        if st.button("🎭 Sentiment Classification", use_container_width=True, key="nav_sent"):
            st.session_state.active_view = "sentiment"
            st.rerun()
        if st.button("📊 Response Quality (RQS)", use_container_width=True, key="nav_rqs"):
            st.session_state.active_view = "rqs"
            st.rerun()

    # -----------------------------------------------------------------------
    # SECTION 2: Settings (Expandable)
    # -----------------------------------------------------------------------
    exp_settings = st.expander(
        "⚙️ Settings",
        expanded=(st.session_state.active_view in ["settings_api", "settings_model", "settings_theme"]),
    )
    with exp_settings:
        if st.button("🔑 Gemini API Key", use_container_width=True, key="nav_set_api"):
            st.session_state.active_view = "settings_api"
            st.rerun()
        if st.button("🎛️ Model Selection", use_container_width=True, key="nav_set_model"):
            st.session_state.active_view = "settings_model"
            st.rerun()
        if st.button("🎨 Theme Settings", use_container_width=True, key="nav_set_theme"):
            st.session_state.active_view = "settings_theme"
            st.rerun()

    # -----------------------------------------------------------------------
    # SECTION 3: Help & Viva (Expandable)
    # -----------------------------------------------------------------------
    exp_help = st.expander(
        "📖 Help & Viva",
        expanded=(st.session_state.active_view in ["help_project", "help_how_to", "help_viva"]),
    )
    with exp_help:
        if st.button("📌 Project Information", use_container_width=True, key="nav_help_proj"):
            st.session_state.active_view = "help_project"
            st.rerun()
        if st.button("💡 How to Use Chatbot", use_container_width=True, key="nav_help_how"):
            st.session_state.active_view = "help_how_to"
            st.rerun()
        if st.button("🎓 Viva Questions & Answers", use_container_width=True, key="nav_help_viva"):
            st.session_state.active_view = "help_viva"
            st.rerun()


# ---------------------------------------------------------------------------
# Helper: Client Resolution
# ---------------------------------------------------------------------------
def resolve_client():
    """Retrieve Gemini client using active key with user guidance if missing."""
    key = st.session_state.custom_api_key.strip() or get_api_key()
    if not key:
        st.error(
            "🛑 **Gemini API Key Required**\n\n"
            "Please configure your Google Gemini API key by expanding **⚙️ Settings** $\\to$ **🔑 Gemini API Key** in the left sidebar, "
            "or adding `GEMINI_API_KEY` to `.env`."
        )
        return None
    return get_gemini_client(key)


# ===========================================================================
# VIEW 1: 💬 MAIN CHAT / QUESTION ANSWERING (Clean New Chat Home Screen)
# ===========================================================================
if st.session_state.active_view == "chat":
    chat_history = get_chat_history()

    # 1. Clean Initial Screen (ONLY Greeting, Heading, and Message Composer)
    if not chat_history:
        # User greeting resolution
        user_name = st.session_state.get("user_name")
        greeting_text = f"नमस्ते, {user_name} 👋" if user_name else "नमस्ते 👋"

        st.markdown(
            f"""
            <div class="hero-center-box">
                <div class="hero-greeting">{greeting_text}</div>
                <div class="hero-heading">How can I help you today?</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 2. Active Chat Conversation
    else:
        # Small header options bar
        top_c1, top_c2 = st.columns([4, 1])
        with top_c1:
            st.caption(f"Model: `{st.session_state.selected_model}` | Temperature: {st.session_state.temperature}")
        with top_c2:
            if st.button("🧹 Clear", use_container_width=True):
                clear_chat_history()
                st.rerun()

        for msg in chat_history:
            role = msg["role"]
            content = msg["content"]
            timestamp = msg.get("timestamp", "")
            latency = msg.get("latency")

            if role == "user":
                st.markdown(
                    f"""
                    <div class="chat-bubble-user">
                        <div style="font-weight: 700; font-size: 0.78rem; opacity: 0.85; margin-bottom: 0.2rem;">YOU</div>
                        {content}
                        <div class="meta-tag" style="color: rgba(255,255,255,0.7); justify-content: flex-end;">
                            <span>{timestamp}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                lat_str = f"⏱️ {latency} ms" if latency else ""
                st.markdown(
                    f"""
                    <div class="chat-bubble-assistant">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                            <span style="font-weight: 700; font-size: 0.82rem; color: #A78BFA;">PROMPTLAB AI</span>
                            <span style="font-size: 0.75rem; color: var(--text-muted); font-family: monospace;">{lat_str}</span>
                        </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown(content)
                st.markdown(
                    f"""
                        <div class="meta-tag">
                            <span>Model: {st.session_state.selected_model}</span>
                            <span>•</span>
                            <span>{timestamp}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # Clean Bottom Message Composer
    user_query = st.chat_input("Message PromptLab AI...")
    if user_query:
        add_chat_message("user", user_query)
        st.markdown(
            f"""
            <div class="chat-bubble-user">
                <div style="font-weight: 700; font-size: 0.78rem; opacity: 0.85; margin-bottom: 0.2rem;">YOU</div>
                {user_query}
            </div>
            """,
            unsafe_allow_html=True,
        )

        client = resolve_client()
        if client:
            sys_inst, formatted_prompt = build_prompt(
                task=NLPTask.QUESTION_ANSWERING,
                strategy=PromptStrategy.ZERO_SHOT,
                user_input=user_query,
            )

            with st.spinner("PromptLab AI is thinking..."):
                reply, err_msg, lat = generate_chat_response(
                    client=client,
                    history=get_chat_history(),
                    model=st.session_state.selected_model,
                    temperature=st.session_state.temperature,
                    system_instruction=sys_inst,
                )

            if err_msg:
                st.error(err_msg)
            elif reply:
                add_chat_message("assistant", reply, latency=lat, model=st.session_state.selected_model)
                st.rerun()


# ===========================================================================
# VIEW 2: 📝 TEXT SUMMARIZATION (Prompt Engineering Lab)
# ===========================================================================
elif st.session_state.active_view == "summarization":
    st.markdown("## 📝 **Text Summarization**")
    st.caption("Condense dense articles and notes with factual grounding and adjustable length modes.")

    col_len, col_strat = st.columns(2)
    with col_len:
        len_mode = st.radio("Target Summary Length:", ["Short", "Medium", "Detailed"], index=1, horizontal=True)
    with col_strat:
        sum_strat = st.selectbox("Prompt Strategy:", [s.value for s in PromptStrategy], index=0)

    sample_article = (
        "The Attention mechanism in deep learning was introduced to address the information bottleneck in Sequence-to-Sequence models. "
        "Traditional RNNs compress all source information into a single fixed-length context vector, which struggles with long sequences. "
        "Bahdanau attention resolved this by computing dynamic alignment scores between decoder hidden states and all encoder outputs. "
        "In 2017, the Transformer architecture entirely replaced recurrent layers with Self-Attention, computing Query, Key, and Value projections. "
        "This allowed massive parallelization during training and enabled foundational LLMs like BERT, GPT, and Gemini."
    )

    sum_input = st.text_area("Paste Article or Text to Summarize:", value=sample_article, height=150)
    words_in = len(sum_input.split())
    st.caption(f"Input: {words_in} words | {len(sum_input)} characters")

    if st.button("✨ Summarize Text", type="primary"):
        is_valid, validation_err = validate_input_text(sum_input, min_length=20, max_length=15000)
        if not is_valid:
            st.warning(validation_err)
        else:
            client = resolve_client()
            if client:
                sys_inst, formatted_prompt = build_prompt(
                    task=NLPTask.SUMMARIZATION,
                    strategy=PromptStrategy(sum_strat),
                    user_input=sum_input,
                    extra_params={"length_mode": len_mode},
                )

                with st.spinner("Generating factual summary..."):
                    summary, err_msg, lat = generate_single_response(
                        client=client,
                        prompt=formatted_prompt,
                        model=st.session_state.selected_model,
                        temperature=0.3,
                        system_instruction=sys_inst,
                    )

                if err_msg:
                    st.error(err_msg)
                elif summary:
                    st.session_state.queries_count = st.session_state.get("queries_count", 0) + 1
                    words_out = len(summary.split())
                    comp_ratio = round((1 - (words_out / max(words_in, 1))) * 100, 1)

                    st.markdown(
                        f"""
                        <div class="tool-card">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem;">
                                <span style="font-weight: 700; color: #A78BFA;">Summary ({len_mode})</span>
                                <span style="font-size: 0.78rem; color: var(--text-muted);">Compressed by {comp_ratio}% • ⏱️ {lat} ms</span>
                            </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.markdown(summary)
                    st.markdown("</div>", unsafe_allow_html=True)

                    with st.expander("🔍 View Prompt Template Used"):
                        st.markdown(format_prompt_for_display(sys_inst, formatted_prompt))

                    st.divider()
                    if st.button("Evaluate Summary with RQS", key="eval_sum_btn"):
                        with st.spinner("Scoring quality via LLM judge..."):
                            eval_res, eval_err = evaluate_response_with_llm(
                                client=client,
                                model=st.session_state.selected_model,
                                task=NLPTask.SUMMARIZATION.value,
                                strategy=sum_strat,
                                input_text=sum_input,
                                generated_response=summary,
                            )
                        if eval_res:
                            record_evaluation(eval_res)
                            st.success(f"**RQS Score: {eval_res.rqs_score} / 5.0**")
                            q1, q2, q3, q4 = st.columns(4)
                            q1.metric("Accuracy", f"{eval_res.accuracy}/5.0")
                            q2.metric("Relevance", f"{eval_res.relevance}/5.0")
                            q3.metric("Consistency", f"{eval_res.consistency}/5.0")
                            q4.metric("Fluency", f"{eval_res.fluency}/5.0")


# ===========================================================================
# VIEW 3: 🎭 SENTIMENT CLASSIFICATION (Prompt Engineering Lab)
# ===========================================================================
elif st.session_state.active_view == "sentiment":
    st.markdown("## 🎭 **Sentiment Classification**")
    st.caption("Classify text into Positive, Negative, or Neutral with linguistic rationale.")

    strat_col, _ = st.columns([1, 1])
    with strat_col:
        sent_strat = st.selectbox("Prompt Strategy:", [s.value for s in PromptStrategy], index=1)

    st.markdown("##### 🧪 **Standard Practical Test Benchmarks:**")
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

    sent_input = st.text_input("Text to Classify:", value=st.session_state.sent_input_val)

    if st.button("🎯 Classify Sentiment", type="primary"):
        is_valid, validation_err = validate_input_text(sent_input, min_length=2)
        if not is_valid:
            st.warning(validation_err)
        else:
            client = resolve_client()
            if client:
                sys_inst, formatted_prompt = build_prompt(
                    task=NLPTask.SENTIMENT_ANALYSIS,
                    strategy=PromptStrategy(sent_strat),
                    user_input=sent_input,
                )

                with st.spinner("Analyzing polarity..."):
                    prediction, err_msg, lat = generate_single_response(
                        client=client,
                        prompt=formatted_prompt,
                        model=st.session_state.selected_model,
                        temperature=0.0,
                        system_instruction=sys_inst,
                    )

                if err_msg:
                    st.error(err_msg)
                elif prediction:
                    st.session_state.queries_count = st.session_state.get("queries_count", 0) + 1

                    pred_upper = prediction.upper()
                    if "POSITIVE" in pred_upper:
                        badge_html = '<span class="badge-positive">🟢 POSITIVE</span>'
                    elif "NEGATIVE" in pred_upper:
                        badge_html = '<span class="badge-negative">🔴 NEGATIVE</span>'
                    else:
                        badge_html = '<span class="badge-neutral">🔵 NEUTRAL</span>'

                    st.markdown("### 🏷️ **Result**")
                    st.markdown(f"{badge_html} &nbsp; <span style='font-size: 0.8rem; color: var(--text-muted);'>⏱️ {lat} ms</span>", unsafe_allow_html=True)
                    st.markdown("#### **Linguistic Explanation:**")
                    st.markdown(prediction)

                    with st.expander("🔍 View Prompt Template Used"):
                        st.markdown(format_prompt_for_display(sys_inst, formatted_prompt))

                    st.divider()
                    if st.button("Evaluate Classification with RQS", key="eval_sent_btn"):
                        with st.spinner("Scoring quality via LLM judge..."):
                            eval_res, eval_err = evaluate_response_with_llm(
                                client=client,
                                model=st.session_state.selected_model,
                                task=NLPTask.SENTIMENT_ANALYSIS.value,
                                strategy=sent_strat,
                                input_text=sent_input,
                                generated_response=prediction,
                            )
                        if eval_res:
                            record_evaluation(eval_res)
                            st.success(f"**RQS Score: {eval_res.rqs_score} / 5.0**")
                            q1, q2, q3, q4 = st.columns(4)
                            q1.metric("Accuracy", f"{eval_res.accuracy}/5.0")
                            q2.metric("Relevance", f"{eval_res.relevance}/5.0")
                            q3.metric("Consistency", f"{eval_res.consistency}/5.0")
                            q4.metric("Fluency", f"{eval_res.fluency}/5.0")


# ===========================================================================
# VIEW 4: 📊 RESPONSE QUALITY (RQS) (Prompt Engineering Lab)
# ===========================================================================
elif st.session_state.active_view == "rqs":
    st.markdown("## 📊 **Response Quality Evaluation (RQS)**")
    st.markdown(
        """
        <div style="background: rgba(139, 92, 246, 0.1); border: 1px solid var(--border-purple); border-radius: 12px; padding: 1.2rem; margin: 1rem 0;">
            <div style="font-weight: 700; color: #A78BFA; font-size: 0.85rem; text-transform: uppercase;">Mathematical Formulation</div>
            <div style="font-size: 1.2rem; font-weight: 700; margin: 0.4rem 0;">
                $$\\text{RQS} = \\frac{\\text{Accuracy} + \\text{Relevance} + \\text{Consistency} + \\text{Fluency}}{4}$$
            </div>
            <div style="font-size: 0.82rem; color: var(--text-secondary);">
                Normalized interval scale: $\\text{RQS} \\in [1.0, 5.0]$.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    eval_logs = get_evaluations()

    if eval_logs:
        df_logs = records_to_dataframe(eval_logs)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Evaluations", len(eval_logs))
        m2.metric("Mean RQS", f"{round(df_logs['RQS Score'].mean(), 2)} / 5.0")
        m3.metric("Accuracy", f"{round(df_logs['Accuracy'].mean(), 2)} / 5.0")
        m4.metric("Relevance", f"{round(df_logs['Relevance'].mean(), 2)} / 5.0")
        m5.metric("Consistency", f"{round(df_logs['Consistency'].mean(), 2)} / 5.0")
    else:
        st.info("ℹ️ No evaluations recorded yet. Run queries in Chat, QA, Summarization, or use the standalone evaluator below.")

    st.divider()

    st.markdown("### 🧪 **Standalone Response Evaluator**")
    with st.expander("Open Standalone Evaluator", expanded=not bool(eval_logs)):
        st_task = st.selectbox("Task Category:", [t.value for t in NLPTask], key="st_eval_task")
        st_strat = st.selectbox("Strategy Applied:", [s.value for s in PromptStrategy], key="st_eval_strat")
        st_inp = st.text_area("User Input Text:", key="st_eval_inp", height=70)
        st_out = st.text_area("AI Response to Evaluate:", key="st_eval_out", height=90)

        if st.button("⚖️ Run Evaluator Judgment", type="primary"):
            if not st_inp.strip() or not st_out.strip():
                st.warning("Please provide both User Input and Generated Response.")
            else:
                client = resolve_client()
                if client:
                    with st.spinner("Scoring response quality..."):
                        eval_res, eval_err = evaluate_response_with_llm(
                            client=client,
                            model=st.session_state.selected_model,
                            task=st_task,
                            strategy=st_strat,
                            input_text=st_inp,
                            generated_response=st_out,
                        )
                    if eval_err:
                        st.error(eval_err)
                    elif eval_res:
                        record_evaluation(eval_res)
                        st.success(f"**RQS Score: {eval_res.rqs_score} / 5.0**")
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Accuracy", f"{eval_res.accuracy}/5.0")
                        c2.metric("Relevance", f"{eval_res.relevance}/5.0")
                        c3.metric("Consistency", f"{eval_res.consistency}/5.0")
                        c4.metric("Fluency", f"{eval_res.fluency}/5.0")
                        st.rerun()

    if eval_logs:
        st.markdown("### 📋 **Evaluation History Records**")
        df_display = records_to_dataframe(eval_logs)
        st.dataframe(df_display, use_container_width=True)

        col_csv, col_clear = st.columns([1, 1])
        with col_csv:
            csv_str = export_evaluations_to_csv(eval_logs)
            st.download_button(
                label="📥 Download Evaluation Report (CSV)",
                data=csv_str,
                file_name=f"promptlab_rqs_report_{int(time.time())}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col_clear:
            if st.button("🗑️ Clear Evaluation History", use_container_width=True):
                clear_evaluations()
                st.rerun()


# ===========================================================================
# VIEW 5: ⚙️ SETTINGS — API KEY
# ===========================================================================
elif st.session_state.active_view == "settings_api":
    st.markdown("## 🔑 **Gemini API Key Configuration**")
    st.caption("Securely configure your Google Gemini API credentials. Stored in session memory and never exposed.")

    custom_key_val = st.text_input(
        "Gemini API Key:",
        type="password",
        value=st.session_state.custom_api_key,
        help="Masked input stored in session memory.",
    )
    if custom_key_val != st.session_state.custom_api_key:
        st.session_state.custom_api_key = custom_key_val
        st.session_state.connection_status = None
        st.rerun()

    configured_k = get_api_key()
    if configured_k:
        masked = f"...{configured_k[-4:]}" if len(configured_k) >= 4 else "configured"
        st.success(f"Active Key Configured: `{masked}`")

    st.markdown("### 🔌 **Connectivity Diagnostic**")
    if st.button("Test Gemini Connection", use_container_width=True, type="primary"):
        with st.spinner("Pinging Google AI servers..."):
            active_k = custom_key_val.strip() or configured_k
            client = get_gemini_client(active_k)
            is_ok, msg, lat = verify_connectivity(client, st.session_state.selected_model)
            st.session_state.connection_status = {"is_ok": is_ok, "msg": msg, "lat": lat}

    if st.session_state.connection_status:
        stat = st.session_state.connection_status
        if stat["is_ok"]:
            st.success(f"🟢 **Connected** ({stat['lat']} ms)\n\n{stat['msg']}")
        else:
            st.error(f"🔴 **Connection Failed**\n\n{stat['msg']}")


# ===========================================================================
# VIEW 6: ⚙️ SETTINGS — MODEL SELECTION
# ===========================================================================
elif st.session_state.active_view == "settings_model":
    st.markdown("## 🎛️ **Model Selection & Decoding Parameters**")

    new_model = st.selectbox(
        "Gemini Model ID:",
        options=SUPPORTED_MODELS,
        index=SUPPORTED_MODELS.index(st.session_state.selected_model) if st.session_state.selected_model in SUPPORTED_MODELS else 0,
    )
    if new_model != st.session_state.selected_model:
        st.session_state.selected_model = new_model
        st.rerun()

    st.caption("Recommended: `gemini-3.5-flash-lite` for low latency and high availability.")

    new_temp = st.slider("Sampling Temperature:", 0.0, 1.0, float(st.session_state.temperature), 0.05)
    if new_temp != st.session_state.temperature:
        st.session_state.temperature = new_temp

    st.caption("Tip: Use T=0.0 for deterministic classification and T=0.7 for conversational Q&A.")


# ===========================================================================
# VIEW 7: ⚙️ SETTINGS — THEME SETTINGS
# ===========================================================================
elif st.session_state.active_view == "settings_theme":
    st.markdown("## 🎨 **Theme Settings**")

    current_theme = st.session_state.get("theme_mode", "dark")
    chosen_theme = st.radio(
        "Select Theme:",
        options=["Dark Navy & Charcoal", "Clean Light"],
        index=0 if current_theme == "dark" else 1,
    )

    new_mode = "dark" if chosen_theme == "Dark Navy & Charcoal" else "light"
    if new_mode != current_theme:
        st.session_state.theme_mode = new_mode
        st.rerun()


# ===========================================================================
# VIEW 8: 📖 HELP & VIVA — PROJECT INFORMATION
# ===========================================================================
elif st.session_state.active_view == "help_project":
    st.markdown("## 📌 **Project Information**")
    st.markdown(
        """
        ### PromptLab AI — Chatbot using LLM with Prompt Engineering
        - **Course:** NLP Practical Project
        - **Platform:** Streamlit Web Application
        - **LLM Engine:** Google Gemini API (`google-genai` SDK)
        - **Core Capabilities:**
          1. Multi-turn Question Answering with uncertainty acknowledgment
          2. Factual Text Summarization (Short, Medium, Detailed)
          3. Sentiment Classification (Positive, Negative, Neutral)
          4. Response Quality Score (RQS) mathematical evaluation
        """
    )


# ===========================================================================
# VIEW 9: 📖 HELP & VIVA — HOW TO USE CHATBOT
# ===========================================================================
elif st.session_state.active_view == "help_how_to":
    st.markdown("## 💡 **How to Use PromptLab AI**")
    st.markdown(
        """
        1. **Start a Conversation**:
           - Click **➕ New Chat** in the left sidebar anytime to start fresh.
           - Type your question in the bottom message composer and press Enter.
        2. **Use Specialized Tools**:
           - Expand **🔬 Prompt Engineering Lab** in the sidebar.
           - Open **Text Summarization** to condense articles with compression statistics.
           - Open **Sentiment Classification** to test reviews against the 3 required benchmarks.
           - Open **Response Quality (RQS)** to evaluate generated outputs and download CSV reports.
        3. **Configure API Key & Model**:
           - Expand **⚙️ Settings** $\\to$ **🔑 Gemini API Key** to test connection latency.
        """
    )


# ===========================================================================
# VIEW 10: 📖 HELP & VIVA — VIVA QUESTIONS & ANSWERS
# ===========================================================================
elif st.session_state.active_view == "help_viva":
    st.markdown("## 🎓 **Viva Defense Questions & Answers**")
    st.markdown(
        """
        **Q1: Why choose Prompt Engineering instead of Fine-Tuning?**  
        *Answer:* Fine-tuning modifies neural network weights, requiring high computational resources (GPUs) and large domain-specific datasets. Prompt engineering steers foundation models via in-context learning with zero weight modification and zero training latency.

        **Q2: What is the mathematical definition of RQS?**  
        *Answer:* RQS (Response Quality Score) is the arithmetic mean of four normalized dimensions:  
        $$\\text{RQS} = \\frac{\\text{Accuracy} + \\text{Relevance} + \\text{Consistency} + \\text{Fluency}}{4}$$  
        where each dimension is scored on an interval scale of $1.0$ to $5.0$.

        **Q3: How does Few-Shot prompting improve stability over Zero-Shot?**  
        *Answer:* Supplying 2-3 exemplar input-output pairs directly inside the prompt context activates pattern recognition in the self-attention mechanism, standardizing format compliance and classification accuracy.

        **Q4: How does Temperature affect LLM decoding?**  
        *Answer:* Temperature scales the logits before the softmax operation ($P(w_i) \\propto \\exp(z_i / T)$). At $T=0.0$, decoding is greedy and deterministic. At higher temperatures ($T \\ge 0.7$), token sampling diversity increases.
        """
    )
