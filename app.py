"""PromptLab AI — LLM Chatbot with Prompt Engineering.

A modern, visually impressive Generative AI web application built for college NLP practical
demonstration. Integrates the Google Gemini API (google-genai SDK), prompt engineering strategies,
and the Response Quality Score (RQS) evaluation framework.
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
# Streamlit Page Setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PromptLab AI — LLM Chatbot with Prompt Engineering",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Session State
init_session_state()

# ---------------------------------------------------------------------------
# Theme System & Dynamic CSS (Dark & Light)
# ---------------------------------------------------------------------------
is_dark = st.session_state.get("theme_mode", "dark") == "dark"

if is_dark:
    theme_vars = """
        --bg-canvas: #090D16;
        --bg-canvas-subtle: #0F172A;
        --bg-card: rgba(19, 28, 46, 0.75);
        --bg-card-hover: rgba(28, 41, 65, 0.85);
        --bg-elevated: #1E293B;
        --border-subtle: rgba(255, 255, 255, 0.08);
        --border-hover: rgba(139, 92, 246, 0.4);
        --border-glow: rgba(6, 182, 212, 0.35);
        --text-primary: #F8FAFC;
        --text-secondary: #94A3B8;
        --text-muted: #64748B;
        --accent-violet: #8B5CF6;
        --accent-violet-glow: rgba(139, 92, 246, 0.25);
        --accent-cyan: #06B6D4;
        --accent-emerald: #10B981;
        --accent-rose: #F43F5E;
        --accent-amber: #F59E0B;
        --chat-user-bg: linear-gradient(135deg, #4338CA 0%, #6366F1 100%);
        --chat-assistant-bg: rgba(19, 28, 46, 0.9);
        --code-bg: #0F172A;
        --shadow-card: 0 4px 20px -2px rgba(0, 0, 0, 0.35);
        --shadow-glow: 0 0 25px -5px rgba(139, 92, 246, 0.2);
    """
else:
    theme_vars = """
        --bg-canvas: #F8FAFC;
        --bg-canvas-subtle: #F1F5F9;
        --bg-card: #FFFFFF;
        --bg-card-hover: #F8FAFC;
        --bg-elevated: #F1F5F9;
        --border-subtle: #E2E8F0;
        --border-hover: rgba(124, 58, 237, 0.4);
        --border-glow: rgba(8, 145, 178, 0.35);
        --text-primary: #0F172A;
        --text-secondary: #475569;
        --text-muted: #94A3B8;
        --accent-violet: #7C3AED;
        --accent-violet-glow: rgba(124, 58, 237, 0.15);
        --accent-cyan: #0891B2;
        --accent-emerald: #059669;
        --accent-rose: #E11D48;
        --accent-amber: #D97706;
        --chat-user-bg: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%);
        --chat-assistant-bg: #FFFFFF;
        --code-bg: #F1F5F9;
        --shadow-card: 0 4px 15px -2px rgba(0, 0, 0, 0.08);
        --shadow-glow: 0 0 20px -5px rgba(124, 58, 237, 0.15);
    """

st.markdown(
    f"""
    <style>
    :root {{
        {theme_vars}
    }}

    /* Global reset and font styling */
    html, body, [class*="st-"] {{
        font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, sans-serif;
    }}

    .stApp {{
        background: var(--bg-canvas);
        color: var(--text-primary);
    }}

    /* Top Brand Hero Banner */
    .brand-hero {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.8rem 1.4rem;
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 14px;
        margin-bottom: 1.5rem;
        box-shadow: var(--shadow-card);
    }}
    .brand-hero-title {{
        font-size: 1.4rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, var(--accent-violet) 0%, var(--accent-cyan) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }}
    .brand-hero-sub {{
        font-size: 0.85rem;
        color: var(--text-secondary);
        margin: 0;
    }}

    /* Modern Glassmorphic Cards */
    .promptlab-card {{
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 14px;
        padding: 1.4rem;
        margin-bottom: 1.2rem;
        box-shadow: var(--shadow-card);
        transition: all 0.25s ease;
    }}
    .promptlab-card:hover {{
        border-color: var(--border-hover);
        box-shadow: var(--shadow-glow);
    }}

    /* Interactive Starter Prompt Cards */
    .starter-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }}
    .starter-card {{
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 1.2rem;
        cursor: pointer;
        transition: all 0.2s ease;
        text-align: left;
    }}
    .starter-card:hover {{
        transform: translateY(-2px);
        border-color: var(--accent-violet);
        background: var(--bg-card-hover);
        box-shadow: var(--shadow-glow);
    }}
    .starter-icon {{
        font-size: 1.6rem;
        margin-bottom: 0.6rem;
        display: inline-block;
    }}
    .starter-title {{
        font-size: 0.95rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.3rem;
    }}
    .starter-desc {{
        font-size: 0.82rem;
        color: var(--text-secondary);
        line-height: 1.4;
    }}

    /* Message Bubbles */
    .chat-bubble-user {{
        background: var(--chat-user-bg);
        color: #FFFFFF;
        padding: 0.9rem 1.2rem;
        border-radius: 16px 16px 4px 16px;
        margin: 0.6rem 0;
        max-width: 85%;
        margin-left: auto;
        box-shadow: 0 4px 15px -3px rgba(99, 102, 241, 0.35);
        font-size: 0.95rem;
        line-height: 1.5;
    }}
    .chat-bubble-assistant {{
        background: var(--chat-assistant-bg);
        color: var(--text-primary);
        border: 1px solid var(--border-subtle);
        padding: 1.2rem 1.4rem;
        border-radius: 16px 16px 16px 4px;
        margin: 0.6rem 0;
        max-width: 95%;
        box-shadow: var(--shadow-card);
        font-size: 0.95rem;
        line-height: 1.6;
    }}
    .chat-meta {{
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-top: 0.6rem;
        font-size: 0.75rem;
        color: var(--text-muted);
    }}

    /* Badges & Status Pills */
    .badge-pill {{
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }}
    .badge-online {{
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }}
    .badge-offline {{
        background: rgba(244, 63, 94, 0.15);
        color: #FB7185;
        border: 1px solid rgba(244, 63, 94, 0.4);
    }}
    .badge-model {{
        background: rgba(139, 92, 246, 0.15);
        color: #A78BFA;
        border: 1px solid rgba(139, 92, 246, 0.35);
    }}
    .badge-positive {{
        background: rgba(16, 185, 129, 0.18);
        color: #34D399;
        border: 1px solid #10B981;
    }}
    .badge-negative {{
        background: rgba(244, 63, 94, 0.18);
        color: #FB7185;
        border: 1px solid #F43F5E;
    }}
    .badge-neutral {{
        background: rgba(6, 182, 212, 0.18);
        color: #38BDF8;
        border: 1px solid #06B6D4;
    }}

    /* Metric Value Box */
    .metric-box {{
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 1.1rem 1rem;
        text-align: center;
        transition: transform 0.2s ease;
    }}
    .metric-box:hover {{
        transform: translateY(-2px);
        border-color: var(--accent-violet);
    }}
    .metric-box-val {{
        font-size: 1.9rem;
        font-weight: 800;
        background: linear-gradient(135deg, var(--accent-violet) 0%, var(--accent-cyan) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
    }}
    .metric-box-label {{
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--text-secondary);
        margin-top: 0.35rem;
    }}

    /* Formula Banner */
    .formula-banner {{
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.12) 0%, rgba(6, 182, 212, 0.08) 100%);
        border: 1px solid var(--accent-violet);
        border-radius: 12px;
        padding: 1rem 1.4rem;
        margin: 1.2rem 0;
    }}

    /* Sidebar Logo & Branding */
    .sidebar-brand {{
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.5rem 0.2rem 1rem 0.2rem;
    }}
    .sidebar-logo-icon {{
        width: 38px;
        height: 38px;
        border-radius: 10px;
        background: linear-gradient(135deg, var(--accent-violet) 0%, var(--accent-cyan) 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        box-shadow: var(--shadow-glow);
    }}
    .sidebar-title {{
        font-size: 1.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: var(--text-primary);
        line-height: 1.1;
    }}
    .sidebar-tagline {{
        font-size: 0.72rem;
        font-weight: 500;
        color: var(--text-secondary);
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }}

    /* Code Container */
    pre, code {{
        border-radius: 8px !important;
    }}

    /* Responsive adjustments */
    @media (max-width: 768px) {{
        .brand-hero {{
            flex-direction: column;
            align-items: flex-start;
            gap: 0.6rem;
        }}
        .chat-bubble-user, .chat-bubble-assistant {{
            max-width: 100%;
        }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar: Branding, Navigation, Mode Toggle & API Controls
# ---------------------------------------------------------------------------
with st.sidebar:
    # 1. Branding Header
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-logo-icon">🧪</div>
            <div>
                <div class="sidebar-title">PromptLab AI</div>
                <div class="sidebar-tagline">NLP Practical • LLM Studio</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. "New Chat" Quick Action Button
    col_new_chat, col_theme = st.columns([2, 1])
    with col_new_chat:
        if st.button("✨ New Chat", use_container_width=True, type="primary"):
            clear_chat_history()
            st.rerun()

    with col_theme:
        current_theme = st.session_state.get("theme_mode", "dark")
        new_theme_label = "☀️ Light" if current_theme == "dark" else "🌙 Dark"
        if st.button(new_theme_label, use_container_width=True, help="Toggle Dark/Light theme"):
            st.session_state.theme_mode = "light" if current_theme == "dark" else "dark"
            st.rerun()

    st.divider()

    # 3. Main Navigation
    st.markdown("##### 🧭 **WORKSPACE NAVIGATION**")
    nav_selection = st.radio(
        "Navigation:",
        options=[
            "💬 AI Assistant",
            "🔬 Prompt Engineering Lab",
            "❓ Question Answering",
            "📝 Text Summarization",
            "🎭 Sentiment Classification",
            "📊 Quality Evaluation (RQS)",
            "⚙️ Settings & System Health",
            "📖 About & Viva Defense",
        ],
        index=0,
        label_visibility="collapsed",
    )

    st.divider()

    # 4. Live API Connectivity Badge
    configured_key = get_api_key()
    active_key = st.session_state.custom_api_key.strip() or configured_key

    if active_key:
        masked_key = f"...{active_key[-4:]}" if len(active_key) >= 4 else "configured"
        st.markdown(
            f"""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <span style="font-size: 0.8rem; font-weight: 600; color: var(--text-secondary);">API STATUS</span>
                <span class="badge-pill badge-online">⚡ Connected</span>
            </div>
            <div style="font-size: 0.76rem; color: var(--text-muted); font-family: monospace;">Key: {masked_key}</div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <span style="font-size: 0.8rem; font-weight: 600; color: var(--text-secondary);">API STATUS</span>
                <span class="badge-pill badge-offline">⚠️ Missing Key</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 5. Session Metrics Mini-Card
    stats = get_session_stats()
    st.markdown(
        f"""
        <div style="background: var(--bg-card); border: 1px solid var(--border-subtle); border-radius: 10px; padding: 0.8rem; margin: 0.8rem 0;">
            <div style="display: flex; justify-content: space-between; font-size: 0.78rem; margin-bottom: 0.3rem;">
                <span style="color: var(--text-secondary);">Queries Executed:</span>
                <span style="font-weight: 700; color: var(--text-primary);">{stats['queries_executed']}</span>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.78rem; margin-bottom: 0.3rem;">
                <span style="color: var(--text-secondary);">Active Model:</span>
                <span style="font-weight: 600; color: var(--accent-cyan);">{st.session_state.selected_model}</span>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.78rem;">
                <span style="color: var(--text-secondary);">Mean RQS:</span>
                <span style="font-weight: 700; color: #10B981;">{stats['average_rqs']}/5.0</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("🧹 Reset Session", use_container_width=True, help="Clear message history and evaluation records"):
        clear_chat_history()
        clear_evaluations()
        st.session_state.queries_count = 0
        st.session_state.connection_status = None
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
            "Please configure your Google Gemini API key via:\n"
            "1. **Settings & System Health** tab in the sidebar\n"
            "2. Local `.env` file (`GEMINI_API_KEY=AIzaSy...`)\n"
            "3. Streamlit Cloud Secrets for public deployment."
        )
        return None
    return get_gemini_client(key)


# ===========================================================================
# PAGE 1: 💬 AI ASSISTANT (Modern Conversational Workspace)
# ===========================================================================
if nav_selection == "💬 AI Assistant":
    # Top Hero Card
    st.markdown(
        """
        <div class="brand-hero">
            <div>
                <h1 class="brand-hero-title">💬 AI Conversational Assistant</h1>
                <p class="brand-hero-sub">Multi-turn dialogue engine with role conditioning and prompt strategy steering.</p>
            </div>
            <div style="display: flex; gap: 0.5rem; align-items: center;">
                <span class="badge-pill badge-model">Active: {}</span>
            </div>
        </div>
        """.format(st.session_state.selected_model),
        unsafe_allow_html=True,
    )

    # Persona & Controls Toolbar
    tool_col1, tool_col2 = st.columns([3, 1])
    with tool_col1:
        persona = st.selectbox(
            "Select Assistant Persona / System Role:",
            options=[
                "🎓 Senior NLP & AI Professor (Detailed, pedagogical, intuitive analogies)",
                "💻 Principal Software Architect (Technical precision, algorithmic complexity)",
                "⚡ Concise & Direct Helper (High-density, direct facts, zero fluff)",
                "🧭 Socratic Inquirer (Guiding questions and hint-driven reasoning)",
            ],
            index=0,
            help="Condition the foundation model with a specialized domain persona.",
        )
    with tool_col2:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            clear_chat_history()
            st.rerun()

    persona_prompts = {
        "🎓 Senior NLP & AI Professor (Detailed, pedagogical, intuitive analogies)": SYSTEM_INSTRUCTIONS[NLPTask.QUESTION_ANSWERING],
        "💻 Principal Software Architect (Technical precision, algorithmic complexity)": (
            "You are a Principal Software Architect. Give rigorous, production-grade technical advice, "
            "highlighting scalability, edge cases, and algorithmic complexity."
        ),
        "⚡ Concise & Direct Helper (High-density, direct facts, zero fluff)": (
            "You are a direct, concise academic assistant. Give high-density, accurate answers without fluff."
        ),
        "🧭 Socratic Inquirer (Guiding questions and hint-driven reasoning)": (
            "You are a Socratic tutor. Help the user discover solutions by providing concise hints, "
            "clarifying questions, and encouraging logical reasoning."
        ),
    }
    active_sys_inst = persona_prompts[persona]

    # Empty State: Interactive Starter Prompt Cards
    chat_history = get_chat_history()
    if not chat_history:
        st.markdown("### 💡 **Explore Foundational NLP Concepts**")
        st.caption("Click any starter topic below to immediately begin exploring:")

        starter_col1, starter_col2 = st.columns(2)
        with starter_col1:
            if st.button(
                "🧠 Transformer Self-Attention\n\nExplain how Query, Key, and Value projections compute scaled dot-product attention.",
                key="starter_1",
                use_container_width=True,
            ):
                add_chat_message("user", "Explain how Query, Key, and Value projections compute scaled dot-product attention.")
                st.rerun()

            if st.button(
                "🔬 Zero-Shot vs Few-Shot\n\nWhat are the theoretical and empirical differences between Zero-Shot and Few-Shot prompting?",
                key="starter_2",
                use_container_width=True,
            ):
                add_chat_message("user", "What are the theoretical and empirical differences between Zero-Shot and Few-Shot prompting?")
                st.rerun()

        with starter_col2:
            if st.button(
                "⚖️ Hallucination Mitigation\n\nWhy do foundation models hallucinate and what prompt strategies reduce ungrounded answers?",
                key="starter_3",
                use_container_width=True,
            ):
                add_chat_message("user", "Why do foundation models hallucinate and what prompt strategies reduce ungrounded answers?")
                st.rerun()

            if st.button(
                "🌡️ Temperature & Top-P Decoding\n\nHow do temperature and nucleus sampling alter the softmax probability distribution?",
                key="starter_4",
                use_container_width=True,
            ):
                add_chat_message("user", "How do temperature and nucleus sampling alter the softmax probability distribution during autoregressive generation?")
                st.rerun()

    # Active Conversation History Rendering
    for msg in chat_history:
        role = msg["role"]
        content = msg["content"]
        timestamp = msg.get("timestamp", "")
        latency = msg.get("latency")

        if role == "user":
            st.markdown(
                f"""
                <div class="chat-bubble-user">
                    <div style="font-weight: 600; font-size: 0.8rem; margin-bottom: 0.25rem; opacity: 0.85;">YOU</div>
                    {content}
                    <div class="chat-meta" style="color: rgba(255,255,255,0.7); justify-content: flex-end;">
                        <span>{timestamp}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            lat_badge = f'<span class="badge-pill badge-model">⏱️ {latency} ms</span>' if latency else ""
            st.markdown(
                f"""
                <div class="chat-bubble-assistant">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <span style="font-weight: 700; font-size: 0.82rem; color: var(--accent-violet);">PROMPTLAB AI</span>
                        <div>{lat_badge}</div>
                    </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(content)
            st.markdown(
                f"""
                    <div class="chat-meta">
                        <span>Model: {st.session_state.selected_model}</span>
                        <span>•</span>
                        <span>{timestamp}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Large Fixed Message Composer at bottom
    user_query = st.chat_input("Ask a question, enter an NLP concept, or explore a prompt...")
    if user_query:
        add_chat_message("user", user_query)
        st.markdown(
            f"""
            <div class="chat-bubble-user">
                <div style="font-weight: 600; font-size: 0.8rem; margin-bottom: 0.25rem; opacity: 0.85;">YOU</div>
                {user_query}
            </div>
            """,
            unsafe_allow_html=True,
        )

        client = resolve_client()
        if client:
            with st.spinner("PromptLab AI is formulating response via Gemini..."):
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
                add_chat_message("assistant", reply, latency=lat, model=st.session_state.selected_model)
                st.rerun()


# ===========================================================================
# PAGE 2: 🔬 PROMPT ENGINEERING LAB (Interactive Comparison Workbench)
# ===========================================================================
elif nav_selection == "🔬 Prompt Engineering Lab":
    st.markdown(
        """
        <div class="brand-hero">
            <div>
                <h1 class="brand-hero-title">🔬 Prompt Engineering Laboratory</h1>
                <p class="brand-hero-sub">Experimentation workbench: Compare Zero-Shot, Few-Shot, Role Prompting, and CoT side-by-side.</p>
            </div>
            <span class="badge-pill badge-model">Methodology Benchmarking</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    task_col, mode_col = st.columns([1, 1])
    with task_col:
        lab_task_str = st.selectbox(
            "Select NLP Task for Experimentation:",
            options=[t.value for t in NLPTask],
            index=0,
        )

    task_enum = NLPTask(lab_task_str)

    default_test_inputs = {
        NLPTask.QUESTION_ANSWERING.value: "How does the self-attention mechanism differ from traditional recurrent hidden state propagation?",
        NLPTask.SUMMARIZATION.value: (
            "The Transformer model relies entirely on self-attention mechanisms to compute representations of its input and output without using sequence-aligned RNNs or convolution. "
            "In sequence transduction models, recurrent neural networks generate a sequence of hidden states as a function of the previous hidden state and the current input token. "
            "This inherently sequential nature precludes parallelization within training examples, which becomes critical at longer sequence lengths, as memory constraints limit batching. "
            "Attention mechanisms have become an integral part of compelling sequence modeling, allowing modeling of dependencies without regard to their distance in the input or output sequences."
        ),
        NLPTask.SENTIMENT_ANALYSIS.value: "The software UI is beautifully designed, but the unexpected server crashes make it unusable for production.",
    }

    lab_input = st.text_area(
        "Enter Input for Comparative Testing:",
        value=default_test_inputs[lab_task_str],
        height=120,
    )

    # Strategy Selection Cards
    st.markdown("##### 🎯 **Select Strategies to Execute Concurrently:**")
    strat_col1, strat_col2, strat_col3, strat_col4 = st.columns(4)

    with strat_col1:
        run_zs = st.checkbox("Zero-Shot", value=True, help="Direct task instructions without exemplars.")
    with strat_col2:
        run_fs = st.checkbox("Few-Shot", value=True, help="Context injected with 2-3 structured demonstrations.")
    with strat_col3:
        run_rp = st.checkbox("Role Prompting", value=True, help="Conditioned with an authoritative persona.")
    with strat_col4:
        run_cot = st.checkbox("Chain-of-Thought", value=False, help="Explicit step-by-step reasoning anchors.")

    selected_strats = []
    if run_zs:
        selected_strats.append(PromptStrategy.ZERO_SHOT)
    if run_fs:
        selected_strats.append(PromptStrategy.FEW_SHOT)
    if run_rp:
        selected_strats.append(PromptStrategy.ROLE_PROMPT)
    if run_cot:
        selected_strats.append(PromptStrategy.CHAIN_OF_THOUGHT)

    if st.button("⚡ Run Multi-Strategy Comparison", type="primary", use_container_width=True):
        is_valid, validation_err = validate_input_text(lab_input, min_length=3)
        if not is_valid:
            st.warning(validation_err)
        elif not selected_strats:
            st.warning("Please select at least one prompting strategy to evaluate.")
        else:
            client = resolve_client()
            if client:
                res_cols = st.columns(len(selected_strats))
                metrics_data = []

                progress_bar = st.progress(0, text="Executing comparative prompts across Gemini...")
                for idx, strat in enumerate(selected_strats):
                    sys_inst, user_prompt = build_prompt(
                        task=task_enum,
                        strategy=strat,
                        user_input=lab_input,
                    )

                    with res_cols[idx]:
                        st.markdown(
                            f"""
                            <div style="border-bottom: 2px solid var(--accent-violet); padding-bottom: 0.4rem; margin-bottom: 0.8rem;">
                                <span style="font-weight: 700; font-size: 0.95rem; color: var(--text-primary);">{strat.value}</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        with st.spinner(f"Running {strat.value}..."):
                            reply, err_msg, lat = generate_single_response(
                                client=client,
                                prompt=user_prompt,
                                model=st.session_state.selected_model,
                                temperature=st.session_state.temperature,
                                system_instruction=sys_inst,
                            )

                        if err_msg:
                            st.error(err_msg)
                        elif reply:
                            st.session_state.queries_count = st.session_state.get("queries_count", 0) + 1
                            st.markdown(
                                f"""
                                <div style="display: flex; gap: 0.4rem; margin-bottom: 0.6rem;">
                                    <span class="badge-pill badge-model">⏱️ {lat} ms</span>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                            st.markdown(
                                f"""
                                <div class="promptlab-card" style="font-size: 0.9rem; line-height: 1.55;">
                                    {reply}
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                            with st.expander("🔍 View Prompt Template Used"):
                                st.markdown(format_prompt_for_display(sys_inst, user_prompt))

                            # Automated RQS scoring
                            with st.spinner("Scoring RQS via LLM judge..."):
                                eval_res, _ = evaluate_response_with_llm(
                                    client=client,
                                    model=st.session_state.selected_model,
                                    task=task_enum.value,
                                    strategy=strat.value,
                                    input_text=lab_input,
                                    generated_response=reply,
                                )

                            if eval_res:
                                record_evaluation(eval_res)
                                st.markdown(
                                    f"""
                                    <div class="metric-box" style="margin-top: 0.5rem;">
                                        <div class="metric-box-val">{eval_res.rqs_score}</div>
                                        <div class="metric-box-label">RQS Score out of 5.0</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                                metrics_data.append(
                                    {
                                        "Strategy": strat.value,
                                        "Latency (ms)": lat,
                                        "RQS Score": eval_res.rqs_score,
                                        "Accuracy": eval_res.accuracy,
                                        "Relevance": eval_res.relevance,
                                        "Consistency": eval_res.consistency,
                                        "Fluency": eval_res.fluency,
                                    }
                                )

                    progress_bar.progress((idx + 1) / len(selected_strats))

                if metrics_data:
                    st.divider()
                    st.markdown("### 📊 **Empirical Comparison Summary**")
                    comp_df = pd.DataFrame(metrics_data)
                    st.dataframe(comp_df, use_container_width=True)

                    c_col1, c_col2 = st.columns(2)
                    with c_col1:
                        st.markdown("##### 🏆 Response Quality Score (RQS)")
                        st.bar_chart(comp_df.set_index("Strategy")["RQS Score"])
                    with c_col2:
                        st.markdown("##### ⏱️ Execution Latency (ms)")
                        st.bar_chart(comp_df.set_index("Strategy")["Latency (ms)"])


# ===========================================================================
# PAGE 3: ❓ QUESTION ANSWERING (Task 1)
# ===========================================================================
elif nav_selection == "❓ Question Answering":
    st.markdown(
        """
        <div class="brand-hero">
            <div>
                <h1 class="brand-hero-title">❓ Question Answering Engine</h1>
                <p class="brand-hero-sub">Pedagogical explanations with hallucination avoidance and uncertainty grounding.</p>
            </div>
            <span class="badge-pill badge-model">NLP Task 1</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    q_col1, q_col2 = st.columns([1, 1])
    with q_col1:
        qa_strat = st.selectbox(
            "Select Prompt Strategy:",
            options=[s.value for s in PromptStrategy],
            index=2,  # Role Prompting default
        )
    with q_col2:
        preset_q = st.selectbox(
            "Load Benchmark NLP Question:",
            options=[
                "-- Custom Question --",
                "How does the Scaled Dot-Product Attention compute similarity between queries and keys?",
                "What is the mathematical definition of Tokenization and why is it essential?",
                "Explain the difference between Extractive and Abstractive text summarization.",
                "How does temperature scaling modify softmax output probabilities?",
            ],
        )

    initial_q = preset_q if preset_q != "-- Custom Question --" else ""
    user_q = st.text_area(
        "Enter Technical or Educational Question:",
        value=initial_q,
        height=100,
        placeholder="e.g. How does Word2Vec continuous bag of words work?",
    )

    if st.button("🚀 Generate Grounded Answer", type="primary"):
        is_valid, validation_err = validate_input_text(user_q, min_length=3)
        if not is_valid:
            st.warning(validation_err)
        else:
            client = resolve_client()
            if client:
                sys_inst, user_prompt = build_prompt(
                    task=NLPTask.QUESTION_ANSWERING,
                    strategy=PromptStrategy(qa_strat),
                    user_input=user_q,
                )

                with st.spinner("PromptLab AI is synthesizing answer..."):
                    reply, err_msg, lat = generate_single_response(
                        client=client,
                        prompt=user_prompt,
                        model=st.session_state.selected_model,
                        temperature=st.session_state.temperature,
                        system_instruction=sys_inst,
                    )

                if err_msg:
                    st.error(err_msg)
                elif reply:
                    st.session_state.queries_count = st.session_state.get("queries_count", 0) + 1
                    st.markdown(
                        f"""
                        <div class="promptlab-card">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem;">
                                <span style="font-weight: 700; color: var(--accent-violet);">Synthesized Response</span>
                                <span class="badge-pill badge-model">⏱️ {lat} ms</span>
                            </div>
                            <div style="font-size: 0.95rem; line-height: 1.6;">
                        """,
                        unsafe_allow_html=True,
                    )
                    st.markdown(reply)
                    st.markdown("</div></div>", unsafe_allow_html=True)

                    with st.expander("🔍 View Prompt Template Used"):
                        st.markdown(format_prompt_for_display(sys_inst, user_prompt))

                    st.divider()
                    st.markdown("#### ⚖️ **Evaluate Quality of this Response**")
                    if st.button("Evaluate Response with RQS Evaluator", key="eval_qa_single"):
                        with st.spinner("Scoring quality via LLM judge..."):
                            eval_res, eval_err = evaluate_response_with_llm(
                                client=client,
                                model=st.session_state.selected_model,
                                task=NLPTask.QUESTION_ANSWERING.value,
                                strategy=qa_strat,
                                input_text=user_q,
                                generated_response=reply,
                            )
                        if eval_err:
                            st.error(eval_err)
                        elif eval_res:
                            record_evaluation(eval_res)
                            st.success(f"**Calculated RQS Score: {eval_res.rqs_score} / 5.0**")
                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("Accuracy", f"{eval_res.accuracy}/5.0")
                            c2.metric("Relevance", f"{eval_res.relevance}/5.0")
                            c3.metric("Consistency", f"{eval_res.consistency}/5.0")
                            c4.metric("Fluency", f"{eval_res.fluency}/5.0")


# ===========================================================================
# PAGE 4: 📝 TEXT SUMMARIZATION (Task 2)
# ===========================================================================
elif nav_selection == "📝 Text Summarization":
    st.markdown(
        """
        <div class="brand-hero">
            <div>
                <h1 class="brand-hero-title">📝 Text Summarization Engine</h1>
                <p class="brand-hero-sub">Factual condensation with configurable length modes and zero external hallucination.</p>
            </div>
            <span class="badge-pill badge-model">NLP Task 2</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_len, col_strat = st.columns(2)
    with col_len:
        len_mode = st.radio(
            "Summary Target Length:",
            options=["Short", "Medium", "Detailed"],
            index=1,
            horizontal=True,
            help="Short: 1-2 sentences. Medium: 3-5 bullet points. Detailed: Executive synthesis.",
        )
    with col_strat:
        sum_strat = st.selectbox(
            "Prompting Strategy:",
            options=[s.value for s in PromptStrategy],
            index=2,
        )

    sample_1 = (
        "The Attention mechanism in deep learning was introduced to address the information bottleneck in Sequence-to-Sequence models. "
        "Traditional RNNs compress all source information into a single fixed-length context vector, which struggles with long sequences. "
        "Bahdanau attention resolved this by computing dynamic alignment scores between decoder hidden states and all encoder outputs. "
        "In 2017, the Transformer architecture entirely replaced recurrent layers with Self-Attention, computing Query, Key, and Value projections. "
        "This allowed massive parallelization during training and enabled foundational LLMs like BERT, GPT, and Gemini."
    )
    sample_2 = (
        "Prompt engineering has emerged as a disciplined paradigm for steering foundation models without weight fine-tuning. "
        "Techniques span Zero-Shot (direct task specification), Few-Shot (demonstrating exemplars in context), "
        "Role Prompting (conditioning model tone and expertise via persona), and Chain-of-Thought (soliciting intermediate reasoning steps). "
        "Empirical research demonstrates that well-structured prompt instructions reduce hallucinations and improve domain accuracy significantly."
    )

    if "sum_input_val" not in st.session_state:
        st.session_state.sum_input_val = sample_1

    s_col1, s_col2 = st.columns(2)
    with s_col1:
        if st.button("📄 Load Transformer Architecture Note", use_container_width=True):
            st.session_state.sum_input_val = sample_1
            st.rerun()
    with s_col2:
        if st.button("📄 Load Prompt Engineering Note", use_container_width=True):
            st.session_state.sum_input_val = sample_2
            st.rerun()

    sum_input = st.text_area(
        "Paste Article or Notes to Summarize:",
        value=st.session_state.sum_input_val,
        height=150,
    )

    words_in = len(sum_input.split())
    st.caption(f"📊 Input: {words_in} words | {len(sum_input)} characters")

    if st.button("✨ Generate Summary", type="primary"):
        is_valid, validation_err = validate_input_text(sum_input, min_length=20, max_length=15000)
        if not is_valid:
            st.warning(validation_err)
        else:
            client = resolve_client()
            if client:
                sys_inst, user_prompt = build_prompt(
                    task=NLPTask.SUMMARIZATION,
                    strategy=PromptStrategy(sum_strat),
                    user_input=sum_input,
                    extra_params={"length_mode": len_mode},
                )

                with st.spinner("Synthesizing factual summary..."):
                    summary, err_msg, lat = generate_single_response(
                        client=client,
                        prompt=user_prompt,
                        model=st.session_state.selected_model,
                        temperature=0.3,  # Low temperature for factual fidelity
                        system_instruction=sys_inst,
                    )

                if err_msg:
                    st.error(err_msg)
                elif summary:
                    st.session_state.queries_count = st.session_state.get("queries_count", 0) + 1
                    words_out = len(summary.split())
                    compression = round((1 - (words_out / max(words_in, 1))) * 100, 1)

                    st.markdown(
                        f"""
                        <div class="promptlab-card">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem;">
                                <span style="font-weight: 700; color: var(--accent-violet);">Executive Summary ({len_mode})</span>
                                <div style="display: flex; gap: 0.5rem;">
                                    <span class="badge-pill badge-positive">Compressed by {compression}%</span>
                                    <span class="badge-pill badge-model">⏱️ {lat} ms</span>
                                </div>
                            </div>
                            <div style="font-size: 0.95rem; line-height: 1.6;">
                        """,
                        unsafe_allow_html=True,
                    )
                    st.markdown(summary)
                    st.markdown("</div></div>", unsafe_allow_html=True)

                    with st.expander("🔍 View Prompt Template Used"):
                        st.markdown(format_prompt_for_display(sys_inst, user_prompt))

                    st.divider()
                    if st.button("Evaluate Summary with RQS", key="eval_sum_single"):
                        with st.spinner("Scoring summary quality via LLM judge..."):
                            eval_res, eval_err = evaluate_response_with_llm(
                                client=client,
                                model=st.session_state.selected_model,
                                task=NLPTask.SUMMARIZATION.value,
                                strategy=sum_strat,
                                input_text=sum_input,
                                generated_response=summary,
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


# ===========================================================================
# PAGE 5: 🎭 SENTIMENT CLASSIFICATION (Task 3)
# ===========================================================================
elif nav_selection == "🎭 Sentiment Classification":
    st.markdown(
        """
        <div class="brand-hero">
            <div>
                <h1 class="brand-hero-title">🎭 Sentiment Classification Engine</h1>
                <p class="brand-hero-sub">Deterministic LLM polarity categorization into Positive, Negative, or Neutral with linguistic rationale.</p>
            </div>
            <span class="badge-pill badge-model">NLP Task 3</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    strat_col, _ = st.columns([1, 1])
    with strat_col:
        sent_strat = st.selectbox(
            "Prompting Strategy:",
            options=[s.value for s in PromptStrategy],
            index=1,  # Few-shot default for stability
        )

    # Required College Benchmarks
    st.markdown("##### 🧪 **Required College Practical Benchmark Cases (Click to Test):**")
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

    if st.button("🎯 Classify Polarity", type="primary"):
        is_valid, validation_err = validate_input_text(sent_input, min_length=2)
        if not is_valid:
            st.warning(validation_err)
        else:
            client = resolve_client()
            if client:
                sys_inst, user_prompt = build_prompt(
                    task=NLPTask.SENTIMENT_ANALYSIS,
                    strategy=PromptStrategy(sent_strat),
                    user_input=sent_input,
                )

                with st.spinner("Analyzing semantic valence via Gemini..."):
                    prediction, err_msg, lat = generate_single_response(
                        client=client,
                        prompt=user_prompt,
                        model=st.session_state.selected_model,
                        temperature=0.0,  # Zero temperature for deterministic classification
                        system_instruction=sys_inst,
                    )

                if err_msg:
                    st.error(err_msg)
                elif prediction:
                    st.session_state.queries_count = st.session_state.get("queries_count", 0) + 1

                    pred_upper = prediction.upper()
                    if "POSITIVE" in pred_upper:
                        badge_html = '<span class="badge-pill badge-positive" style="font-size: 1rem; padding: 0.4rem 1.2rem;">🟢 POSITIVE</span>'
                    elif "NEGATIVE" in pred_upper:
                        badge_html = '<span class="badge-pill badge-negative" style="font-size: 1rem; padding: 0.4rem 1.2rem;">🔴 NEGATIVE</span>'
                    else:
                        badge_html = '<span class="badge-pill badge-neutral" style="font-size: 1rem; padding: 0.4rem 1.2rem;">🔵 NEUTRAL</span>'

                    st.markdown(
                        f"""
                        <div class="promptlab-card" style="margin-top: 1rem;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                                <div>{badge_html}</div>
                                <span class="badge-pill badge-model">⏱️ {lat} ms</span>
                            </div>
                            <div style="font-size: 0.95rem; line-height: 1.6;">
                        """,
                        unsafe_allow_html=True,
                    )
                    st.markdown(prediction)
                    st.markdown("</div></div>", unsafe_allow_html=True)

                    with st.expander("🔍 View Prompt Template Used"):
                        st.markdown(format_prompt_for_display(sys_inst, user_prompt))

                    st.divider()
                    if st.button("Evaluate Classification with RQS", key="eval_sent_single"):
                        with st.spinner("Scoring classification quality via LLM judge..."):
                            eval_res, eval_err = evaluate_response_with_llm(
                                client=client,
                                model=st.session_state.selected_model,
                                task=NLPTask.SENTIMENT_ANALYSIS.value,
                                strategy=sent_strat,
                                input_text=sent_input,
                                generated_response=prediction,
                            )
                        if eval_err:
                            st.error(eval_err)
                        elif eval_res:
                            record_evaluation(eval_res)
                            st.success(f"**Calculated RQS Score: {eval_res.rqs_score} / 5.0**")
                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("Accuracy", f"{eval_res.accuracy}/5.0")
                            c2.metric("Relevance", f"{eval_res.relevance}/5.0")
                            c3.metric("Consistency", f"{eval_res.consistency}/5.0")
                            c4.metric("Fluency", f"{eval_res.fluency}/5.0")


# ===========================================================================
# PAGE 6: 📊 QUALITY EVALUATION (RQS Dashboard & CSV Export)
# ===========================================================================
elif nav_selection == "📊 Quality Evaluation (RQS)":
    st.markdown(
        """
        <div class="brand-hero">
            <div>
                <h1 class="brand-hero-title">📊 Response Quality Evaluation (RQS)</h1>
                <p class="brand-hero-sub">Mathematical framework quantifying Accuracy, Relevance, Consistency, and Fluency.</p>
            </div>
            <span class="badge-pill badge-model">Mathematical Framework</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Mathematical Formula Banner
    st.markdown(
        """
        <div class="formula-banner">
            <div style="font-size: 0.85rem; font-weight: 700; color: var(--accent-violet); text-transform: uppercase; margin-bottom: 0.4rem;">
                MATHEMATICAL FORMULATION
            </div>
            <div style="font-size: 1.15rem; font-weight: 600;">
                $$\\text{RQS} = \\frac{\\text{Accuracy} + \\text{Relevance} + \\text{Consistency} + \\text{Fluency}}{4}$$
            </div>
            <div style="font-size: 0.82rem; color: var(--text-secondary); margin-top: 0.4rem;">
                Where each dimension is evaluated on an interval $[1.0, 5.0]$, yielding $\\text{RQS} \\in [1.0, 5.0]$.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    eval_logs = get_evaluations()

    # Aggregate Metrics Cards
    if eval_logs:
        df_logs = records_to_dataframe(eval_logs)
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.markdown(
                f'<div class="metric-box"><div class="metric-box-val">{len(eval_logs)}</div><div class="metric-box-label">Evaluations</div></div>',
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f'<div class="metric-box"><div class="metric-box-val">{round(df_logs["RQS Score"].mean(), 2)}</div><div class="metric-box-label">Mean RQS</div></div>',
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f'<div class="metric-box"><div class="metric-box-val">{round(df_logs["Accuracy"].mean(), 2)}</div><div class="metric-box-label">Accuracy</div></div>',
                unsafe_allow_html=True,
            )
        with m4:
            st.markdown(
                f'<div class="metric-box"><div class="metric-box-val">{round(df_logs["Relevance"].mean(), 2)}</div><div class="metric-box-label">Relevance</div></div>',
                unsafe_allow_html=True,
            )
        with m5:
            st.markdown(
                f'<div class="metric-box"><div class="metric-box-val">{round(df_logs["Consistency"].mean(), 2)}</div><div class="metric-box-label">Consistency</div></div>',
                unsafe_allow_html=True,
            )
    else:
        st.info("ℹ️ No evaluations recorded yet. Run queries in Chat, QA, Summarization, or use the standalone evaluator below.")

    st.divider()

    # Standalone Response Evaluator Tool
    st.markdown("### 🧪 **Standalone Response Evaluator**")
    st.caption("Benchmark any arbitrary Input-Response pair against the 4 RQS dimensions using an LLM-as-a-judge.")

    with st.expander("Open Standalone Evaluator", expanded=not bool(eval_logs)):
        st_task = st.selectbox("Task Category:", [t.value for t in NLPTask], key="standalone_task")
        st_strat = st.selectbox("Strategy Applied:", [s.value for s in PromptStrategy], key="standalone_strat")
        st_inp = st.text_area("User Input Text:", key="standalone_inp", height=70)
        st_out = st.text_area("AI Response to Evaluate:", key="standalone_out", height=100)
        st_ref = st.text_area("Ground Truth / Benchmark Reference (Optional):", key="standalone_ref", height=60)

        if st.button("⚖️ Run LLM Judge Evaluation", type="primary"):
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
                            reference_answer=st_ref,
                        )
                    if eval_err:
                        st.error(eval_err)
                    elif eval_res:
                        record_evaluation(eval_res)
                        st.success(f"**Calculated RQS Score: {eval_res.rqs_score} / 5.0**")
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Accuracy", f"{eval_res.accuracy}/5.0")
                        c2.metric("Relevance", f"{eval_res.relevance}/5.0")
                        c3.metric("Consistency", f"{eval_res.consistency}/5.0")
                        c4.metric("Fluency", f"{eval_res.fluency}/5.0")
                        st.rerun()

    # Evaluation Records Table & CSV Download
    if eval_logs:
        st.markdown("### 📋 **Evaluation Records History**")
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

    st.markdown("---")
    st.markdown(
        """
        > **⚠️ Academic Evaluation Methodology Note:**
        > - **LLM-as-a-Judge**: Evaluator scores are heuristic estimates computed by an LLM and should not be treated as absolute ground truth.
        > - **Consistency Testing**: Rigorous consistency assessment requires multi-sample variance testing across diverse temperatures.
        """
    )


# ===========================================================================
# PAGE 7: ⚙️ SETTINGS & SYSTEM HEALTH (Configuration & Ping)
# ===========================================================================
elif nav_selection == "⚙️ Settings & System Health":
    st.markdown(
        """
        <div class="brand-hero">
            <div>
                <h1 class="brand-hero-title">⚙️ Settings & System Health</h1>
                <p class="brand-hero-sub">Gemini API credentials, model parameters, and real-time connectivity testing.</p>
            </div>
            <span class="badge-pill badge-model">System Control</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    settings_card1, settings_card2 = st.columns(2)

    with settings_card1:
        st.markdown("### 🔑 **API Key Configuration**")
        st.caption("Stored securely in session memory. Never exposed in HTML or committed to Git.")

        custom_key_val = st.text_input(
            "Gemini API Key:",
            type="password",
            value=st.session_state.custom_api_key,
            help="Override GEMINI_API_KEY from .env or secrets.",
        )
        if custom_key_val != st.session_state.custom_api_key:
            st.session_state.custom_api_key = custom_key_val
            st.session_state.connection_status = None
            st.rerun()

        configured_k = get_api_key()
        if configured_k:
            masked = f"...{configured_k[-4:]}" if len(configured_k) >= 4 else "configured"
            st.success(f"Active Key Configured: `{masked}`")
        else:
            st.warning("No API Key detected. Please enter key above.")

        st.markdown("### 🔌 **Connectivity Diagnostic**")
        if st.button("Test Gemini Connection", use_container_width=True, type="primary"):
            with st.spinner("Pinging Google AI servers..."):
                active_k = custom_key_val.strip() or configured_k
                client = get_gemini_client(active_k)
                is_ok, msg, lat = verify_connectivity(client, st.session_state.selected_model)
                st.session_state.connection_status = {
                    "is_ok": is_ok,
                    "msg": msg,
                    "lat": lat,
                }

        if st.session_state.connection_status:
            stat = st.session_state.connection_status
            if stat["is_ok"]:
                st.markdown(
                    f"""
                    <div style="background: rgba(16, 185, 129, 0.15); border: 1px solid #10B981; border-radius: 10px; padding: 1rem; margin-top: 0.8rem;">
                        <div style="font-weight: 700; color: #34D399;">🟢 Connection Verified Successfully</div>
                        <div style="font-size: 0.85rem; color: var(--text-primary); margin-top: 0.3rem;">{stat['msg']}</div>
                        <div style="font-size: 0.78rem; color: var(--text-muted); margin-top: 0.3rem;">Latency: <b>{stat['lat']} ms</b></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div style="background: rgba(244, 63, 94, 0.15); border: 1px solid #F43F5E; border-radius: 10px; padding: 1rem; margin-top: 0.8rem;">
                        <div style="font-weight: 700; color: #FB7185;">🔴 Connection Failed</div>
                        <div style="font-size: 0.85rem; color: var(--text-primary); margin-top: 0.3rem;">{stat['msg']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with settings_card2:
        st.markdown("### 🎛️ **Model & Decoding Parameters**")

        current_model = st.session_state.selected_model
        model_idx = SUPPORTED_MODELS.index(current_model) if current_model in SUPPORTED_MODELS else 0
        new_model = st.selectbox(
            "Gemini Model ID:",
            options=SUPPORTED_MODELS,
            index=model_idx,
            help="Select an active Gemini model with verified availability.",
        )
        if new_model != st.session_state.selected_model:
            st.session_state.selected_model = new_model
            st.rerun()

        st.markdown(
            """
            <div style="font-size: 0.8rem; color: var(--text-secondary); margin-bottom: 1rem;">
                • <b>gemini-3.5-flash-lite</b>: Recommended default (ultra-fast, highly reliable)<br>
                • <b>gemini-3.8-flash</b>: Next-generation reasoning foundation model
            </div>
            """,
            unsafe_allow_html=True,
        )

        new_temp = st.slider(
            "Sampling Temperature:",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.temperature,
            step=0.05,
            help="0.0 = Deterministic & focused; 1.0 = Diverse & creative.",
        )
        if new_temp != st.session_state.temperature:
            st.session_state.temperature = new_temp

        st.caption(
            "Tip: Use T=0.0 for Sentiment Classification and fact-based QA; use T=0.7 for creative exploration."
        )


# ===========================================================================
# PAGE 8: 📖 ABOUT & VIVA DEFENSE (Academic Report & Viva Guide)
# ===========================================================================
elif nav_selection == "📖 About & Viva Defense":
    st.markdown(
        """
        <div class="brand-hero">
            <div>
                <h1 class="brand-hero-title">📖 Academic Report & Viva Defense</h1>
                <p class="brand-hero-sub">Complete college practical report, mathematical proofs, architecture, and viva defense.</p>
            </div>
            <span class="badge-pill badge-model">College Practical Defense</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_doc1, tab_doc2, tab_doc3, tab_doc4 = st.tabs(
        ["📌 Project Overview", "🧠 NLP Theory & Math", "🏗️ System Architecture", "🎓 Viva Defense Q&A"]
    )

    with tab_doc1:
        st.markdown(
            """
            ### 🎯 Project Title:
            **PromptLab AI — Chatbot using LLM with Prompt Engineering**

            #### 1. Aim
            To build and benchmark an interactive Natural Language Processing web application that leverages Large Language Models 
            via formal prompt engineering strategies (**Zero-Shot**, **Few-Shot**, **Role Prompting**, and **Chain-of-Thought**) 
            and evaluates output fidelity through a rigorous mathematical model (**Response Quality Score — RQS**).

            #### 2. Key Objectives
            - **Real Generative AI Integration**: Live connection to Google's official Gemini API (`google-genai` SDK).
            - **Core NLP Tasks**: Practical implementation of Question Answering, Text Summarization, and Sentiment Classification.
            - **Prompt Engineering Paradigms**: Side-by-side comparative laboratory evaluating performance across strategies.
            - **Mathematical RQS Model**: Quantifying Accuracy, Relevance, Consistency, and Fluency.
            - **Cloud Deployment Ready**: Ready for Streamlit Community Cloud hosting with zero secret leakage.
            """
        )

    with tab_doc2:
        st.markdown(
            """
            ### 🧠 Theory of LLMs and Prompt Engineering

            #### 1. Large Language Models (LLMs)
            Modern LLMs are autoregressive deep neural networks based on the **Transformer architecture** (Vaswani et al., 2017). 
            They model conditional probabilities across text sequences:
            $$P(w_1, w_2, \\dots, w_T) = \\prod_{t=1}^T P(w_t \\mid w_1, \\dots, w_{t-1})$$

            #### 2. In-Context Learning & Prompt Engineering
            Unlike fine-tuning which modifies neural network weights via backpropagation, prompt engineering exploits 
            *in-context learning* (Brown et al., 2020) to steer foundation models at inference time with zero parameter updates.

            #### 3. Mathematical Model: Response Quality Score (RQS)
            $$\\text{RQS} = \\frac{\\text{Accuracy} + \\text{Relevance} + \\text{Consistency} + \\text{Fluency}}{4}$$
            Where each dimension is evaluated on an interval $[1.0, 5.0]$:
            - **Accuracy (A)**: Factual correctness and absence of hallucinations.
            - **Relevance (R)**: Pertinence to user query without drift.
            - **Consistency (C)**: Internal logical coherence and non-contradiction across claims.
            - **Fluency (F)**: Grammatical syntax, clarity, and readability.
            """
        )

    with tab_doc3:
        st.markdown(
            """
            ### 🏗️ System Architecture & Workflow

            ```mermaid
            flowchart TD
                User([👤 User / Student]) -->|Interacts via Browser| StreamlitApp[🖥️ Streamlit Frontend app.py]
                
                subgraph CoreModules [Application Core]
                    StreamlitApp --> Chat[💬 AI Assistant]
                    StreamlitApp --> Lab[🔬 Prompt Engineering Lab]
                    StreamlitApp --> QA[❓ Question Answering]
                    StreamlitApp --> Sum[📝 Text Summarization]
                    StreamlitApp --> Sent[🎭 Sentiment Classification]
                    StreamlitApp --> Eval[📊 RQS Quality Dashboard]
                end

                subgraph LogicEngine [Prompt & API Engine]
                    CoreModules --> Prompts[utils/prompts.py\nTemplate & Strategy Router]
                    Prompts --> Client[utils/gemini_client.py\nOfficial google-genai Client]
                    Client --> GeminiAPI[☁️ Google Gemini API\ngemini-3.5-flash-lite]
                    GeminiAPI --> Client
                end

                subgraph EvaluationEngine [Quality Benchmark]
                    Client --> Evaluator[utils/evaluation.py\nLLM Judge Engine]
                    Evaluator --> RQS[Compute RQS Formula]
                    RQS --> State[(utils/history.py\nSession State & Pandas)]
                    State --> CSV[CSV Export]
                end

                State -.->|Live Metrics| StreamlitApp
            ```
            """
        )

    with tab_doc4:
        st.markdown(
            """
            ### 🎓 College Viva & Defense Preparation Guide

            **Q1: Why did you choose Prompt Engineering instead of Fine-Tuning?**  
            *Answer:* Fine-tuning alters neural weights, requiring high computational resources (GPUs), domain-specific labeled training sets, and carries the risk of catastrophic forgetting. Prompt engineering leverages in-context learning in pretrained foundation models, enabling instant adaptation with zero weight modifications.

            **Q2: What is the difference between Zero-Shot and Few-Shot prompting?**  
            *Answer:* Zero-Shot provides only the task instructions. Few-Shot supplies 2–3 exemplar demonstrations directly inside the prompt context. This activates pattern matching in the self-attention mechanism, dramatically improving formatting compliance and classification stability.

            **Q3: How does the Temperature parameter affect generation?**  
            *Answer:* Temperature scales the logits before the softmax operation ($P(w_i) \\propto \\exp(z_i / T)$). At $T=0.0$, the model performs deterministic greedy decoding (ideal for classification). At higher temperatures ($T=0.7 - 1.0$), token diversity increases.

            **Q4: How did you compute RQS?**  
            *Answer:* RQS stands for Response Quality Score, computed as the arithmetic mean of four normalized dimensions: Accuracy, Relevance, Consistency, and Fluency on an interval scale of $1.0$ to $5.0$.

            **Q5: How are credentials secured?**  
            *Answer:* API keys are never hardcoded in source code or committed to Git. Locally, `.env` is managed with `python-dotenv` and protected by `.gitignore`. For cloud deployment, Streamlit Cloud Secrets are utilized.
            """
        )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "🔬 **PromptLab AI** — NLP Practical Project | Built with Python, Streamlit, and Google Gemini API."
)
