"""PromptLab AI — LLM Chatbot with Prompt Engineering.

Reference-Based Modern AI Studio Interface.
Recreates the reference dark-navy aesthetic, ambient purple/violet glowing AI orb,
3 horizontal suggestion cards, left dark sidebar with recent chats, and bottom rounded composer.
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
    load_chat_thread,
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

# ---------------------------------------------------------------------------
# Theme System & Dynamic CSS (Matching Reference UI: Navy, Violet, Cyan Glow)
# ---------------------------------------------------------------------------
is_dark = st.session_state.get("theme_mode", "dark") == "dark"

if is_dark:
    theme_vars = """
        --bg-app: #080B12;
        --bg-sidebar: #0B0E17;
        --bg-main-panel: #0E1322;
        --bg-card: rgba(18, 24, 41, 0.75);
        --bg-card-hover: rgba(28, 38, 65, 0.9);
        --bg-composer: #131A2D;
        --border-subtle: rgba(255, 255, 255, 0.08);
        --border-purple: rgba(139, 92, 246, 0.3);
        --border-purple-hover: rgba(168, 85, 247, 0.6);
        --text-primary: #F8FAFC;
        --text-secondary: #94A3B8;
        --text-muted: #64748B;
        --accent-purple-gradient: linear-gradient(135deg, #8B5CF6 0%, #6D28D9 100%);
        --accent-purple-hover: linear-gradient(135deg, #9333EA 0%, #7C3AED 100%);
        --accent-cyan: #06B6D4;
        --accent-emerald: #10B981;
        --accent-rose: #F43F5E;
        --accent-amber: #F59E0B;
        --chat-user-bg: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        --chat-assistant-bg: rgba(16, 22, 38, 0.85);
        --shadow-glow-purple: 0 0 35px -5px rgba(139, 92, 246, 0.35);
        --shadow-card: 0 4px 20px -2px rgba(0, 0, 0, 0.45);
    """
else:
    theme_vars = """
        --bg-app: #F4F6FB;
        --bg-sidebar: #FFFFFF;
        --bg-main-panel: #FFFFFF;
        --bg-card: #F8FAFC;
        --bg-card-hover: #F1F5F9;
        --bg-composer: #FFFFFF;
        --border-subtle: #E2E8F0;
        --border-purple: rgba(124, 58, 237, 0.3);
        --border-purple-hover: rgba(124, 58, 237, 0.7);
        --text-primary: #0F172A;
        --text-secondary: #475569;
        --text-muted: #94A3B8;
        --accent-purple-gradient: linear-gradient(135deg, #7C3AED 0%, #6D28D9 100%);
        --accent-purple-hover: linear-gradient(135deg, #8B5CF6 0%, #7C3AED 100%);
        --accent-cyan: #0891B2;
        --accent-emerald: #059669;
        --accent-rose: #E11D48;
        --accent-amber: #D97706;
        --chat-user-bg: linear-gradient(135deg, #6366F1 0%, #7C3AED 100%);
        --chat-assistant-bg: #F8FAFC;
        --shadow-glow-purple: 0 0 25px -5px rgba(124, 58, 237, 0.2);
        --shadow-card: 0 4px 15px -2px rgba(0, 0, 0, 0.08);
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

    /* Sidebar container customization */
    section[data-testid="stSidebar"] {{
        background-color: var(--bg-sidebar) !important;
        border-right: 1px solid var(--border-subtle);
    }}

    /* Sidebar Logo Header */
    .sidebar-header {{
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.5rem 0.2rem 1.2rem 0.2rem;
    }}
    .sidebar-logo-orb {{
        width: 36px;
        height: 36px;
        border-radius: 10px;
        background: var(--accent-purple-gradient);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        box-shadow: 0 0 15px rgba(139, 92, 246, 0.5);
    }}
    .sidebar-title {{
        font-size: 1.15rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: var(--text-primary);
        line-height: 1.2;
    }}
    .sidebar-sub {{
        font-size: 0.72rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}

    /* Custom + New Chat Button */
    .btn-new-chat {{
        display: block;
        width: 100%;
        padding: 0.7rem 1rem;
        background: var(--accent-purple-gradient);
        color: #FFFFFF !important;
        font-weight: 700;
        font-size: 0.92rem;
        border-radius: 10px;
        text-align: center;
        text-decoration: none;
        border: none;
        cursor: pointer;
        transition: all 0.2s ease;
        box-shadow: 0 4px 15px rgba(109, 40, 217, 0.4);
        margin-bottom: 1.2rem;
    }}
    .btn-new-chat:hover {{
        background: var(--accent-purple-hover);
        box-shadow: 0 6px 20px rgba(139, 92, 246, 0.6);
        transform: translateY(-1px);
    }}

    /* Sidebar Recent Chat Links */
    .recent-header {{
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-muted);
        margin-bottom: 0.6rem;
    }}
    .recent-item {{
        display: flex;
        align-items: center;
        gap: 0.6rem;
        padding: 0.55rem 0.75rem;
        border-radius: 8px;
        color: var(--text-secondary);
        font-size: 0.85rem;
        cursor: pointer;
        transition: all 0.15s ease;
        margin-bottom: 0.25rem;
        border: 1px solid transparent;
        text-overflow: ellipsis;
        white-space: nowrap;
        overflow: hidden;
    }}
    .recent-item:hover {{
        background: var(--bg-card-hover);
        color: var(--text-primary);
        border-color: var(--border-purple);
    }}
    .recent-item-active {{
        background: rgba(139, 92, 246, 0.15);
        color: #A78BFA;
        border-color: var(--border-purple);
        font-weight: 600;
    }}

    /* Sidebar Bottom Profile */
    .sidebar-footer {{
        display: flex;
        align-items: center;
        gap: 0.7rem;
        padding: 0.8rem 0.4rem;
        border-top: 1px solid var(--border-subtle);
        margin-top: 1.5rem;
    }}
    .footer-avatar {{
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background: var(--accent-purple-gradient);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.85rem;
        color: white;
        font-weight: 700;
    }}
    .footer-user {{
        font-size: 0.82rem;
        font-weight: 600;
        color: var(--text-primary);
        line-height: 1.2;
    }}
    .footer-role {{
        font-size: 0.72rem;
        color: var(--text-muted);
    }}

    /* Main Chat Panel Container */
    .main-chat-container {{
        background: var(--bg-main-panel);
        border: 1px solid var(--border-subtle);
        border-radius: 18px;
        padding: 1.2rem 1.6rem;
        min-height: 85vh;
        box-shadow: var(--shadow-card);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }}

    /* Top Workspace Bar (Model selector & actions) */
    .workspace-top-bar {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding-bottom: 0.8rem;
        border-bottom: 1px solid var(--border-subtle);
        margin-bottom: 1.5rem;
    }}
    .model-pill {{
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.3rem 0.8rem;
        background: rgba(139, 92, 246, 0.12);
        border: 1px solid var(--border-purple);
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        color: #C4B5FD;
    }}

    /* Glowing AI Center Orb */
    .orb-container {{
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 2rem 0 1.2rem 0;
    }}
    .glowing-orb {{
        width: 90px;
        height: 90px;
        border-radius: 50%;
        background: radial-gradient(circle at 35% 35%, #F472B6 0%, #A855F7 45%, #6366F1 80%, #312E81 100%);
        box-shadow: 0 0 45px 10px rgba(168, 85, 247, 0.4), inset 0 0 20px rgba(255, 255, 255, 0.6);
        animation: orb-pulse 4s ease-in-out infinite;
    }}
    @keyframes orb-pulse {{
        0%, 100% {{
            transform: scale(1);
            filter: drop-shadow(0 0 25px rgba(168, 85, 247, 0.5));
        }}
        50% {{
            transform: scale(1.08);
            filter: drop-shadow(0 0 40px rgba(236, 72, 153, 0.7));
        }}
    }}

    /* Welcome Heading & Subtitle */
    .welcome-title {{
        text-align: center;
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
    }}
    .welcome-sub {{
        text-align: center;
        font-size: 0.95rem;
        color: var(--text-secondary);
        max-width: 620px;
        margin: 0 auto 2.2rem auto;
        line-height: 1.5;
    }}

    /* 3 Horizontal Suggestion Cards (Matching Reference) */
    .suggestion-row {{
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1.2rem;
        margin-bottom: 2.5rem;
    }}
    .suggestion-card {{
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 14px;
        padding: 1.3rem;
        text-align: left;
        cursor: pointer;
        transition: all 0.25s ease;
        box-shadow: var(--shadow-card);
    }}
    .suggestion-card:hover {{
        transform: translateY(-3px);
        border-color: var(--border-purple-hover);
        box-shadow: var(--shadow-glow-purple);
        background: var(--bg-card-hover);
    }}
    .card-icon-pill {{
        width: 38px;
        height: 38px;
        border-radius: 10px;
        background: rgba(139, 92, 246, 0.15);
        border: 1px solid var(--border-purple);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        margin-bottom: 0.8rem;
    }}
    .card-title {{
        font-size: 1rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.35rem;
    }}
    .card-desc {{
        font-size: 0.82rem;
        color: var(--text-secondary);
        line-height: 1.45;
    }}

    /* Chat Messages Styling */
    .chat-bubble-user {{
        background: var(--chat-user-bg);
        color: #FFFFFF;
        padding: 0.9rem 1.3rem;
        border-radius: 16px 16px 4px 16px;
        margin: 0.6rem 0;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.3);
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
        max-width: 92%;
        box-shadow: var(--shadow-card);
        font-size: 0.95rem;
        line-height: 1.6;
    }}

    /* Message Meta & Badges */
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

    /* Strategy Toolbar */
    .strategy-bar {{
        display: flex;
        gap: 0.5rem;
        align-items: center;
        margin-bottom: 0.6rem;
    }}

    /* Responsive */
    @media (max-width: 900px) {{
        .suggestion-row {{
            grid-template-columns: 1fr;
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
# Sidebar: Logo, + New Chat, Recent Chats, Settings, Help & Profile
# ---------------------------------------------------------------------------
with st.sidebar:
    # 1. PromptLab AI Logo
    st.markdown(
        """
        <div class="sidebar-header">
            <div class="sidebar-logo-orb">⚛️</div>
            <div>
                <div class="sidebar-title">PromptLab AI</div>
                <div class="sidebar-sub">LLM Prompt Studio</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Prominent Purple Gradient + New Chat Button
    if st.button("➕  New Chat", key="btn_sidebar_new_chat", use_container_width=True, type="primary"):
        clear_chat_history()
        st.session_state.active_nav = "💬 Chat"
        st.rerun()

    st.markdown("<div style='height: 0.6rem;'></div>", unsafe_allow_html=True)

    # 3. Main Navigation
    st.markdown('<div class="recent-header">Workspace</div>', unsafe_allow_html=True)
    nav_tabs = [
        "💬 Chat",
        "🔬 Prompt Engineering Lab",
        "❓ Question Answering",
        "📝 Text Summarization",
        "🎭 Sentiment Classification",
        "📊 Response Quality (RQS)",
    ]

    selected_nav = st.radio(
        "Workspace View:",
        options=nav_tabs,
        index=nav_tabs.index(st.session_state.get("active_nav", "💬 Chat")) if st.session_state.get("active_nav") in nav_tabs else 0,
        label_visibility="collapsed",
    )
    if selected_nav != st.session_state.get("active_nav"):
        st.session_state.active_nav = selected_nav
        st.rerun()

    st.divider()

    # 4. Recent Chats Section
    st.markdown('<div class="recent-header">Recent Chats</div>', unsafe_allow_html=True)
    recent_threads = st.session_state.get("recent_chats", [])

    for thread in recent_threads[:4]:
        t_title = thread.get("title", "Conversation")
        t_id = thread.get("id", "")
        is_active = t_id == st.session_state.get("active_chat_id")
        active_cls = "recent-item-active" if is_active else ""

        col_t, col_del = st.columns([5, 1])
        with col_t:
            if st.button(f"💬 {t_title}", key=f"thread_{t_id}", use_container_width=True):
                load_chat_thread(t_id)
                st.session_state.active_nav = "💬 Chat"
                st.rerun()

    st.divider()

    # 5. Settings & Help Navigation
    col_settings, col_about = st.columns(2)
    with col_settings:
        if st.button("⚙️ Settings", use_container_width=True):
            st.session_state.active_nav = "⚙️ Settings"
            st.rerun()
    with col_about:
        if st.button("📖 Help / Viva", use_container_width=True):
            st.session_state.active_nav = "📖 Help / Viva"
            st.rerun()

    # Theme Toggle Switcher
    col_th_lbl, col_th_btn = st.columns([2, 1])
    with col_th_lbl:
        st.caption(f"Theme: **{st.session_state.get('theme_mode', 'dark').capitalize()}**")
    with col_th_btn:
        toggle_label = "☀️" if is_dark else "🌙"
        if st.button(toggle_label, help="Toggle Dark/Light theme"):
            st.session_state.theme_mode = "light" if is_dark else "dark"
            st.rerun()

    # 6. Bottom Profile Section
    st.markdown(
        """
        <div class="sidebar-footer">
            <div class="footer-avatar">AI</div>
            <div>
                <div class="footer-user">Adinath • Student</div>
                <div class="footer-role">NLP Practical Project</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Helper: Client Resolution
# ---------------------------------------------------------------------------
def resolve_client():
    """Retrieve Gemini client using active key with user guidance if missing."""
    key = st.session_state.custom_api_key.strip() or get_api_key()
    if not key:
        st.error(
            "🛑 **Gemini API Key Required**\n\n"
            "Please configure your Google Gemini API key by clicking **⚙️ Settings** in the left sidebar, "
            "or adding `GEMINI_API_KEY` to `.env`."
        )
        return None
    return get_gemini_client(key)


# ===========================================================================
# VIEW 1: 💬 MAIN CHAT WORKSPACE (Matching Reference Screenshot)
# ===========================================================================
if st.session_state.get("active_nav") == "💬 Chat":
    # Top Workspace Bar: Model Selector on Left, Options on Right
    top_col1, top_col2 = st.columns([3, 2])
    with top_col1:
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; gap: 0.6rem;">
                <span class="model-pill">✨ {st.session_state.selected_model}</span>
                <span style="font-size: 0.78rem; color: var(--text-muted);">Temperature: {st.session_state.temperature}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with top_col2:
        top_act1, top_act2 = st.columns(2)
        with top_act1:
            if st.button("🔬 Open Lab", use_container_width=True):
                st.session_state.active_nav = "🔬 Prompt Engineering Lab"
                st.rerun()
        with top_act2:
            if st.button("🧹 Clear Chat", use_container_width=True):
                clear_chat_history()
                st.rerun()

    st.markdown("<hr style='border: none; border-top: 1px solid var(--border-subtle); margin: 0.4rem 0 1rem 0;'>", unsafe_allow_html=True)

    chat_history = get_chat_history()

    # Initial Screen: Animated AI Orb, Heading, Subtitle & 3 Horizontal Cards
    if not chat_history:
        st.markdown(
            """
            <div class="orb-container">
                <div class="glowing-orb"></div>
            </div>
            <div class="welcome-title">Welcome to PromptLab AI</div>
            <div class="welcome-sub">
                Your intelligent workspace for questions, writing, summarization, and prompt engineering.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 3 Suggestion Cards in One Horizontal Row
        card_col1, card_col2, card_col3 = st.columns(3)

        with card_col1:
            if st.button(
                "💬  Ask Anything\n\nGet clear, helpful answers to your questions.",
                key="card_btn_qa",
                use_container_width=True,
            ):
                add_chat_message("user", "Explain how the self-attention mechanism works in transformer models.")
                st.rerun()

        with card_col2:
            if st.button(
                "📝  Summarize Text\n\nTurn long content into concise summaries.",
                key="card_btn_sum",
                use_container_width=True,
            ):
                st.session_state.active_nav = "📝 Text Summarization"
                st.rerun()

        with card_col3:
            if st.button(
                "🎭  Analyze Sentiment\n\nClassify text as positive, negative, or neutral.",
                key="card_btn_sent",
                use_container_width=True,
            ):
                st.session_state.active_nav = "🎭 Sentiment Classification"
                st.rerun()

        st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)

    # Active Conversation History Rendering
    else:
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
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
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

    # Strategy Selector Bar directly above composer
    strat_col, persona_col = st.columns([1, 1])
    with strat_col:
        strat_choice = st.segmented_control(
            "Prompt Strategy:",
            options=["Zero-Shot", "Few-Shot", "Role Prompting", "Chain-of-Thought"],
            default=st.session_state.get("prompt_strategy", "Zero-Shot"),
            key="chat_strat_control",
        )
        if strat_choice:
            st.session_state.prompt_strategy = strat_choice

    with persona_col:
        persona_choice = st.selectbox(
            "Persona Grounding:",
            options=[
                "NLP & AI Professor (Pedagogical & grounded)",
                "Principal Software Architect (System design & code)",
                "Concise Assistant (Direct & factual)",
            ],
            index=0,
            label_visibility="collapsed",
        )

    # Large Bottom Message Composer (Matching Reference)
    user_msg = st.chat_input("Send a message or ask anything...")
    if user_msg:
        add_chat_message("user", user_msg)
        st.markdown(
            f"""
            <div class="chat-bubble-user">
                <div style="font-weight: 700; font-size: 0.78rem; opacity: 0.85; margin-bottom: 0.2rem;">YOU</div>
                {user_msg}
            </div>
            """,
            unsafe_allow_html=True,
        )

        client = resolve_client()
        if client:
            strategy_enum_map = {
                "Zero-Shot": PromptStrategy.ZERO_SHOT,
                "Few-Shot": PromptStrategy.FEW_SHOT,
                "Role Prompting": PromptStrategy.ROLE_PROMPT,
                "Chain-of-Thought": PromptStrategy.CHAIN_OF_THOUGHT,
            }
            active_strat = strategy_enum_map.get(st.session_state.prompt_strategy, PromptStrategy.ZERO_SHOT)
            sys_inst, formatted_p = build_prompt(
                task=NLPTask.QUESTION_ANSWERING,
                strategy=active_strat,
                user_input=user_msg,
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
# VIEW 2: 🔬 PROMPT ENGINEERING LAB (Side-by-side Strategy Benchmarking)
# ===========================================================================
elif st.session_state.get("active_nav") == "🔬 Prompt Engineering Lab":
    st.markdown("## 🔬 **Prompt Engineering Laboratory**")
    st.caption("Benchmark Zero-Shot, Few-Shot, Role Prompting, and Chain-of-Thought on identical inputs.")

    lab_task_str = st.selectbox(
        "Select NLP Task:",
        options=[t.value for t in NLPTask],
        index=0,
    )
    task_enum = NLPTask(lab_task_str)

    default_test_inputs = {
        NLPTask.QUESTION_ANSWERING.value: "How does the Scaled Dot-Product Attention compute similarity between queries and keys?",
        NLPTask.SUMMARIZATION.value: (
            "The Transformer model relies entirely on self-attention mechanisms to compute representations of its input and output without using sequence-aligned RNNs or convolution. "
            "In sequence transduction models, recurrent neural networks generate a sequence of hidden states as a function of the previous hidden state and the current input token. "
            "This inherently sequential nature precludes parallelization within training examples, which becomes critical at longer sequence lengths, as memory constraints limit batching. "
            "Attention mechanisms have become an integral part of compelling sequence modeling, allowing modeling of dependencies without regard to their distance in the input or output sequences."
        ),
        NLPTask.SENTIMENT_ANALYSIS.value: "The software UI is beautifully designed, but the unexpected server crashes make it unusable for production.",
    }

    lab_input = st.text_area(
        "Test Input for Multi-Strategy Execution:",
        value=default_test_inputs[lab_task_str],
        height=120,
    )

    st.markdown("##### 🎯 **Select Strategies to Benchmark:**")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        run_zs = st.checkbox("Zero-Shot", value=True)
    with c2:
        run_fs = st.checkbox("Few-Shot", value=True)
    with c3:
        run_rp = st.checkbox("Role Prompting", value=True)
    with c4:
        run_cot = st.checkbox("Chain-of-Thought", value=False)

    selected_strats = []
    if run_zs:
        selected_strats.append(PromptStrategy.ZERO_SHOT)
    if run_fs:
        selected_strats.append(PromptStrategy.FEW_SHOT)
    if run_rp:
        selected_strats.append(PromptStrategy.ROLE_PROMPT)
    if run_cot:
        selected_strats.append(PromptStrategy.CHAIN_OF_THOUGHT)

    if st.button("⚡ Run Comparative Experiment", type="primary", use_container_width=True):
        is_valid, validation_err = validate_input_text(lab_input, min_length=3)
        if not is_valid:
            st.warning(validation_err)
        elif not selected_strats:
            st.warning("Please select at least one strategy.")
        else:
            client = resolve_client()
            if client:
                res_cols = st.columns(len(selected_strats))
                metrics_data = []

                progress_bar = st.progress(0, text="Executing prompts across Gemini...")
                for idx, strat in enumerate(selected_strats):
                    sys_inst, user_prompt = build_prompt(task_enum, strat, lab_input)

                    with res_cols[idx]:
                        st.markdown(f"#### 🎯 **{strat.value}**")
                        with st.spinner(f"Generating ({strat.value})..."):
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
                            st.caption(f"⚡ Latency: **{lat} ms**")

                            with st.expander("🔍 View Prompt Template"):
                                st.markdown(format_prompt_for_display(sys_inst, user_prompt))

                            st.markdown(
                                f"""
                                <div style="background: var(--bg-card); border: 1px solid var(--border-subtle); border-radius: 10px; padding: 1rem; margin-bottom: 0.8rem; font-size: 0.9rem;">
                                    {reply}
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                            # Automated RQS scoring
                            with st.spinner("Scoring quality..."):
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
                                st.metric("RQS Score", f"{eval_res.rqs_score}/5.0")
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
                    st.markdown("### 📊 **Comparative Analysis Summary**")
                    comp_df = pd.DataFrame(metrics_data)
                    st.dataframe(comp_df, use_container_width=True)

                    c_col1, c_col2 = st.columns(2)
                    with c_col1:
                        st.markdown("##### 🏆 RQS Score Comparison")
                        st.bar_chart(comp_df.set_index("Strategy")["RQS Score"])
                    with c_col2:
                        st.markdown("##### ⏱️ Latency Comparison (ms)")
                        st.bar_chart(comp_df.set_index("Strategy")["Latency (ms)"])


# ===========================================================================
# VIEW 3: ❓ QUESTION ANSWERING TOOL (Task 1)
# ===========================================================================
elif st.session_state.get("active_nav") == "❓ Question Answering":
    st.markdown("## ❓ **Question Answering Engine**")
    st.caption("Technical and pedagogical explanations with uncertainty acknowledgment.")

    q_col1, q_col2 = st.columns([1, 1])
    with q_col1:
        qa_strat = st.selectbox("Prompt Strategy:", [s.value for s in PromptStrategy], index=2)
    with q_col2:
        preset_q = st.selectbox(
            "Load Sample NLP Question:",
            options=[
                "-- Custom Question --",
                "What is Self-Attention and how does it compute query, key, and value vectors?",
                "How does Temperature affect LLM decoding probability distributions?",
                "Explain the difference between Extractive and Abstractive text summarization.",
            ],
        )

    initial_q = preset_q if preset_q != "-- Custom Question --" else ""
    user_q = st.text_area("Enter Technical Question:", value=initial_q, height=100)

    if st.button("🚀 Generate Answer", type="primary"):
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

                with st.spinner("Synthesizing answer..."):
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
                    st.success(f"Generated ({lat} ms)")
                    st.markdown("### 💡 **Answer**")
                    st.markdown(reply)

                    with st.expander("🔍 View Prompt Template"):
                        st.markdown(format_prompt_for_display(sys_inst, user_prompt))

                    st.divider()
                    if st.button("Evaluate with RQS", key="eval_qa_btn"):
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
                            st.success(f"**RQS Score: {eval_res.rqs_score} / 5.0**")
                            q1, q2, q3, q4 = st.columns(4)
                            q1.metric("Accuracy", f"{eval_res.accuracy}/5")
                            q2.metric("Relevance", f"{eval_res.relevance}/5")
                            q3.metric("Consistency", f"{eval_res.consistency}/5")
                            q4.metric("Fluency", f"{eval_res.fluency}/5")


# ===========================================================================
# VIEW 4: 📝 TEXT SUMMARIZATION (Task 2)
# ===========================================================================
elif st.session_state.get("active_nav") == "📝 Text Summarization":
    st.markdown("## 📝 **Text Summarization Engine**")
    st.caption("Factual condensation across Short, Medium, and Detailed modes.")

    len_col, strat_col = st.columns(2)
    with len_col:
        len_mode = st.radio("Target Length:", ["Short", "Medium", "Detailed"], index=1, horizontal=True)
    with strat_col:
        sum_strat = st.selectbox("Prompt Strategy:", [s.value for s in PromptStrategy], index=2)

    sample_1 = (
        "The Attention mechanism in deep learning was introduced to address the information bottleneck in Sequence-to-Sequence models. "
        "Traditional RNNs compress all source information into a single fixed-length context vector, which struggles with long sequences. "
        "Bahdanau attention resolved this by computing dynamic alignment scores between decoder hidden states and all encoder outputs. "
        "In 2017, the Transformer architecture entirely replaced recurrent layers with Self-Attention, computing Query, Key, and Value projections. "
        "This allowed massive parallelization during training and enabled foundational LLMs like BERT, GPT, and Gemini."
    )

    sum_input = st.text_area("Paste Article or Text to Summarize:", value=sample_1, height=140)

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

                with st.spinner("Synthesizing summary..."):
                    summary, err_msg, lat = generate_single_response(
                        client=client,
                        prompt=user_prompt,
                        model=st.session_state.selected_model,
                        temperature=0.3,
                        system_instruction=sys_inst,
                    )

                if err_msg:
                    st.error(err_msg)
                elif summary:
                    st.session_state.queries_count = st.session_state.get("queries_count", 0) + 1
                    st.success(f"Generated ({lat} ms)")
                    st.markdown("### 📋 **Summary**")
                    st.markdown(summary)

                    with st.expander("🔍 View Prompt Template"):
                        st.markdown(format_prompt_for_display(sys_inst, user_prompt))

                    st.divider()
                    if st.button("Evaluate Summary with RQS", key="eval_sum_btn"):
                        with st.spinner("Evaluating quality..."):
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
                            q1.metric("Accuracy", f"{eval_res.accuracy}/5")
                            q2.metric("Relevance", f"{eval_res.relevance}/5")
                            q3.metric("Consistency", f"{eval_res.consistency}/5")
                            q4.metric("Fluency", f"{eval_res.fluency}/5")


# ===========================================================================
# VIEW 5: 🎭 SENTIMENT CLASSIFICATION (Task 3)
# ===========================================================================
elif st.session_state.get("active_nav") == "🎭 Sentiment Classification":
    st.markdown("## 🎭 **Sentiment Classification Engine**")
    st.caption("Deterministic categorization into Positive, Negative, or Neutral with linguistic rationale.")

    strat_col, _ = st.columns([1, 1])
    with strat_col:
        sent_strat = st.selectbox("Prompt Strategy:", [s.value for s in PromptStrategy], index=1)

    st.markdown("##### 🧪 **College Practical Benchmark Cases (Click to Test):**")
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

                with st.spinner("Analyzing polarity..."):
                    prediction, err_msg, lat = generate_single_response(
                        client=client,
                        prompt=user_prompt,
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

                    with st.expander("🔍 View Prompt Template"):
                        st.markdown(format_prompt_for_display(sys_inst, user_prompt))

                    st.divider()
                    if st.button("Evaluate with RQS", key="eval_sent_btn"):
                        with st.spinner("Scoring quality..."):
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
                            q1.metric("Accuracy", f"{eval_res.accuracy}/5")
                            q2.metric("Relevance", f"{eval_res.relevance}/5")
                            q3.metric("Consistency", f"{eval_res.consistency}/5")
                            q4.metric("Fluency", f"{eval_res.fluency}/5")


# ===========================================================================
# VIEW 6: 📊 RESPONSE QUALITY (RQS Dashboard & CSV Export)
# ===========================================================================
elif st.session_state.get("active_nav") == "📊 Response Quality (RQS)":
    st.markdown("## 📊 **Response Quality Evaluation Dashboard (RQS)**")

    # Formula Display
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

    # Standalone Evaluator
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
# VIEW 7: ⚙️ SETTINGS (API Key, Model, Temperature, Diagnostics)
# ===========================================================================
elif st.session_state.get("active_nav") == "⚙️ Settings":
    st.markdown("## ⚙️ **Settings & System Health**")

    c_key, c_model = st.columns(2)
    with c_key:
        st.markdown("### 🔑 **API Key Configuration**")
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

    with c_model:
        st.markdown("### 🎛️ **Model & Decoding Parameters**")
        new_model = st.selectbox(
            "Gemini Model ID:",
            options=SUPPORTED_MODELS,
            index=SUPPORTED_MODELS.index(st.session_state.selected_model) if st.session_state.selected_model in SUPPORTED_MODELS else 0,
        )
        if new_model != st.session_state.selected_model:
            st.session_state.selected_model = new_model
            st.rerun()

        new_temp = st.slider("Temperature:", 0.0, 1.0, float(st.session_state.temperature), 0.05)
        if new_temp != st.session_state.temperature:
            st.session_state.temperature = new_temp


# ===========================================================================
# VIEW 8: 📖 HELP / VIVA (Academic Documentation)
# ===========================================================================
elif st.session_state.get("active_nav") == "📖 Help / Viva":
    st.markdown("## 📖 **Academic Report & Viva Defense Guide**")
    t1, t2, t3 = st.tabs(["📌 Project Overview", "🧠 NLP Theory & Math", "🎓 Viva Defense Q&A"])

    with t1:
        st.markdown(
            """
            ### 🎯 Project Title:
            **PromptLab AI — Chatbot using LLM with Prompt Engineering**
            - **Author:** Adinath (NLP Student Researcher)
            - **Platform:** Streamlit & Google Gemini API (`google-genai` SDK)
            - **Core Tasks:** Question Answering, Text Summarization, Sentiment Classification
            - **Quality Model:** Response Quality Score (RQS)
            """
        )
    with t2:
        st.markdown(
            """
            ### 🧠 Theory & Mathematical Formulation
            $$\\text{RQS} = \\frac{\\text{Accuracy} + \\text{Relevance} + \\text{Consistency} + \\text{Fluency}}{4}$$
            - **Accuracy**: Truthfulness and absence of hallucination.
            - **Relevance**: Direct alignment with user query.
            - **Consistency**: Internal logical cohesion across claims.
            - **Fluency**: Grammatical precision and syntax.
            """
        )
    with t3:
        st.markdown(
            """
            ### 🎓 Key Viva Questions
            - **Q: Why prompt engineering over fine-tuning?**  
              *A:* Fine-tuning modifies neural network weights, requiring high computational resources and domain training sets. Prompt engineering steers foundation models via in-context learning with zero weight modification.
            - **Q: How does few-shot prompting work?**  
              *A:* Supplying 2-3 input-output examples directly in the prompt context activates pattern recognition in the self-attention mechanism, standardizing format and output labels.
            """
        )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption("⚛️ **PromptLab AI** — NLP Practical Project | Developed with Python, Streamlit, and Google Gemini API.")
