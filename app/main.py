"""
Point d'entrée principal de l'application Streamlit.

Assistant Juridique RAG pour le cabinet Parenti & Associés.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from app.config import APP_TITLE, PAGE_CHAT, PAGE_DOCUMENTS
from app.views.chat import render_chat_page, render_sidebar as render_chat_sidebar
from app.views.documents import render_documents_page
from app.utils.logger import setup_logging


def inject_custom_css() -> None:
    """Injecte le CSS personnalisé pour un design moderne."""
    st.markdown("""
        <style>
        /* ===== IMPORTS ===== */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        /* ===== CSS VARIABLES ===== */
        :root {
            --primary: #2563eb;
            --primary-light: #3b82f6;
            --primary-dark: #1d4ed8;
            --primary-bg: rgba(37, 99, 235, 0.08);

            --success: #059669;
            --success-bg: rgba(5, 150, 105, 0.08);

            --warning: #d97706;
            --warning-bg: rgba(217, 119, 6, 0.08);

            --error: #dc2626;
            --error-bg: rgba(220, 38, 38, 0.08);

            --gray-50: #f9fafb;
            --gray-100: #f3f4f6;
            --gray-200: #e5e7eb;
            --gray-300: #d1d5db;
            --gray-400: #9ca3af;
            --gray-500: #6b7280;
            --gray-600: #4b5563;
            --gray-700: #374151;
            --gray-800: #1f2937;
            --gray-900: #111827;

            --white: #ffffff;
            --black: #000000;

            --sidebar-bg: linear-gradient(180deg, #1e3a8a 0%, #1d4ed8 100%);

            --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
            --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.08), 0 2px 4px -2px rgb(0 0 0 / 0.08);
            --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.08), 0 4px 6px -4px rgb(0 0 0 / 0.08);

            --radius-sm: 6px;
            --radius-md: 10px;
            --radius-lg: 14px;
            --radius-xl: 20px;
            --radius-full: 9999px;
        }

        /* ===== GLOBAL STYLES ===== */
        .stApp {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--gray-50);
        }

        /* Hide default header */
        header[data-testid="stHeader"] {
            display: none;
        }

        /* Main container - plus compact */
        .main .block-container {
            padding: 1.25rem 2rem 3rem 2rem;
            max-width: 1000px;
        }

        /* ===== SIDEBAR ===== */
        section[data-testid="stSidebar"] {
            background: var(--sidebar-bg);
            border-right: none;
            width: 280px !important;
        }

        section[data-testid="stSidebar"] > div:first-child {
            padding: 0 0.75rem 0.75rem 0.75rem;
        }

        section[data-testid="stSidebar"] .stMarkdown,
        section[data-testid="stSidebar"] .stMarkdown p {
            color: white !important;
        }

        section[data-testid="stSidebar"] hr {
            border-color: rgba(255, 255, 255, 0.2);
            margin: 0.75rem 0;
        }

        /* Style de base pour les boutons sidebar - bleu foncé élégant */
        section[data-testid="stSidebar"] .stButton > button {
            background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 500 !important;
            font-size: 0.73rem !important;
            padding: 0.45rem 0.7rem !important;
            transition: all 0.2s ease !important;
            text-align: left !important;
            min-height: 34px !important;
            line-height: 1.3 !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2) !important;
        }

        section[data-testid="stSidebar"] .stButton > button:hover {
            background: linear-gradient(135deg, #152e70 0%, #1a3a9e 100%) !important;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.25) !important;
        }

        /* Bouton nouvelle conversation - blanc élégant (doit être APRES le style de base) */
        section[data-testid="stSidebar"] .stButton > button[kind="primary"],
        section[data-testid="stSidebar"] .stButton > button[kind="primary"] p,
        section[data-testid="stSidebar"] .stButton > button[kind="primary"] span,
        section[data-testid="stSidebar"] .stButton > button[kind="primary"] div {
            background: linear-gradient(135deg, #ffffff 0%, #f0f4ff 100%) !important;
            color: #1e3a8a !important;
            font-weight: 600 !important;
            font-size: 0.8rem !important;
            padding: 0.6rem 0.85rem !important;
            min-height: 38px !important;
            border-radius: 10px !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15) !important;
        }

        section[data-testid="stSidebar"] .stButton > button[kind="primary"] p,
        section[data-testid="stSidebar"] .stButton > button[kind="primary"] span,
        section[data-testid="stSidebar"] .stButton > button[kind="primary"] div {
            background: transparent !important;
            padding: 0 !important;
            min-height: auto !important;
            box-shadow: none !important;
            border-radius: 0 !important;
        }

        section[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
            background: linear-gradient(135deg, #ffffff 0%, #e0e7ff 100%) !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
            transform: translateY(-1px) !important;
        }

        /* Bouton supprimer - rouge élégant (petite colonne) */
        section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > [data-testid="stVerticalBlockBorderWrapper"]:last-child .stButton > button,
        section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > div:last-child .stButton > button {
            background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%) !important;
            color: white !important;
            box-shadow: 0 1px 3px rgba(185, 28, 28, 0.3) !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            text-align: center !important;
            padding: 0 !important;
        }

        section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > [data-testid="stVerticalBlockBorderWrapper"]:last-child .stButton > button:hover,
        section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > div:last-child .stButton > button:hover {
            background: linear-gradient(135deg, #b91c1c 0%, #991b1b 100%) !important;
            box-shadow: 0 2px 6px rgba(153, 27, 27, 0.4) !important;
        }

        section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > [data-testid="stVerticalBlockBorderWrapper"]:last-child .stButton > button p,
        section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > div:last-child .stButton > button p {
            margin: 0 !important;
            padding: 0 !important;
            line-height: 1 !important;
        }

        section[data-testid="stSidebar"] .stCaption {
            color: rgba(255, 255, 255, 0.6) !important;
            font-size: 0.75rem;
        }

        /* ===== TOP NAVIGATION ===== */
        .top-nav-container {
            display: flex;
            justify-content: center;
            margin-bottom: 1.5rem;
        }

        .top-nav-pills {
            display: inline-flex;
            background: var(--white);
            border-radius: var(--radius-full);
            padding: 4px;
            box-shadow: var(--shadow-md);
            border: 1px solid var(--gray-200);
            gap: 4px;
        }

        /* Style des boutons de navigation en haut */
        .main .stButton > button {
            background: transparent;
            color: var(--gray-600);
            border: none;
            border-radius: var(--radius-full);
            padding: 0.5rem 1.25rem;
            font-weight: 500;
            font-size: 0.875rem;
            transition: all 0.15s ease;
            box-shadow: none;
            min-height: 36px;
        }

        .main .stButton > button:hover {
            background: var(--gray-100);
            color: var(--gray-800);
            transform: none;
            box-shadow: none;
        }

        .main .stButton > button[kind="primary"] {
            background: var(--primary);
            color: white;
            box-shadow: 0 2px 8px rgba(37, 99, 235, 0.3);
        }

        .main .stButton > button[kind="primary"]:hover {
            background: var(--primary-dark);
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35);
        }

        /* ===== CHAT STYLES ===== */
        .stChatMessage {
            background: var(--white);
            border-radius: var(--radius-lg);
            padding: 1rem;
            margin-bottom: 0.75rem;
            box-shadow: var(--shadow-sm);
            border: 1px solid var(--gray-100);
        }

        [data-testid="stChatMessageContent"] {
            font-size: 0.9rem;
            line-height: 1.65;
            color: var(--gray-700);
        }

        /* Chat input - plus compact */
        .stChatInput > div {
            border-radius: var(--radius-lg) !important;
            border: 1px solid var(--gray-300) !important;
            background: var(--white) !important;
            box-shadow: var(--shadow-sm) !important;
        }

        .stChatInput > div:focus-within {
            border-color: var(--primary) !important;
            box-shadow: 0 0 0 3px var(--primary-bg) !important;
        }

        .stChatInput textarea {
            font-size: 0.9rem;
            padding: 0.75rem 1rem;
        }

        /* ===== FILE UPLOADER - plus compact ===== */
        [data-testid="stFileUploader"] {
            background: var(--white);
            border: 2px dashed var(--gray-300);
            border-radius: var(--radius-lg);
            padding: 1.25rem;
            transition: all 0.2s ease;
        }

        [data-testid="stFileUploader"]:hover {
            border-color: var(--primary);
            background: var(--primary-bg);
        }

        /* ===== EXPANDER - plus compact ===== */
        .streamlit-expanderHeader {
            background: var(--gray-50);
            border-radius: var(--radius-md);
            font-weight: 500;
            font-size: 0.875rem;
            color: var(--gray-700);
            padding: 0.75rem 1rem;
            border: 1px solid var(--gray-200);
        }

        .streamlit-expanderContent {
            background: var(--white);
            border: 1px solid var(--gray-200);
            border-top: none;
            border-radius: 0 0 var(--radius-md) var(--radius-md);
            padding: 0.75rem 1rem;
        }

        /* ===== ALERTS - plus compacts ===== */
        .stAlert {
            border-radius: var(--radius-md);
            padding: 0.75rem 1rem;
            font-size: 0.875rem;
        }

        .stSuccess { background: var(--success-bg); border-left: 3px solid var(--success); }
        .stWarning { background: var(--warning-bg); border-left: 3px solid var(--warning); }
        .stError { background: var(--error-bg); border-left: 3px solid var(--error); }
        .stInfo { background: var(--primary-bg); border-left: 3px solid var(--primary); }

        /* ===== CUSTOM COMPONENTS ===== */
        .welcome-container {
            text-align: center;
            padding: 2.5rem 1.5rem;
            background: var(--white);
            border-radius: var(--radius-xl);
            border: 1px solid var(--gray-200);
            margin: 1rem 0;
        }

        .welcome-icon { font-size: 3rem; margin-bottom: 1rem; }
        .welcome-title { font-size: 1.25rem; font-weight: 600; color: var(--gray-800); margin-bottom: 0.5rem; }
        .welcome-subtitle { font-size: 0.9rem; color: var(--gray-500); max-width: 350px; margin: 0 auto; line-height: 1.5; }

        .logo-container {
            text-align: center;
            padding: 0.5rem 0.75rem 1rem 0.75rem;
            margin-top: -1rem;
        }

        .logo-icon {
            width: 48px;
            height: 48px;
            background: rgba(255, 255, 255, 0.15);
            border-radius: var(--radius-md);
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 0.75rem auto;
            font-size: 1.5rem;
        }

        .logo-text {
            font-size: 1rem;
            font-weight: 600;
            color: white;
            margin-bottom: 0.15rem;
        }

        .logo-subtitle {
            font-size: 0.7rem;
            color: rgba(255, 255, 255, 0.7);
            font-weight: 500;
        }

        .nav-label {
            font-size: 0.65rem;
            font-weight: 600;
            color: rgba(255, 255, 255, 0.5);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.5rem;
            padding-left: 0.25rem;
        }

        .section-header {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            margin-bottom: 1rem;
            padding-bottom: 0.75rem;
            border-bottom: 1px solid var(--gray-200);
        }

        .section-icon {
            width: 32px;
            height: 32px;
            background: var(--primary-bg);
            border-radius: var(--radius-sm);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1rem;
        }

        .section-title {
            font-size: 1rem;
            font-weight: 600;
            color: var(--gray-800);
        }

        .stat-card {
            background: var(--white);
            border-radius: var(--radius-md);
            padding: 1rem;
            border: 1px solid var(--gray-200);
            text-align: center;
        }

        .stat-value {
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--primary);
            margin-bottom: 0.15rem;
        }

        .stat-label {
            font-size: 0.75rem;
            color: var(--gray-500);
            font-weight: 500;
        }

        .chat-header {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.75rem 1rem;
            background: var(--white);
            border-radius: var(--radius-md);
            margin-bottom: 1rem;
            border: 1px solid var(--gray-200);
        }

        .chat-avatar {
            width: 40px;
            height: 40px;
            background: var(--primary);
            border-radius: var(--radius-sm);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.25rem;
        }

        .chat-info h3 {
            font-size: 0.95rem;
            font-weight: 600;
            color: var(--gray-800);
            margin: 0 0 0.15rem 0;
        }

        .chat-info p {
            font-size: 0.8rem;
            color: var(--gray-500);
            margin: 0;
        }

        /* Scrollbar plus fine */
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: var(--gray-300); border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: var(--gray-400); }

        /* Masquer les éléments Streamlit par défaut */
        #MainMenu, footer, .stDeployButton { display: none !important; }

        /* Divider */
        hr { border: none; height: 1px; background: var(--gray-200); margin: 1rem 0; }
        </style>
    """, unsafe_allow_html=True)


def render_navigation() -> str:
    """Affiche la navigation et retourne la page sélectionnée."""

    # Initialiser la page courante dans session_state
    if "current_page" not in st.session_state:
        st.session_state.current_page = PAGE_CHAT

    # Sidebar pour les conversations uniquement
    with st.sidebar:
        # Logo compact
        st.markdown("""
            <div class="logo-container">
                <div class="logo-icon">⚖️</div>
                <div class="logo-text">Parenti & Associés</div>
                <div class="logo-subtitle">Assistant Juridique IA</div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Section conversations
        render_chat_sidebar()

    # Navigation en haut - style pills centré
    is_chat_active = st.session_state.current_page == PAGE_CHAT
    is_docs_active = st.session_state.current_page == PAGE_DOCUMENTS

    # Container centré pour les boutons
    col_left, col_nav, col_right = st.columns([2, 3, 2])

    with col_nav:
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button(
                "💬 Chat",
                key="nav_chat",
                use_container_width=True,
                type="primary" if is_chat_active else "secondary"
            ):
                st.session_state.current_page = PAGE_CHAT
                st.rerun()
        with btn_col2:
            if st.button(
                "📁 Documents",
                key="nav_docs",
                use_container_width=True,
                type="primary" if is_docs_active else "secondary"
            ):
                st.session_state.current_page = PAGE_DOCUMENTS
                st.rerun()

    return st.session_state.current_page


def main() -> None:
    """Fonction principale de l'application."""
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    setup_logging()
    inject_custom_css()

    current_page = render_navigation()

    if current_page == PAGE_CHAT:
        render_chat_page()
    elif current_page == PAGE_DOCUMENTS:
        render_documents_page()


if __name__ == "__main__":
    main()
