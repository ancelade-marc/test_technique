"""
Interface de chat pour l'assistant juridique.

Implémente l'interface conversationnelle avec support
du streaming et de l'historique des conversations.
"""

import streamlit as st
from typing import Optional

from app.config import get_settings
from app.services.conversation import ConversationManager, Conversation
from app.utils.logger import get_logger


logger = get_logger("chat_page")


def initialize_session_state() -> None:
    """Initialise les variables de session Streamlit."""
    try:
        if "conversation_manager" not in st.session_state:
            st.session_state.conversation_manager = ConversationManager()

        if "current_conversation_id" not in st.session_state:
            st.session_state.current_conversation_id = None

        if "rag_chain" not in st.session_state:
            st.session_state.rag_chain = None

        if "messages" not in st.session_state:
            st.session_state.messages = []

    except Exception as e:
        logger.error(f"Erreur initialisation session: {str(e)}")
        st.error(f"Erreur d'initialisation: {str(e)}")


def get_rag_chain():
    """Récupère ou initialise le RAG chain."""
    if st.session_state.rag_chain is None:
        from app.core.rag import RAGChain
        st.session_state.rag_chain = RAGChain()
    return st.session_state.rag_chain


def get_current_conversation() -> Optional[Conversation]:
    """Récupère la conversation courante."""
    if st.session_state.current_conversation_id:
        return st.session_state.conversation_manager.get_conversation(
            st.session_state.current_conversation_id
        )
    return None


def create_new_conversation() -> None:
    """Crée une nouvelle conversation."""
    conv = st.session_state.conversation_manager.create_conversation()
    st.session_state.current_conversation_id = conv.id
    st.session_state.messages = []


def load_conversation(conversation_id: str) -> None:
    """Charge une conversation existante."""
    conv = st.session_state.conversation_manager.get_conversation(conversation_id)
    if conv:
        st.session_state.current_conversation_id = conversation_id
        st.session_state.messages = [
            {"role": msg.role, "content": msg.content, "sources": msg.sources}
            for msg in conv.messages
        ]


def render_sidebar() -> None:
    """Affiche la sidebar avec l'historique des conversations."""
    initialize_session_state()

    st.markdown('<div class="nav-label">Conversations</div>', unsafe_allow_html=True)

    if st.button("+ Nouvelle", use_container_width=True, type="primary"):
        create_new_conversation()
        st.rerun()

    conversations = st.session_state.conversation_manager.list_conversations()

    if conversations:
        # Conteneur pour les conversations avec CSS dédié
        st.markdown('<div class="conversations-list">', unsafe_allow_html=True)

        for conv in conversations[:6]:
            is_current = conv.id == st.session_state.current_conversation_id
            title = conv.title[:20] + "..." if len(conv.title) > 20 else conv.title

            col1, col2 = st.columns([6, 1])

            with col1:
                btn_label = f"{'› ' if is_current else ''}{title}"
                if st.button(btn_label, key=f"conv_{conv.id}", use_container_width=True):
                    load_conversation(conv.id)
                    st.rerun()

            with col2:
                if st.button("🗑", key=f"del_{conv.id}", use_container_width=True):
                    st.session_state.conversation_manager.delete_conversation(conv.id)
                    if conv.id == st.session_state.current_conversation_id:
                        st.session_state.current_conversation_id = None
                        st.session_state.messages = []
                    st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.caption("Aucune conversation")


def render_chat_header() -> None:
    """Affiche l'en-tête du chat."""
    st.markdown("""
        <div class="chat-header">
            <div class="chat-avatar">🤖</div>
            <div class="chat-info">
                <h3>Assistant Juridique IA</h3>
                <p>Posez vos questions sur les documents du cabinet</p>
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_welcome_message() -> None:
    """Affiche le message de bienvenue."""
    st.markdown("""
        <div class="welcome-container">
            <div class="welcome-icon">⚖️</div>
            <div class="welcome-title">Assistant Juridique</div>
            <div class="welcome-subtitle">
                Posez vos questions sur les documents du cabinet.
            </div>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
            <div class="stat-card">
                <div style="font-size: 1.5rem; margin-bottom: 0.35rem;">📄</div>
                <div class="stat-label">Recherche</div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class="stat-card">
                <div style="font-size: 1.5rem; margin-bottom: 0.35rem;">⚡</div>
                <div class="stat-label">Instantané</div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div class="stat-card">
                <div style="font-size: 1.5rem; margin-bottom: 0.35rem;">🔒</div>
                <div class="stat-label">Sécurisé</div>
            </div>
        """, unsafe_allow_html=True)


def render_chat_messages() -> None:
    """Affiche les messages de la conversation."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🤖"):
            st.markdown(message["content"])

            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("📚 Sources consultées", expanded=False):
                    for source in message["sources"]:
                        st.markdown(f"• `{source}`")


def process_user_input(user_input: str) -> None:
    """Traite la question de l'utilisateur et génère une réponse."""
    if not st.session_state.current_conversation_id:
        create_new_conversation()

    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "sources": []
    })

    st.session_state.conversation_manager.add_message(
        st.session_state.current_conversation_id,
        "user",
        user_input
    )

    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="🤖"):
        try:
            rag_chain = get_rag_chain()

            if not rag_chain.is_ready():
                response = """
                **Aucun document indexé** 📭

                Pour commencer à utiliser l'assistant, veuillez d'abord :
                1. Aller dans l'onglet **Documents**
                2. Uploader vos fichiers juridiques
                3. Revenir ici pour poser vos questions
                """
                st.markdown(response)
                sources = []
            else:
                with st.spinner("🔍 Recherche dans les documents..."):
                    result = rag_chain.query(user_input)

                response = result.answer
                sources = [
                    doc.metadata.get("source_id", "Inconnu")
                    for doc in result.sources
                ]

                st.markdown(response)

                if result.has_context and sources:
                    with st.expander("📚 Sources consultées", expanded=False):
                        unique_sources = list(set(sources))
                        for source in unique_sources:
                            st.markdown(f"• `{source}`")

        except Exception as e:
            logger.error(f"Erreur lors de la génération: {str(e)}")
            response = f"""
            **Une erreur s'est produite** ⚠️

            Impossible de traiter votre question. Détail : `{str(e)}`

            Veuillez vérifier votre configuration et réessayer.
            """
            st.error(response)
            sources = []

    unique_sources = list(set(sources)) if sources else []
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "sources": unique_sources
    })

    st.session_state.conversation_manager.add_message(
        st.session_state.current_conversation_id,
        "assistant",
        response,
        unique_sources
    )


def render_chat_page() -> None:
    """Point d'entrée principal de la page chat."""
    try:
        initialize_session_state()

        settings = get_settings()

        if not settings.openai_api_key:
            st.markdown("""
                <div class="welcome-container" style="border-color: #f59e0b;">
                    <div class="welcome-icon">🔑</div>
                    <div class="welcome-title">Configuration requise</div>
                    <div class="welcome-subtitle">
                        La clé API OpenAI n'est pas configurée.
                        Créez un fichier <code>.env</code> à la racine du projet.
                    </div>
                </div>
            """, unsafe_allow_html=True)

            st.code("OPENAI_API_KEY=sk-votre-cle-api", language="bash")
            return

        # Header du chat
        render_chat_header()

        # Message de bienvenue ou messages existants
        if not st.session_state.messages:
            render_welcome_message()
        else:
            render_chat_messages()

        # Zone de saisie
        if user_input := st.chat_input("💬 Posez votre question juridique..."):
            process_user_input(user_input)

    except Exception as e:
        st.error(f"Erreur lors du chargement de la page: {str(e)}")
        logger.error(f"Erreur page chat: {str(e)}", exc_info=True)
