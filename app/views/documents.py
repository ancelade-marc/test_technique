"""
Interface de gestion des documents.

Permet l'upload, la visualisation et la suppression
des documents de la base RAG.
"""

import streamlit as st
from datetime import datetime

from app.services.file_handler import FileHandler
from app.config import get_settings
from app.utils.logger import get_logger


logger = get_logger("documents_page")


def initialize_services() -> None:
    """Initialise les services nécessaires."""
    try:
        if "file_handler" not in st.session_state:
            st.session_state.file_handler = FileHandler()

        if "vectorstore_manager" not in st.session_state:
            st.session_state.vectorstore_manager = None

        if "document_processor" not in st.session_state:
            st.session_state.document_processor = None

    except Exception as e:
        st.error(f"Erreur lors de l'initialisation: {str(e)}")
        logger.error(f"Erreur initialisation: {str(e)}")


def get_vectorstore():
    """Récupère ou initialise le vectorstore manager."""
    if st.session_state.vectorstore_manager is None:
        from app.core.vectorstore import VectorStoreManager
        st.session_state.vectorstore_manager = VectorStoreManager()
    return st.session_state.vectorstore_manager


def get_document_processor():
    """Récupère ou initialise le document processor."""
    if st.session_state.document_processor is None:
        from app.services.document_processor import DocumentProcessor
        st.session_state.document_processor = DocumentProcessor()
    return st.session_state.document_processor


def format_file_size(size_bytes: int) -> str:
    """Formate une taille en octets de manière lisible."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} Ko"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} Mo"


def format_date(dt: datetime) -> str:
    """Formate une date de manière lisible."""
    if dt is None:
        return "N/A"
    return dt.strftime("%d/%m/%Y à %H:%M")


def render_page_header() -> None:
    """Affiche l'en-tête de la page."""
    st.markdown("""
        <div class="chat-header">
            <div class="chat-avatar" style="background: #059669;">📁</div>
            <div class="chat-info">
                <h3>Documents</h3>
                <p>Gérez la base de connaissances</p>
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_stats_cards(files: list, chunks_count: int) -> None:
    """Affiche les cartes de statistiques."""
    total_size = sum(f.size for f in files) if files else 0

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{len(files)}</div>
                <div class="stat-label">Documents</div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{format_file_size(total_size)}</div>
                <div class="stat-label">Taille totale</div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{chunks_count}</div>
                <div class="stat-label">Segments indexés</div>
            </div>
        """, unsafe_allow_html=True)


def render_upload_section() -> None:
    """Affiche la section d'upload de documents."""
    st.markdown("""
        <div class="section-header">
            <div class="section-icon">📤</div>
            <div class="section-title">Ajouter</div>
        </div>
    """, unsafe_allow_html=True)

    settings = get_settings()
    allowed_ext = ", ".join(settings.allowed_extensions)

    uploaded_files = st.file_uploader(
        f"Formats : {allowed_ext} (max {settings.max_file_size_mb} Mo)",
        type=[ext.replace(".", "") for ext in settings.allowed_extensions],
        accept_multiple_files=True,
        key="file_uploader",
        help="Glissez-déposez vos fichiers"
    )

    if uploaded_files:
        st.success(f"{len(uploaded_files)} fichier(s) sélectionné(s)")

        if st.button("Traiter et indexer", type="primary", use_container_width=True):
            process_uploaded_files(uploaded_files)


def process_uploaded_files(uploaded_files: list) -> None:
    """Traite les fichiers uploadés."""
    file_handler: FileHandler = st.session_state.file_handler

    try:
        doc_processor = get_document_processor()
    except Exception as e:
        st.error(f"❌ Erreur d'initialisation du processeur: {str(e)}")
        return

    progress_bar = st.progress(0)
    status_container = st.empty()

    results = []
    total = len(uploaded_files)

    for i, uploaded_file in enumerate(uploaded_files):
        status_container.info(f"⏳ Traitement de **{uploaded_file.name}**...")

        try:
            is_valid, error_msg = file_handler.validate_file(
                uploaded_file.name,
                uploaded_file.size
            )

            if not is_valid:
                results.append({
                    "name": uploaded_file.name,
                    "success": False,
                    "message": error_msg
                })
                continue

            file_info = file_handler.save_file(
                uploaded_file,
                uploaded_file.name
            )

            result = doc_processor.process_file(file_info)

            results.append({
                "name": uploaded_file.name,
                "success": result.success,
                "message": result.message
            })

        except ValueError as e:
            results.append({
                "name": uploaded_file.name,
                "success": False,
                "message": str(e)
            })

        except Exception as e:
            logger.error(f"Erreur traitement {uploaded_file.name}: {str(e)}")
            results.append({
                "name": uploaded_file.name,
                "success": False,
                "message": f"Erreur: {str(e)}"
            })

        progress_bar.progress((i + 1) / total)

    status_container.empty()
    progress_bar.empty()

    display_processing_results(results)


def display_processing_results(results: list) -> None:
    """Affiche les résultats du traitement."""
    success_count = sum(1 for r in results if r["success"])
    error_count = len(results) - success_count

    if success_count > 0:
        st.success(f"✅ {success_count} fichier(s) traité(s) avec succès")

    if error_count > 0:
        st.error(f"❌ {error_count} fichier(s) en erreur")

    with st.expander("📋 Détail du traitement", expanded=True):
        for result in results:
            if result["success"]:
                st.markdown(f"✅ **{result['name']}** - {result['message']}")
            else:
                st.markdown(f"❌ **{result['name']}** - {result['message']}")

    st.rerun()


def render_documents_list() -> None:
    """Affiche la liste des documents indexés."""
    st.markdown("""
        <div class="section-header">
            <div class="section-icon">📚</div>
            <div class="section-title">Indexés</div>
        </div>
    """, unsafe_allow_html=True)

    file_handler: FileHandler = st.session_state.file_handler
    files = file_handler.list_files()

    indexed_sources = []
    chunks_count = 0
    settings = get_settings()

    if settings.openai_api_key:
        try:
            vectorstore = get_vectorstore()
            indexed_sources = vectorstore.get_all_sources()
            chunks_count = vectorstore.get_document_count()
        except Exception as e:
            logger.warning(f"Impossible de récupérer les stats: {str(e)}")

    # Stats cards
    render_stats_cards(files, chunks_count)

    st.markdown("<br>", unsafe_allow_html=True)

    if not files:
        st.markdown("""
            <div class="welcome-container" style="padding: 2rem;">
                <div class="welcome-icon">📭</div>
                <div class="welcome-title">Aucun document</div>
                <div class="welcome-subtitle">
                    Uploadez vos premiers fichiers ci-dessus pour commencer à utiliser l'assistant.
                </div>
            </div>
        """, unsafe_allow_html=True)
        return

    # Liste des documents
    for file_info in files:
        is_indexed = file_info.name in indexed_sources
        icon = "✅" if is_indexed else "⏳"
        file_icon = "📄" if file_info.extension == ".txt" else ("📊" if file_info.extension == ".csv" else "🌐")

        col1, col2, col3, col4 = st.columns([5, 2, 2, 1])

        with col1:
            st.markdown(f"""
                <div style="display: flex; align-items: center; gap: 0.75rem;">
                    <span style="font-size: 1.5rem;">{file_icon}</span>
                    <div>
                        <div style="font-weight: 600; color: #1e293b;">{icon} {file_info.name}</div>
                        <div style="font-size: 0.8rem; color: #64748b;">{file_info.extension.upper()}</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
                <div style="text-align: center; padding-top: 0.5rem;">
                    <div style="font-weight: 600; color: #1e293b;">{format_file_size(file_info.size)}</div>
                    <div style="font-size: 0.75rem; color: #94a3b8;">Taille</div>
                </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
                <div style="text-align: center; padding-top: 0.5rem;">
                    <div style="font-size: 0.85rem; color: #64748b;">{format_date(file_info.uploaded_at)}</div>
                </div>
            """, unsafe_allow_html=True)

        with col4:
            if st.button("🗑️", key=f"delete_{file_info.name}", help="Supprimer ce document"):
                delete_document(file_info.name)

        st.markdown("<hr style='margin: 0.75rem 0; border: none; height: 1px; background: #e2e8f0;'>", unsafe_allow_html=True)


def delete_document(filename: str) -> None:
    """Supprime un document de la base."""
    try:
        doc_processor = get_document_processor()
        success = doc_processor.remove_document(filename)

        if success:
            st.toast(f"Document supprimé", icon="🗑️")
            st.rerun()
        else:
            st.toast("Erreur lors de la suppression", icon="❌")
    except Exception as e:
        st.toast(f"Erreur: {str(e)}", icon="❌")


def render_maintenance_section() -> None:
    """Affiche la section de maintenance."""
    with st.expander("⚙️ Maintenance"):
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Réindexer", use_container_width=True):
                reindex_all_documents()

        with col2:
            if st.button("Vider la base", use_container_width=True):
                clear_database()


def reindex_all_documents() -> None:
    """Réindexe tous les documents."""
    try:
        doc_processor = get_document_processor()

        with st.spinner("🔄 Réindexation en cours..."):
            results = doc_processor.reindex_all()

        success_count = sum(1 for r in results if r.success)
        st.success(f"✅ Réindexation terminée: {success_count}/{len(results)} documents")
        st.rerun()
    except Exception as e:
        st.error(f"❌ Erreur: {str(e)}")


def clear_database() -> None:
    """Vide la base vectorielle."""
    try:
        vectorstore = get_vectorstore()

        if vectorstore.clear():
            st.success("✅ Base vectorielle vidée avec succès")
            st.rerun()
        else:
            st.error("❌ Erreur lors de la suppression")
    except Exception as e:
        st.error(f"❌ Erreur: {str(e)}")


def render_documents_page() -> None:
    """Point d'entrée principal de la page documents."""
    try:
        initialize_services()

        settings = get_settings()

        # Header
        render_page_header()

        if not settings.openai_api_key:
            st.warning("Clé API OpenAI non configurée. Ajoutez OPENAI_API_KEY dans .env")

        # Section upload
        render_upload_section()

        # Liste des documents
        render_documents_list()

        # Maintenance
        render_maintenance_section()

    except Exception as e:
        st.error(f"Erreur: {str(e)}")
        logger.error(f"Erreur page documents: {str(e)}", exc_info=True)
