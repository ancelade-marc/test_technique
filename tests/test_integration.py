"""
Tests d'intégration pour le pipeline RAG complet.

Ces tests vérifient le bon fonctionnement des composants ensemble,
en utilisant des mocks pour les appels API externes (OpenAI).
"""

import pytest
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from io import BytesIO

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.documents import Document

from app.core.rag import RAGChain, RAGResponse
from app.core.vectorstore import VectorStoreManager
from app.core.llm import LLMClient
from app.core.embeddings import EmbeddingsManager
from app.services.document_processor import DocumentProcessor, ProcessingResult
from app.services.file_handler import FileHandler, FileInfo
from app.services.conversation import ConversationManager
from app.config import NO_CONTEXT_MESSAGE


class TestRAGPipelineIntegration:
    """Tests d'intégration pour le pipeline RAG complet."""

    def setup_method(self):
        """Configuration avant chaque test."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Nettoyage après chaque test."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_file_upload_to_query_flow(self):
        """Test du flux complet: upload → processing → query."""
        # 1. Setup des mocks
        mock_vectorstore = Mock()
        mock_llm = Mock()
        mock_embeddings = Mock()

        # Mock des réponses
        doc = Document(
            page_content="Une SAS est une Société par Actions Simplifiée.",
            metadata={"source_id": "juridique.txt"}
        )
        mock_vectorstore.search_with_scores.return_value = [(doc, 0.1)]
        mock_vectorstore.add_documents.return_value = ["id1"]
        mock_vectorstore.get_document_count.return_value = 1
        mock_llm.invoke.return_value = "Une SAS est une forme juridique de société."

        # 2. Créer le RAGChain
        rag_chain = RAGChain(
            vectorstore_manager=mock_vectorstore,
            llm_client=mock_llm
        )

        # 3. Simuler une query
        response = rag_chain.query("Qu'est-ce qu'une SAS?")

        # 4. Vérifications
        assert response.has_context is True
        assert "SAS" in response.answer
        assert len(response.sources) == 1

    def test_conversation_with_rag_context(self):
        """Test de conversation avec contexte RAG."""
        # Setup mocks
        mock_vectorstore = Mock()
        mock_llm = Mock()

        doc = Document(
            page_content="Article 1: Les parties conviennent...",
            metadata={"source_id": "contrat.txt"}
        )
        mock_vectorstore.search_with_scores.return_value = [(doc, 0.15)]
        mock_vectorstore.get_document_count.return_value = 1
        mock_llm.invoke.return_value = "L'article 1 stipule que les parties..."

        # Créer les composants
        rag_chain = RAGChain(
            vectorstore_manager=mock_vectorstore,
            llm_client=mock_llm
        )

        # Simuler plusieurs interactions
        response1 = rag_chain.query("Que dit l'article 1?")
        assert response1.has_context is True

        # Simuler une deuxième question
        mock_llm.invoke.return_value = "Non, le contrat ne mentionne pas de pénalités."
        response2 = rag_chain.query("Y a-t-il des pénalités?")
        assert response2.has_context is True

    def test_no_context_handling(self):
        """Test du comportement sans contexte."""
        mock_vectorstore = Mock()
        mock_llm = Mock()

        # Pas de documents trouvés
        mock_vectorstore.search_with_scores.return_value = []
        mock_vectorstore.get_document_count.return_value = 0

        rag_chain = RAGChain(
            vectorstore_manager=mock_vectorstore,
            llm_client=mock_llm
        )

        response = rag_chain.query("Question hors contexte")

        assert response.has_context is False
        assert response.answer == NO_CONTEXT_MESSAGE
        # Le LLM ne devrait pas être appelé
        mock_llm.invoke.assert_not_called()


class TestFileHandlerIntegration:
    """Tests d'intégration pour le gestionnaire de fichiers."""

    def setup_method(self):
        """Configuration avant chaque test."""
        self.temp_dir = tempfile.mkdtemp()

        self.paths_patcher = patch('app.services.file_handler.get_paths')
        self.settings_patcher = patch('app.services.file_handler.get_settings')

        mock_paths = Mock()
        mock_paths.documents = Path(self.temp_dir) / "documents"
        mock_paths.documents.mkdir(parents=True, exist_ok=True)

        mock_settings = Mock()
        mock_settings.allowed_extensions = [".txt", ".csv", ".html"]
        mock_settings.max_file_size_mb = 10

        self.mock_paths = self.paths_patcher.start()
        self.mock_settings = self.settings_patcher.start()
        self.mock_paths.return_value = mock_paths
        self.mock_settings.return_value = mock_settings

    def teardown_method(self):
        """Nettoyage après chaque test."""
        self.paths_patcher.stop()
        self.settings_patcher.stop()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_and_list_files(self):
        """Test de sauvegarde et listage de fichiers."""
        handler = FileHandler()

        # Sauvegarder un fichier
        content = BytesIO(b"Contenu du document juridique")
        file_info = handler.save_file(content, "document.txt")

        assert file_info.name == "document.txt"
        assert file_info.size == len(b"Contenu du document juridique")

        # Lister les fichiers
        files = handler.list_files()
        assert len(files) == 1
        assert files[0].name == "document.txt"

    def test_save_and_delete_file(self):
        """Test de sauvegarde et suppression."""
        handler = FileHandler()

        # Sauvegarder
        content = BytesIO(b"Contenu temporaire")
        file_info = handler.save_file(content, "temp.txt")

        # Supprimer
        result = handler.delete_file(file_info.name)
        assert result is True

        # Vérifier la suppression
        files = handler.list_files()
        assert len(files) == 0

    def test_duplicate_detection(self):
        """Test de détection des doublons."""
        handler = FileHandler()

        # Premier fichier
        content1 = BytesIO(b"Contenu identique")
        handler.save_file(content1, "doc1.txt")

        # Tentative de doublon
        content2 = BytesIO(b"Contenu identique")
        with pytest.raises(ValueError) as exc_info:
            handler.save_file(content2, "doc2.txt")

        assert "identique" in str(exc_info.value)

    def test_get_file_content(self):
        """Test de récupération du contenu."""
        handler = FileHandler()

        # Sauvegarder
        original_content = "Texte juridique français avec accents: été, café"
        content = BytesIO(original_content.encode('utf-8'))
        file_info = handler.save_file(content, "french.txt")

        # Récupérer
        retrieved = handler.get_file_content(file_info.name)
        assert retrieved == original_content


class TestConversationManagerIntegration:
    """Tests d'intégration pour le gestionnaire de conversations."""

    def setup_method(self):
        """Configuration avant chaque test."""
        self.temp_dir = tempfile.mkdtemp()

        self.paths_patcher = patch('app.services.conversation.get_paths')
        mock_paths = Mock()
        mock_paths.conversations = Path(self.temp_dir) / "conversations"
        mock_paths.conversations.mkdir(parents=True, exist_ok=True)

        self.mock_paths = self.paths_patcher.start()
        self.mock_paths.return_value = mock_paths

    def teardown_method(self):
        """Nettoyage après chaque test."""
        self.paths_patcher.stop()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_full_conversation_lifecycle(self):
        """Test du cycle de vie complet d'une conversation."""
        manager = ConversationManager()

        # Créer
        conv = manager.create_conversation()
        assert conv.id is not None

        # Ajouter des messages
        manager.add_message(conv.id, "user", "Bonjour")
        manager.add_message(conv.id, "assistant", "Bonjour! Comment puis-je vous aider?")
        manager.add_message(conv.id, "user", "Qu'est-ce qu'une SARL?")
        manager.add_message(
            conv.id,
            "assistant",
            "Une SARL est une Société à Responsabilité Limitée...",
            sources=["guide_societes.txt"]
        )

        # Récupérer le contexte
        messages = manager.get_messages_for_context(conv.id)
        assert len(messages) == 4

        # Lister les conversations
        all_convs = manager.list_conversations()
        assert len(all_convs) == 1

        # Supprimer
        manager.delete_conversation(conv.id)
        assert manager.get_conversation(conv.id) is None

    def test_persistence_across_instances(self):
        """Test de la persistance entre instances."""
        # Première instance
        manager1 = ConversationManager()
        conv = manager1.create_conversation()
        manager1.add_message(conv.id, "user", "Question persistante")

        # Nouvelle instance (simule redémarrage)
        manager2 = ConversationManager()

        # Vérifier la persistance
        retrieved = manager2.get_conversation(conv.id)
        assert retrieved is not None
        assert len(retrieved.messages) == 1
        assert retrieved.messages[0].content == "Question persistante"


class TestDocumentProcessorIntegration:
    """Tests d'intégration pour le processeur de documents."""

    def setup_method(self):
        """Configuration avant chaque test."""
        self.temp_dir = tempfile.mkdtemp()

        # Setup mocks
        self.mock_vectorstore = Mock()
        self.mock_vectorstore.add_documents.return_value = ["id1", "id2"]
        self.mock_vectorstore.get_all_sources.return_value = []

        self.mock_file_handler = Mock()

    def teardown_method(self):
        """Nettoyage après chaque test."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('app.services.document_processor.get_settings')
    def test_process_text_file(self, mock_settings):
        """Test du traitement d'un fichier texte."""
        mock_settings.return_value = Mock(chunk_size=200, chunk_overlap=50)

        self.mock_file_handler.get_file_content.return_value = (
            "Texte juridique avec des informations importantes. "
            "Ce document contient des clauses contractuelles. "
            "Les parties sont Jean Dupont et Marie Martin."
        )

        processor = DocumentProcessor(
            vectorstore=self.mock_vectorstore,
            file_handler=self.mock_file_handler
        )

        file_info = FileInfo(
            name="contrat.txt",
            path=Path(self.temp_dir) / "contrat.txt",
            extension=".txt",
            size=500,
            hash="abc123",
            uploaded_at=datetime.now()
        )

        result = processor.process_file(file_info)

        assert result.success is True
        assert result.chunks_count >= 1
        self.mock_vectorstore.add_documents.assert_called_once()


class TestEndToEndScenarios:
    """Tests de scénarios end-to-end."""

    def test_legal_document_qa_scenario(self):
        """Scénario: Q&A sur documents juridiques."""
        # Setup mocks
        mock_vectorstore = Mock()
        mock_llm = Mock()

        # Documents simulés
        contract_doc = Document(
            page_content="Article 5: Le prestataire s'engage à fournir les services décrits en annexe A.",
            metadata={"source_id": "contrat_prestation.txt", "chunk_index": 0}
        )
        annex_doc = Document(
            page_content="Annexe A: Services de conseil juridique en droit des affaires.",
            metadata={"source_id": "annexe_a.txt", "chunk_index": 0}
        )

        # Simuler la recherche
        mock_vectorstore.search_with_scores.return_value = [
            (contract_doc, 0.1),
            (annex_doc, 0.15)
        ]
        mock_vectorstore.get_document_count.return_value = 2

        # Réponse du LLM
        mock_llm.invoke.return_value = (
            "Selon l'Article 5 du contrat de prestation, le prestataire s'engage à fournir "
            "les services décrits en Annexe A, qui comprennent des services de conseil "
            "juridique en droit des affaires."
        )

        # Exécuter le scénario
        rag_chain = RAGChain(
            vectorstore_manager=mock_vectorstore,
            llm_client=mock_llm
        )

        response = rag_chain.query("Quels services sont fournis par le prestataire?")

        # Vérifications
        assert response.has_context is True
        assert len(response.sources) == 2
        assert "Article 5" in response.answer or "services" in response.answer.lower()

        # Vérifier les sources
        sources_summary = rag_chain.get_sources_summary(response.sources)
        source_names = [s["name"] for s in sources_summary]
        assert "contrat_prestation.txt" in source_names
        assert "annexe_a.txt" in source_names

    def test_multi_turn_conversation_scenario(self):
        """Scénario: Conversation multi-tours."""
        mock_vectorstore = Mock()
        mock_llm = Mock()

        # Premier tour
        doc1 = Document(
            page_content="Une SAS nécessite un capital social minimum de 1 euro.",
            metadata={"source_id": "guide_sas.txt"}
        )
        mock_vectorstore.search_with_scores.return_value = [(doc1, 0.1)]
        mock_vectorstore.get_document_count.return_value = 1
        mock_llm.invoke.return_value = "Le capital social minimum d'une SAS est de 1 euro."

        rag = RAGChain(vectorstore_manager=mock_vectorstore, llm_client=mock_llm)
        response1 = rag.query("Quel est le capital minimum d'une SAS?")

        assert "1 euro" in response1.answer

        # Deuxième tour
        doc2 = Document(
            page_content="La SAS peut avoir un ou plusieurs présidents.",
            metadata={"source_id": "guide_sas.txt"}
        )
        mock_vectorstore.search_with_scores.return_value = [(doc2, 0.12)]
        mock_llm.invoke.return_value = "Une SAS peut avoir un ou plusieurs présidents."

        response2 = rag.query("Et pour la direction?")

        assert response2.has_context is True
        assert "président" in response2.answer.lower()


class TestErrorRecovery:
    """Tests de récupération d'erreurs."""

    def test_rag_recovers_from_empty_vectorstore(self):
        """Test de récupération avec vectorstore vide."""
        mock_vectorstore = Mock()
        mock_llm = Mock()

        mock_vectorstore.search_with_scores.return_value = []
        mock_vectorstore.get_document_count.return_value = 0

        rag = RAGChain(vectorstore_manager=mock_vectorstore, llm_client=mock_llm)

        # Ne devrait pas lever d'exception
        response = rag.query("Question quelconque")

        assert response.has_context is False
        assert response.answer == NO_CONTEXT_MESSAGE

    def test_graceful_handling_of_llm_error(self):
        """Test de gestion gracieuse des erreurs LLM."""
        mock_vectorstore = Mock()
        mock_llm = Mock()

        doc = Document(page_content="Content", metadata={"source_id": "doc.txt"})
        mock_vectorstore.search_with_scores.return_value = [(doc, 0.1)]
        mock_llm.invoke.side_effect = Exception("LLM API Error")

        rag = RAGChain(vectorstore_manager=mock_vectorstore, llm_client=mock_llm)

        # Devrait propager l'erreur
        with pytest.raises(Exception) as exc_info:
            rag.query("Question")

        assert "LLM API Error" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
