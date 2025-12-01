"""
Tests unitaires pour le module RAG (Retrieval-Augmented Generation).
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.documents import Document

from app.core.rag import RAGChain, RAGResponse
from app.config import NO_CONTEXT_MESSAGE


class TestRAGResponse:
    """Tests pour la dataclass RAGResponse."""

    def test_rag_response_creation(self):
        """Vérifie la création d'une RAGResponse."""
        docs = [Document(page_content="Test content", metadata={"source_id": "test.txt"})]
        response = RAGResponse(
            answer="Test answer",
            sources=docs,
            has_context=True
        )
        assert response.answer == "Test answer"
        assert len(response.sources) == 1
        assert response.has_context is True

    def test_rag_response_without_context(self):
        """Vérifie la création d'une RAGResponse sans contexte."""
        response = RAGResponse(
            answer=NO_CONTEXT_MESSAGE,
            sources=[],
            has_context=False
        )
        assert response.has_context is False
        assert len(response.sources) == 0


class TestRAGChain:
    """Tests pour la classe RAGChain."""

    def setup_method(self):
        """Configuration avant chaque test."""
        self.mock_vectorstore = Mock()
        self.mock_llm = Mock()

        # Configuration du mock vectorstore
        self.mock_vectorstore.get_document_count.return_value = 5
        self.mock_vectorstore.search_with_scores.return_value = []

        # Configuration du mock LLM
        self.mock_llm.invoke.return_value = "Réponse générée"

        self.rag_chain = RAGChain(
            vectorstore_manager=self.mock_vectorstore,
            llm_client=self.mock_llm
        )

    def test_init_with_custom_dependencies(self):
        """Vérifie l'initialisation avec des dépendances personnalisées."""
        rag = RAGChain(
            vectorstore_manager=self.mock_vectorstore,
            llm_client=self.mock_llm
        )
        assert rag.vectorstore == self.mock_vectorstore
        assert rag.llm == self.mock_llm

    def test_is_ready_with_documents(self):
        """Vérifie is_ready() quand des documents sont indexés."""
        self.mock_vectorstore.get_document_count.return_value = 10
        assert self.rag_chain.is_ready() is True

    def test_is_ready_without_documents(self):
        """Vérifie is_ready() quand aucun document n'est indexé."""
        self.mock_vectorstore.get_document_count.return_value = 0
        assert self.rag_chain.is_ready() is False

    def test_query_with_relevant_documents(self):
        """Vérifie query() avec des documents pertinents."""
        doc = Document(
            page_content="Contenu juridique pertinent",
            metadata={"source_id": "contrat.txt"}
        )
        # Score bas = plus pertinent dans ChromaDB
        self.mock_vectorstore.search_with_scores.return_value = [(doc, 0.1)]
        self.mock_llm.invoke.return_value = "Réponse basée sur le contrat"

        response = self.rag_chain.query("Qu'est-ce qu'une SAS?")

        assert response.has_context is True
        assert response.answer == "Réponse basée sur le contrat"
        assert len(response.sources) == 1
        assert response.sources[0].metadata["source_id"] == "contrat.txt"

    def test_query_without_documents(self):
        """Vérifie query() sans documents disponibles."""
        self.mock_vectorstore.search_with_scores.return_value = []

        response = self.rag_chain.query("Question sans contexte")

        assert response.has_context is False
        assert response.answer == NO_CONTEXT_MESSAGE
        assert len(response.sources) == 0

    def test_query_filters_low_relevance_documents(self):
        """Vérifie le filtrage des documents peu pertinents."""
        doc1 = Document(page_content="Pertinent", metadata={"source_id": "doc1.txt"})
        doc2 = Document(page_content="Non pertinent", metadata={"source_id": "doc2.txt"})

        # doc1 pertinent (score < 0.3), doc2 non pertinent (score > 0.3)
        self.mock_vectorstore.search_with_scores.return_value = [
            (doc1, 0.15),
            (doc2, 0.8)
        ]

        response = self.rag_chain.query("Question test")

        # Seul doc1 devrait être dans les sources
        assert len(response.sources) == 1
        assert response.sources[0].metadata["source_id"] == "doc1.txt"

    def test_query_uses_fallback_when_no_relevant_docs(self):
        """Vérifie le fallback quand aucun doc n'est suffisamment pertinent."""
        doc = Document(page_content="Peu pertinent", metadata={"source_id": "doc.txt"})

        # Score > 0.3, donc pas pertinent selon le seuil
        self.mock_vectorstore.search_with_scores.return_value = [(doc, 0.5)]

        response = self.rag_chain.query("Question test")

        # Le document devrait être utilisé comme fallback
        assert response.has_context is True
        assert len(response.sources) == 1

    def test_query_with_custom_k(self):
        """Vérifie query() avec un k personnalisé."""
        docs = [
            (Document(page_content=f"Doc {i}", metadata={"source_id": f"doc{i}.txt"}), 0.1)
            for i in range(10)
        ]
        self.mock_vectorstore.search_with_scores.return_value = docs

        response = self.rag_chain.query("Question", k=3)

        # Vérifie que search_with_scores a été appelé avec k=3
        self.mock_vectorstore.search_with_scores.assert_called_once()
        call_args = self.mock_vectorstore.search_with_scores.call_args
        assert call_args[1]["k"] == 3

    def test_build_context_formats_correctly(self):
        """Vérifie le formatage du contexte."""
        docs = [
            Document(page_content="Contenu 1", metadata={"source_id": "source1.txt"}),
            Document(page_content="Contenu 2", metadata={"source_id": "source2.txt"})
        ]

        context = self.rag_chain._build_context(docs)

        assert "[Document 1 - Source: source1.txt]" in context
        assert "[Document 2 - Source: source2.txt]" in context
        assert "Contenu 1" in context
        assert "Contenu 2" in context
        assert "---" in context

    def test_build_context_handles_missing_source_id(self):
        """Vérifie le contexte avec source_id manquant."""
        docs = [Document(page_content="Contenu", metadata={})]

        context = self.rag_chain._build_context(docs)

        assert "[Document 1 - Source: Document inconnu]" in context

    def test_get_sources_summary(self):
        """Vérifie le résumé des sources."""
        docs = [
            Document(page_content="Contenu court", metadata={"source_id": "doc1.txt"}),
            Document(page_content="A" * 300, metadata={"source_id": "doc2.txt"})
        ]

        summary = self.rag_chain.get_sources_summary(docs)

        assert len(summary) == 2
        assert summary[0]["name"] == "doc1.txt"
        assert summary[0]["preview"] == "Contenu court"
        # Le deuxième devrait être tronqué
        assert len(summary[1]["preview"]) <= 203  # 200 + "..."

    def test_get_sources_summary_deduplicates(self):
        """Vérifie la déduplication des sources."""
        docs = [
            Document(page_content="Chunk 1", metadata={"source_id": "doc.txt"}),
            Document(page_content="Chunk 2", metadata={"source_id": "doc.txt"})
        ]

        summary = self.rag_chain.get_sources_summary(docs)

        # Une seule source malgré deux documents du même fichier
        assert len(summary) == 1

    def test_query_raises_on_error(self):
        """Vérifie la propagation des erreurs."""
        self.mock_vectorstore.search_with_scores.side_effect = Exception("Erreur DB")

        with pytest.raises(Exception) as exc_info:
            self.rag_chain.query("Question")

        assert "Erreur DB" in str(exc_info.value)


class TestRAGChainStreaming:
    """Tests pour le streaming RAG."""

    def setup_method(self):
        """Configuration avant chaque test."""
        self.mock_vectorstore = Mock()
        self.mock_llm = Mock()

        self.rag_chain = RAGChain(
            vectorstore_manager=self.mock_vectorstore,
            llm_client=self.mock_llm
        )

    def test_query_stream_yields_tokens(self):
        """Vérifie que query_stream yield les tokens."""
        doc = Document(page_content="Contenu", metadata={"source_id": "doc.txt"})
        self.mock_vectorstore.search_with_scores.return_value = [(doc, 0.1)]
        self.mock_llm.stream.return_value = iter(["Token1", "Token2", "Token3"])

        tokens = list(self.rag_chain.query_stream("Question"))

        assert tokens == ["Token1", "Token2", "Token3"]

    def test_query_stream_without_context(self):
        """Vérifie le streaming sans contexte."""
        self.mock_vectorstore.search_with_scores.return_value = []

        tokens = list(self.rag_chain.query_stream("Question"))

        assert tokens == [NO_CONTEXT_MESSAGE]


class TestRAGChainMinRelevanceScore:
    """Tests pour le seuil de pertinence."""

    def test_min_relevance_score_constant(self):
        """Vérifie la valeur du seuil de pertinence."""
        assert RAGChain.MIN_RELEVANCE_SCORE == 0.3

    def test_documents_at_threshold_are_included(self):
        """Vérifie que les documents au seuil exact sont inclus."""
        mock_vectorstore = Mock()
        mock_llm = Mock()

        doc = Document(page_content="Test", metadata={"source_id": "test.txt"})
        mock_vectorstore.search_with_scores.return_value = [(doc, 0.3)]
        mock_llm.invoke.return_value = "Réponse"

        rag = RAGChain(vectorstore_manager=mock_vectorstore, llm_client=mock_llm)
        response = rag.query("Question")

        # Score == 0.3 devrait être inclus (<=)
        assert response.has_context is True
        assert len(response.sources) == 1


class TestRAGChainStreamingEdgeCases:
    """Tests pour les cas limites du streaming."""

    def setup_method(self):
        """Configuration avant chaque test."""
        self.mock_vectorstore = Mock()
        self.mock_llm = Mock()

        self.rag_chain = RAGChain(
            vectorstore_manager=self.mock_vectorstore,
            llm_client=self.mock_llm
        )

    def test_query_stream_with_fallback_docs(self):
        """Vérifie le streaming avec fallback sur documents peu pertinents."""
        doc = Document(page_content="Contenu", metadata={"source_id": "doc.txt"})
        # Score > 0.3, donc pas pertinent selon le seuil
        self.mock_vectorstore.search_with_scores.return_value = [(doc, 0.5)]
        self.mock_llm.stream.return_value = iter(["Token"])

        tokens = list(self.rag_chain.query_stream("Question"))

        assert tokens == ["Token"]

    def test_query_stream_raises_on_error(self):
        """Vérifie la propagation des erreurs en streaming."""
        self.mock_vectorstore.search_with_scores.side_effect = Exception("Stream Error")

        with pytest.raises(Exception) as exc_info:
            list(self.rag_chain.query_stream("Question"))

        assert "Stream Error" in str(exc_info.value)


class TestRAGChainDefaultInit:
    """Tests pour l'initialisation par défaut."""

    @patch('app.core.rag.VectorStoreManager')
    @patch('app.core.rag.LLMClient')
    @patch('app.core.rag.get_settings')
    def test_init_creates_default_dependencies(self, mock_settings, mock_llm, mock_vectorstore):
        """Vérifie la création des dépendances par défaut."""
        mock_settings.return_value = Mock(retriever_k=4)

        rag = RAGChain()

        # Les dépendances par défaut devraient être créées
        mock_vectorstore.assert_called_once()
        mock_llm.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
