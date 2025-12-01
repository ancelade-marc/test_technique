"""
Tests unitaires pour le gestionnaire de base vectorielle.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.documents import Document

from app.core.vectorstore import VectorStoreManager


class TestVectorStoreManagerInit:
    """Tests pour l'initialisation du VectorStoreManager."""

    @patch('app.core.vectorstore.get_paths')
    @patch('app.core.vectorstore.get_settings')
    def test_init_with_default_embeddings(self, mock_settings, mock_paths):
        """Vérifie l'initialisation avec embeddings par défaut."""
        mock_paths.return_value = Mock(vectorstore=Path("/tmp/vectorstore"))
        mock_settings.return_value = Mock(retriever_k=4)

        manager = VectorStoreManager()

        assert manager.embeddings_manager is not None
        assert manager._vectorstore is None  # Lazy loading

    @patch('app.core.vectorstore.get_paths')
    @patch('app.core.vectorstore.get_settings')
    def test_init_with_custom_embeddings(self, mock_settings, mock_paths):
        """Vérifie l'initialisation avec embeddings personnalisés."""
        mock_paths.return_value = Mock(vectorstore=Path("/tmp/vectorstore"))
        mock_settings.return_value = Mock(retriever_k=4)

        custom_embeddings = Mock()
        manager = VectorStoreManager(embeddings_manager=custom_embeddings)

        assert manager.embeddings_manager == custom_embeddings

    def test_collection_name_constant(self):
        """Vérifie le nom de la collection."""
        assert VectorStoreManager.COLLECTION_NAME == "legal_documents"


class TestVectorStoreManagerOperations:
    """Tests pour les opérations du VectorStoreManager."""

    def setup_method(self):
        """Configuration avant chaque test."""
        self.mock_embeddings = Mock()
        self.mock_embeddings.get_langchain_embeddings.return_value = Mock()

        # Patcher les dépendances
        self.paths_patcher = patch('app.core.vectorstore.get_paths')
        self.settings_patcher = patch('app.core.vectorstore.get_settings')
        self.chroma_patcher = patch('app.core.vectorstore.Chroma')

        self.mock_paths = self.paths_patcher.start()
        self.mock_settings = self.settings_patcher.start()
        self.mock_chroma = self.chroma_patcher.start()

        self.mock_paths.return_value = Mock(vectorstore=Path("/tmp/vectorstore"))
        self.mock_settings.return_value = Mock(retriever_k=4)

        # Configuration du mock Chroma
        self.mock_collection = Mock()
        self.mock_chroma_instance = Mock()
        self.mock_chroma_instance._collection = self.mock_collection
        self.mock_chroma.return_value = self.mock_chroma_instance

    def teardown_method(self):
        """Nettoyage après chaque test."""
        self.paths_patcher.stop()
        self.settings_patcher.stop()
        self.chroma_patcher.stop()

    def test_vectorstore_property_creates_chroma(self):
        """Vérifie la création lazy du vectorstore."""
        manager = VectorStoreManager(embeddings_manager=self.mock_embeddings)

        # Premier accès - devrait créer l'instance
        _ = manager.vectorstore

        self.mock_chroma.assert_called_once()

    def test_vectorstore_property_is_cached(self):
        """Vérifie que le vectorstore est mis en cache."""
        manager = VectorStoreManager(embeddings_manager=self.mock_embeddings)

        # Multiples accès
        _ = manager.vectorstore
        _ = manager.vectorstore
        _ = manager.vectorstore

        # Chroma ne devrait être appelé qu'une fois
        assert self.mock_chroma.call_count == 1

    def test_add_documents(self):
        """Vérifie l'ajout de documents."""
        manager = VectorStoreManager(embeddings_manager=self.mock_embeddings)

        docs = [
            Document(page_content="Content 1", metadata={}),
            Document(page_content="Content 2", metadata={})
        ]

        self.mock_chroma_instance.add_documents.return_value = ["id1", "id2"]

        ids = manager.add_documents(docs, source_id="test.txt")

        assert len(ids) == 2
        # Vérifier que source_id a été ajouté aux métadonnées
        for doc in docs:
            assert doc.metadata.get("source_id") == "test.txt"

    def test_add_documents_without_source_id(self):
        """Vérifie l'ajout de documents sans source_id."""
        manager = VectorStoreManager(embeddings_manager=self.mock_embeddings)

        docs = [Document(page_content="Content", metadata={"existing": "value"})]
        self.mock_chroma_instance.add_documents.return_value = ["id1"]

        manager.add_documents(docs)

        # La métadonnée existante devrait être préservée
        assert docs[0].metadata.get("existing") == "value"

    def test_search(self):
        """Vérifie la recherche de documents."""
        manager = VectorStoreManager(embeddings_manager=self.mock_embeddings)

        expected_docs = [
            Document(page_content="Result 1", metadata={}),
            Document(page_content="Result 2", metadata={})
        ]
        self.mock_chroma_instance.similarity_search.return_value = expected_docs

        results = manager.search("query test", k=2)

        assert len(results) == 2
        self.mock_chroma_instance.similarity_search.assert_called_once_with(
            query="query test",
            k=2,
            filter=None
        )

    def test_search_with_default_k(self):
        """Vérifie la recherche avec k par défaut."""
        manager = VectorStoreManager(embeddings_manager=self.mock_embeddings)

        # Configurer le mock pour retourner une liste
        self.mock_chroma_instance.similarity_search.return_value = []
        self.mock_chroma_instance.similarity_search.reset_mock()
        manager.search("query")

        # Vérifie que similarity_search a été appelé avec les bons arguments
        call_args = self.mock_chroma_instance.similarity_search.call_args
        assert call_args[1]["query"] == "query"
        assert call_args[1]["k"] == 4  # Valeur par défaut de settings

    def test_search_with_filter(self):
        """Vérifie la recherche avec filtre."""
        manager = VectorStoreManager(embeddings_manager=self.mock_embeddings)

        filter_dict = {"source_id": "specific.txt"}
        # Configurer le mock pour retourner une liste
        self.mock_chroma_instance.similarity_search.return_value = []
        self.mock_chroma_instance.similarity_search.reset_mock()
        manager.search("query", filter_dict=filter_dict)

        call_args = self.mock_chroma_instance.similarity_search.call_args
        assert call_args[1]["query"] == "query"
        assert call_args[1]["filter"] == filter_dict

    def test_search_with_scores(self):
        """Vérifie la recherche avec scores."""
        manager = VectorStoreManager(embeddings_manager=self.mock_embeddings)

        doc = Document(page_content="Content", metadata={})
        self.mock_chroma_instance.similarity_search_with_score.return_value = [(doc, 0.15)]

        results = manager.search_with_scores("query", k=3)

        assert len(results) == 1
        assert results[0][1] == 0.15

    def test_delete_by_source(self):
        """Vérifie la suppression par source."""
        manager = VectorStoreManager(embeddings_manager=self.mock_embeddings)

        self.mock_collection.get.return_value = {"ids": ["id1", "id2", "id3"]}

        result = manager.delete_by_source("test.txt")

        assert result is True
        self.mock_collection.get.assert_called_once_with(
            where={"source_id": "test.txt"},
            include=[]
        )
        self.mock_collection.delete.assert_called_once_with(ids=["id1", "id2", "id3"])

    def test_delete_by_source_not_found(self):
        """Vérifie la suppression quand source non trouvée."""
        manager = VectorStoreManager(embeddings_manager=self.mock_embeddings)

        self.mock_collection.get.return_value = {"ids": []}

        result = manager.delete_by_source("nonexistent.txt")

        assert result is False
        self.mock_collection.delete.assert_not_called()

    def test_get_all_sources(self):
        """Vérifie la récupération de toutes les sources."""
        manager = VectorStoreManager(embeddings_manager=self.mock_embeddings)

        self.mock_collection.get.return_value = {
            "metadatas": [
                {"source_id": "doc1.txt"},
                {"source_id": "doc2.txt"},
                {"source_id": "doc1.txt"},  # Doublon
                {"other": "value"},  # Sans source_id
                None  # Metadata None
            ]
        }

        sources = manager.get_all_sources()

        assert len(sources) == 2
        assert "doc1.txt" in sources
        assert "doc2.txt" in sources

    def test_get_all_sources_empty(self):
        """Vérifie le cas sans sources."""
        manager = VectorStoreManager(embeddings_manager=self.mock_embeddings)

        self.mock_collection.get.return_value = {"metadatas": []}

        sources = manager.get_all_sources()

        assert sources == []

    def test_get_document_count(self):
        """Vérifie le comptage des documents."""
        manager = VectorStoreManager(embeddings_manager=self.mock_embeddings)

        self.mock_collection.count.return_value = 42

        count = manager.get_document_count()

        assert count == 42

    def test_get_document_count_on_error(self):
        """Vérifie le comptage en cas d'erreur."""
        manager = VectorStoreManager(embeddings_manager=self.mock_embeddings)

        self.mock_collection.count.side_effect = Exception("DB Error")

        count = manager.get_document_count()

        assert count == 0

    def test_get_retriever(self):
        """Vérifie la création d'un retriever."""
        manager = VectorStoreManager(embeddings_manager=self.mock_embeddings)

        mock_retriever = Mock()
        self.mock_chroma_instance.as_retriever.return_value = mock_retriever

        retriever = manager.get_retriever(k=5)

        self.mock_chroma_instance.as_retriever.assert_called_once_with(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

    def test_get_retriever_default_k(self):
        """Vérifie le retriever avec k par défaut."""
        manager = VectorStoreManager(embeddings_manager=self.mock_embeddings)

        manager.get_retriever()

        self.mock_chroma_instance.as_retriever.assert_called_once_with(
            search_type="similarity",
            search_kwargs={"k": 4}  # Valeur par défaut
        )

    def test_clear(self):
        """Vérifie la suppression totale."""
        manager = VectorStoreManager(embeddings_manager=self.mock_embeddings)

        self.mock_collection.get.return_value = {"ids": ["id1", "id2"]}

        result = manager.clear()

        assert result is True
        self.mock_collection.delete.assert_called_once_with(ids=["id1", "id2"])

    def test_clear_empty_collection(self):
        """Vérifie clear sur collection vide."""
        manager = VectorStoreManager(embeddings_manager=self.mock_embeddings)

        self.mock_collection.get.return_value = {"ids": []}

        result = manager.clear()

        assert result is True
        self.mock_collection.delete.assert_not_called()

    def test_clear_on_error(self):
        """Vérifie clear en cas d'erreur."""
        manager = VectorStoreManager(embeddings_manager=self.mock_embeddings)

        self.mock_collection.get.side_effect = Exception("DB Error")

        result = manager.clear()

        assert result is False


class TestVectorStoreManagerErrors:
    """Tests pour la gestion des erreurs."""

    def setup_method(self):
        """Configuration avant chaque test."""
        self.mock_embeddings = Mock()
        self.mock_embeddings.get_langchain_embeddings.return_value = Mock()

        self.paths_patcher = patch('app.core.vectorstore.get_paths')
        self.settings_patcher = patch('app.core.vectorstore.get_settings')
        self.chroma_patcher = patch('app.core.vectorstore.Chroma')

        self.mock_paths = self.paths_patcher.start()
        self.mock_settings = self.settings_patcher.start()
        self.mock_chroma = self.chroma_patcher.start()

        self.mock_paths.return_value = Mock(vectorstore=Path("/tmp/vectorstore"))
        self.mock_settings.return_value = Mock(retriever_k=4)

        self.mock_collection = Mock()
        self.mock_chroma_instance = Mock()
        self.mock_chroma_instance._collection = self.mock_collection
        self.mock_chroma.return_value = self.mock_chroma_instance

    def teardown_method(self):
        """Nettoyage après chaque test."""
        self.paths_patcher.stop()
        self.settings_patcher.stop()
        self.chroma_patcher.stop()

    def test_add_documents_raises_on_error(self):
        """Vérifie la propagation des erreurs à l'ajout."""
        manager = VectorStoreManager(embeddings_manager=self.mock_embeddings)

        self.mock_chroma_instance.add_documents.side_effect = Exception("Add Error")

        with pytest.raises(Exception) as exc_info:
            manager.add_documents([Document(page_content="Test", metadata={})])

        assert "Add Error" in str(exc_info.value)

    def test_search_raises_on_error(self):
        """Vérifie la propagation des erreurs à la recherche."""
        manager = VectorStoreManager(embeddings_manager=self.mock_embeddings)

        self.mock_chroma_instance.similarity_search.side_effect = Exception("Search Error")

        with pytest.raises(Exception) as exc_info:
            manager.search("query")

        assert "Search Error" in str(exc_info.value)

    def test_search_with_scores_raises_on_error(self):
        """Vérifie la propagation des erreurs à la recherche avec scores."""
        manager = VectorStoreManager(embeddings_manager=self.mock_embeddings)

        self.mock_chroma_instance.similarity_search_with_score.side_effect = Exception("Score Error")

        with pytest.raises(Exception) as exc_info:
            manager.search_with_scores("query")

        assert "Score Error" in str(exc_info.value)

    def test_delete_by_source_raises_on_error(self):
        """Vérifie la propagation des erreurs à la suppression."""
        manager = VectorStoreManager(embeddings_manager=self.mock_embeddings)

        self.mock_collection.get.side_effect = Exception("Delete Error")

        with pytest.raises(Exception) as exc_info:
            manager.delete_by_source("test.txt")

        assert "Delete Error" in str(exc_info.value)

    def test_get_all_sources_returns_empty_on_error(self):
        """Vérifie le retour vide en cas d'erreur get_all_sources."""
        manager = VectorStoreManager(embeddings_manager=self.mock_embeddings)

        self.mock_collection.get.side_effect = Exception("Error")

        sources = manager.get_all_sources()

        assert sources == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
