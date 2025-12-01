"""
Tests unitaires pour le gestionnaire d'embeddings.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.embeddings import EmbeddingsManager


class TestEmbeddingsManagerInit:
    """Tests pour l'initialisation du EmbeddingsManager."""

    @patch('app.core.embeddings.get_settings')
    def test_init_with_defaults(self, mock_settings):
        """Vérifie l'initialisation avec les valeurs par défaut."""
        mock_settings.return_value = Mock(
            embedding_model="text-embedding-3-small"
        )

        manager = EmbeddingsManager()

        assert manager.model == "text-embedding-3-small"
        assert manager._embeddings is None  # Lazy loading

    @patch('app.core.embeddings.get_settings')
    def test_init_with_custom_model(self, mock_settings):
        """Vérifie l'initialisation avec un modèle personnalisé."""
        mock_settings.return_value = Mock(
            embedding_model="text-embedding-3-small"
        )

        manager = EmbeddingsManager(model="text-embedding-3-large")

        assert manager.model == "text-embedding-3-large"


class TestEmbeddingsManagerProperty:
    """Tests pour la propriété embeddings (lazy loading)."""

    @patch('app.core.embeddings.get_settings')
    def test_embeddings_raises_without_api_key(self, mock_settings):
        """Vérifie l'erreur si la clé API est manquante."""
        mock_settings.return_value = Mock(
            embedding_model="model",
            openai_api_key=""
        )

        manager = EmbeddingsManager()

        with pytest.raises(ValueError) as exc_info:
            _ = manager.embeddings

        assert "clé API OpenAI" in str(exc_info.value)

    @patch('app.core.embeddings.OpenAIEmbeddings')
    @patch('app.core.embeddings.get_settings')
    def test_embeddings_creates_openai_embeddings(self, mock_settings, mock_openai_embeddings):
        """Vérifie la création de l'instance OpenAIEmbeddings."""
        mock_settings.return_value = Mock(
            embedding_model="text-embedding-3-small",
            openai_api_key="test-key"
        )

        manager = EmbeddingsManager()
        _ = manager.embeddings

        mock_openai_embeddings.assert_called_once_with(
            model="text-embedding-3-small",
            api_key="test-key"
        )

    @patch('app.core.embeddings.OpenAIEmbeddings')
    @patch('app.core.embeddings.get_settings')
    def test_embeddings_is_cached(self, mock_settings, mock_openai_embeddings):
        """Vérifie que l'instance embeddings est mise en cache."""
        mock_settings.return_value = Mock(
            embedding_model="model",
            openai_api_key="key"
        )

        manager = EmbeddingsManager()

        # Multiples accès
        _ = manager.embeddings
        _ = manager.embeddings
        _ = manager.embeddings

        # OpenAIEmbeddings ne devrait être appelé qu'une fois
        assert mock_openai_embeddings.call_count == 1


class TestEmbeddingsManagerMethods:
    """Tests pour les méthodes du EmbeddingsManager."""

    def setup_method(self):
        """Configuration avant chaque test."""
        self.settings_patcher = patch('app.core.embeddings.get_settings')
        self.openai_patcher = patch('app.core.embeddings.OpenAIEmbeddings')

        self.mock_settings = self.settings_patcher.start()
        self.mock_openai_embeddings = self.openai_patcher.start()

        self.mock_settings.return_value = Mock(
            embedding_model="model",
            openai_api_key="key"
        )

        self.mock_embeddings_instance = Mock()
        self.mock_openai_embeddings.return_value = self.mock_embeddings_instance

    def teardown_method(self):
        """Nettoyage après chaque test."""
        self.settings_patcher.stop()
        self.openai_patcher.stop()

    def test_embed_text_returns_vector(self):
        """Vérifie que embed_text retourne un vecteur."""
        self.mock_embeddings_instance.embed_query.return_value = [0.1, 0.2, 0.3]

        manager = EmbeddingsManager()
        result = manager.embed_text("Test text")

        assert result == [0.1, 0.2, 0.3]
        self.mock_embeddings_instance.embed_query.assert_called_once_with("Test text")

    def test_embed_text_handles_empty_string(self):
        """Vérifie le comportement avec une chaîne vide."""
        self.mock_embeddings_instance.embed_query.return_value = [0.0, 0.0, 0.0]

        manager = EmbeddingsManager()
        result = manager.embed_text("")

        assert result == [0.0, 0.0, 0.0]

    def test_embed_text_raises_on_error(self):
        """Vérifie la propagation des erreurs."""
        self.mock_embeddings_instance.embed_query.side_effect = Exception("API Error")

        manager = EmbeddingsManager()

        with pytest.raises(Exception) as exc_info:
            manager.embed_text("Test")

        assert "API Error" in str(exc_info.value)

    def test_embed_texts_returns_vectors(self):
        """Vérifie que embed_texts retourne une liste de vecteurs."""
        self.mock_embeddings_instance.embed_documents.return_value = [
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6]
        ]

        manager = EmbeddingsManager()
        result = manager.embed_texts(["Text 1", "Text 2", "Text 3"])

        assert len(result) == 3
        assert result[0] == [0.1, 0.2]
        self.mock_embeddings_instance.embed_documents.assert_called_once_with(
            ["Text 1", "Text 2", "Text 3"]
        )

    def test_embed_texts_empty_list(self):
        """Vérifie le comportement avec une liste vide."""
        self.mock_embeddings_instance.embed_documents.return_value = []

        manager = EmbeddingsManager()
        result = manager.embed_texts([])

        assert result == []

    def test_embed_texts_raises_on_error(self):
        """Vérifie la propagation des erreurs."""
        self.mock_embeddings_instance.embed_documents.side_effect = Exception("Batch Error")

        manager = EmbeddingsManager()

        with pytest.raises(Exception) as exc_info:
            manager.embed_texts(["Text 1", "Text 2"])

        assert "Batch Error" in str(exc_info.value)

    def test_get_langchain_embeddings_returns_instance(self):
        """Vérifie que get_langchain_embeddings retourne l'instance."""
        manager = EmbeddingsManager()

        result = manager.get_langchain_embeddings()

        assert result == self.mock_embeddings_instance

    def test_get_langchain_embeddings_is_same_instance(self):
        """Vérifie que l'instance retournée est la même."""
        manager = EmbeddingsManager()

        instance1 = manager.get_langchain_embeddings()
        instance2 = manager.get_langchain_embeddings()

        assert instance1 is instance2


class TestEmbeddingsManagerWithDifferentModels:
    """Tests pour différentes configurations de modèles."""

    @patch('app.core.embeddings.OpenAIEmbeddings')
    @patch('app.core.embeddings.get_settings')
    def test_text_embedding_3_small(self, mock_settings, mock_openai):
        """Test avec text-embedding-3-small."""
        mock_settings.return_value = Mock(
            embedding_model="text-embedding-3-small",
            openai_api_key="key"
        )

        manager = EmbeddingsManager()
        _ = manager.embeddings

        mock_openai.assert_called_once_with(
            model="text-embedding-3-small",
            api_key="key"
        )

    @patch('app.core.embeddings.OpenAIEmbeddings')
    @patch('app.core.embeddings.get_settings')
    def test_text_embedding_3_large(self, mock_settings, mock_openai):
        """Test avec text-embedding-3-large."""
        mock_settings.return_value = Mock(
            embedding_model="default-model",
            openai_api_key="key"
        )

        manager = EmbeddingsManager(model="text-embedding-3-large")
        _ = manager.embeddings

        mock_openai.assert_called_once_with(
            model="text-embedding-3-large",
            api_key="key"
        )

    @patch('app.core.embeddings.OpenAIEmbeddings')
    @patch('app.core.embeddings.get_settings')
    def test_text_embedding_ada_002(self, mock_settings, mock_openai):
        """Test avec le modèle legacy ada-002."""
        mock_settings.return_value = Mock(
            embedding_model="default-model",
            openai_api_key="key"
        )

        manager = EmbeddingsManager(model="text-embedding-ada-002")
        _ = manager.embeddings

        mock_openai.assert_called_once_with(
            model="text-embedding-ada-002",
            api_key="key"
        )


class TestEmbeddingsManagerVectorDimensions:
    """Tests pour vérifier les dimensions des vecteurs."""

    def setup_method(self):
        """Configuration avant chaque test."""
        self.settings_patcher = patch('app.core.embeddings.get_settings')
        self.openai_patcher = patch('app.core.embeddings.OpenAIEmbeddings')

        self.mock_settings = self.settings_patcher.start()
        self.mock_openai_embeddings = self.openai_patcher.start()

        self.mock_settings.return_value = Mock(
            embedding_model="model",
            openai_api_key="key"
        )

        self.mock_embeddings_instance = Mock()
        self.mock_openai_embeddings.return_value = self.mock_embeddings_instance

    def teardown_method(self):
        """Nettoyage après chaque test."""
        self.settings_patcher.stop()
        self.openai_patcher.stop()

    def test_embed_text_returns_correct_dimensions(self):
        """Vérifie les dimensions d'un vecteur."""
        # Simuler un vecteur de dimension 1536 (ada-002)
        vector = [0.1] * 1536
        self.mock_embeddings_instance.embed_query.return_value = vector

        manager = EmbeddingsManager()
        result = manager.embed_text("Test")

        assert len(result) == 1536

    def test_embed_texts_all_same_dimensions(self):
        """Vérifie que tous les vecteurs ont les mêmes dimensions."""
        vectors = [[0.1] * 1536, [0.2] * 1536, [0.3] * 1536]
        self.mock_embeddings_instance.embed_documents.return_value = vectors

        manager = EmbeddingsManager()
        results = manager.embed_texts(["A", "B", "C"])

        # Toutes les dimensions devraient être identiques
        dimensions = [len(v) for v in results]
        assert all(d == 1536 for d in dimensions)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
