"""
Tests unitaires pour le module de configuration.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, Mock

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import (
    Settings,
    Paths,
    get_settings,
    get_paths,
    APP_TITLE,
    APP_ICON,
    PAGE_CHAT,
    PAGE_DOCUMENTS,
    SYSTEM_PROMPT,
    NO_CONTEXT_MESSAGE
)


class TestSettings:
    """Tests pour la classe Settings."""

    def test_settings_default_values(self):
        """Vérifie les valeurs par défaut des settings."""
        # Note: les valeurs peuvent être surchargées par .env
        settings = Settings()

        # Vérifier les types et contraintes plutôt que les valeurs exactes
        assert isinstance(settings.llm_model, str)
        assert len(settings.llm_model) > 0
        assert isinstance(settings.embedding_model, str)
        assert settings.chunk_size > 0
        assert settings.chunk_overlap >= 0
        assert settings.retriever_k > 0
        assert 0 <= settings.temperature <= 2
        assert settings.max_tokens > 0
        assert isinstance(settings.debug, bool)
        assert settings.log_level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        assert ".txt" in settings.allowed_extensions
        assert ".csv" in settings.allowed_extensions
        assert ".html" in settings.allowed_extensions
        assert settings.max_file_size_mb > 0

    def test_settings_from_env(self):
        """Vérifie le chargement depuis les variables d'environnement."""
        env_vars = {
            'OPENAI_API_KEY': 'test-key-123',
            'LLM_MODEL': 'gpt-4',
            'TEMPERATURE': '0.5',
            'CHUNK_SIZE': '500',
            'DEBUG': 'true',
            'LOG_LEVEL': 'DEBUG'
        }

        with patch.dict('os.environ', env_vars, clear=False):
            settings = Settings()

            assert settings.openai_api_key == 'test-key-123'
            assert settings.llm_model == 'gpt-4'
            assert settings.temperature == 0.5
            assert settings.chunk_size == 500
            assert settings.debug is True
            assert settings.log_level == 'DEBUG'

    def test_settings_allowed_extensions_list(self):
        """Vérifie que allowed_extensions est une liste."""
        settings = Settings()
        assert isinstance(settings.allowed_extensions, list)
        assert len(settings.allowed_extensions) >= 3


class TestPaths:
    """Tests pour la classe Paths."""

    def test_paths_creation(self):
        """Vérifie la création des chemins."""
        paths = Paths()

        assert paths.root is not None
        assert paths.app is not None
        assert paths.data is not None
        assert paths.documents is not None
        assert paths.vectorstore is not None
        assert paths.conversations is not None

    def test_paths_are_path_objects(self):
        """Vérifie que les chemins sont des objets Path."""
        paths = Paths()

        assert isinstance(paths.root, Path)
        assert isinstance(paths.app, Path)
        assert isinstance(paths.data, Path)
        assert isinstance(paths.documents, Path)
        assert isinstance(paths.vectorstore, Path)
        assert isinstance(paths.conversations, Path)

    def test_paths_relationships(self):
        """Vérifie les relations entre les chemins."""
        paths = Paths()

        assert paths.app.parent == paths.root
        assert paths.data.parent == paths.root
        assert paths.documents.parent == paths.data
        assert paths.vectorstore.parent == paths.data
        assert paths.conversations.parent == paths.data

    def test_paths_directories_exist(self):
        """Vérifie que les dossiers sont créés."""
        paths = Paths()

        assert paths.documents.exists()
        assert paths.vectorstore.exists()
        assert paths.conversations.exists()


class TestGetSettings:
    """Tests pour la fonction get_settings."""

    def test_get_settings_returns_settings(self):
        """Vérifie que get_settings retourne un Settings."""
        # Clear cache first
        get_settings.cache_clear()

        settings = get_settings()

        assert isinstance(settings, Settings)

    def test_get_settings_is_cached(self):
        """Vérifie que get_settings est mis en cache."""
        get_settings.cache_clear()

        settings1 = get_settings()
        settings2 = get_settings()

        assert settings1 is settings2


class TestGetPaths:
    """Tests pour la fonction get_paths."""

    def test_get_paths_returns_paths(self):
        """Vérifie que get_paths retourne un Paths."""
        # Clear cache first
        get_paths.cache_clear()

        paths = get_paths()

        assert isinstance(paths, Paths)

    def test_get_paths_is_cached(self):
        """Vérifie que get_paths est mis en cache."""
        get_paths.cache_clear()

        paths1 = get_paths()
        paths2 = get_paths()

        assert paths1 is paths2


class TestConstants:
    """Tests pour les constantes de l'interface."""

    def test_app_title(self):
        """Vérifie APP_TITLE."""
        assert APP_TITLE == "Assistant Juridique"

    def test_app_icon(self):
        """Vérifie APP_ICON."""
        assert APP_ICON == "balance_scale"

    def test_page_chat(self):
        """Vérifie PAGE_CHAT."""
        assert PAGE_CHAT == "Chat"

    def test_page_documents(self):
        """Vérifie PAGE_DOCUMENTS."""
        assert PAGE_DOCUMENTS == "Documents"


class TestSystemPrompt:
    """Tests pour le prompt système."""

    def test_system_prompt_contains_placeholders(self):
        """Vérifie que SYSTEM_PROMPT contient les placeholders."""
        assert "{context}" in SYSTEM_PROMPT
        assert "{question}" in SYSTEM_PROMPT

    def test_system_prompt_is_formattable(self):
        """Vérifie que SYSTEM_PROMPT peut être formaté."""
        formatted = SYSTEM_PROMPT.format(
            context="Contexte test",
            question="Question test"
        )

        assert "Contexte test" in formatted
        assert "Question test" in formatted

    def test_system_prompt_contains_instructions(self):
        """Vérifie que SYSTEM_PROMPT contient des instructions."""
        assert "juridique" in SYSTEM_PROMPT.lower()
        assert "documents" in SYSTEM_PROMPT.lower()

    def test_no_context_message(self):
        """Vérifie NO_CONTEXT_MESSAGE."""
        assert len(NO_CONTEXT_MESSAGE) > 0
        assert "document" in NO_CONTEXT_MESSAGE.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
