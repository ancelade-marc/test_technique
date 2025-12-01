"""
Tests unitaires pour le client LLM.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.llm import LLMClient


class TestLLMClientInit:
    """Tests pour l'initialisation du LLMClient."""

    @patch('app.core.llm.get_settings')
    def test_init_with_defaults(self, mock_settings):
        """Vérifie l'initialisation avec les valeurs par défaut."""
        mock_settings.return_value = Mock(
            llm_model="gpt-4o-mini",
            temperature=0.1,
            max_tokens=2048
        )

        client = LLMClient()

        assert client.model == "gpt-4o-mini"
        assert client.temperature == 0.1
        assert client.max_tokens == 2048
        assert client._client is None  # Lazy loading

    @patch('app.core.llm.get_settings')
    def test_init_with_custom_values(self, mock_settings):
        """Vérifie l'initialisation avec des valeurs personnalisées."""
        mock_settings.return_value = Mock(
            llm_model="default-model",
            temperature=0.5,
            max_tokens=1000
        )

        client = LLMClient(
            model="gpt-4",
            temperature=0.7,
            max_tokens=4096
        )

        assert client.model == "gpt-4"
        assert client.temperature == 0.7
        assert client.max_tokens == 4096

    @patch('app.core.llm.get_settings')
    def test_init_with_zero_temperature(self, mock_settings):
        """Vérifie que temperature=0 est respecté."""
        mock_settings.return_value = Mock(
            llm_model="model",
            temperature=0.5,
            max_tokens=1000
        )

        client = LLMClient(temperature=0)

        assert client.temperature == 0


class TestLLMClientProperty:
    """Tests pour la propriété client (lazy loading)."""

    @patch('app.core.llm.get_settings')
    def test_client_raises_without_api_key(self, mock_settings):
        """Vérifie l'erreur si la clé API est manquante."""
        mock_settings.return_value = Mock(
            llm_model="model",
            temperature=0.1,
            max_tokens=1000,
            openai_api_key=""
        )

        client = LLMClient()

        with pytest.raises(ValueError) as exc_info:
            _ = client.client

        assert "clé API OpenAI" in str(exc_info.value)

    @patch('app.core.llm.ChatOpenAI')
    @patch('app.core.llm.get_settings')
    def test_client_creates_chat_openai(self, mock_settings, mock_chat_openai):
        """Vérifie la création du client ChatOpenAI."""
        mock_settings.return_value = Mock(
            llm_model="gpt-4o-mini",
            temperature=0.1,
            max_tokens=2048,
            openai_api_key="test-key"
        )

        client = LLMClient()
        _ = client.client

        mock_chat_openai.assert_called_once_with(
            model="gpt-4o-mini",
            temperature=0.1,
            max_tokens=2048,
            api_key="test-key"
        )

    @patch('app.core.llm.ChatOpenAI')
    @patch('app.core.llm.get_settings')
    def test_client_is_cached(self, mock_settings, mock_chat_openai):
        """Vérifie que le client est mis en cache."""
        mock_settings.return_value = Mock(
            llm_model="model",
            temperature=0.1,
            max_tokens=1000,
            openai_api_key="key"
        )

        client = LLMClient()

        # Appels multiples
        _ = client.client
        _ = client.client
        _ = client.client

        # ChatOpenAI ne devrait être appelé qu'une fois
        assert mock_chat_openai.call_count == 1


class TestLLMClientInvoke:
    """Tests pour la méthode invoke."""

    @patch('app.core.llm.ChatOpenAI')
    @patch('app.core.llm.get_settings')
    def test_invoke_returns_content(self, mock_settings, mock_chat_openai):
        """Vérifie que invoke retourne le contenu de la réponse."""
        mock_settings.return_value = Mock(
            llm_model="model",
            temperature=0.1,
            max_tokens=1000,
            openai_api_key="key"
        )

        mock_response = Mock()
        mock_response.content = "Réponse du LLM"
        mock_chat_openai.return_value.invoke.return_value = mock_response

        client = LLMClient()
        result = client.invoke("Question test")

        assert result == "Réponse du LLM"

    @patch('app.core.llm.ChatOpenAI')
    @patch('app.core.llm.get_settings')
    def test_invoke_with_system_prompt(self, mock_settings, mock_chat_openai):
        """Vérifie l'ajout du system prompt."""
        mock_settings.return_value = Mock(
            llm_model="model",
            temperature=0.1,
            max_tokens=1000,
            openai_api_key="key"
        )

        mock_response = Mock()
        mock_response.content = "Réponse"
        mock_chat_instance = mock_chat_openai.return_value
        mock_chat_instance.invoke.return_value = mock_response

        client = LLMClient()
        client.invoke("Question", system_prompt="Tu es un assistant")

        # Vérifier que deux messages ont été envoyés
        call_args = mock_chat_instance.invoke.call_args[0][0]
        assert len(call_args) == 2
        assert call_args[0].content == "Tu es un assistant"
        assert call_args[1].content == "Question"

    @patch('app.core.llm.ChatOpenAI')
    @patch('app.core.llm.get_settings')
    def test_invoke_without_system_prompt(self, mock_settings, mock_chat_openai):
        """Vérifie l'appel sans system prompt."""
        mock_settings.return_value = Mock(
            llm_model="model",
            temperature=0.1,
            max_tokens=1000,
            openai_api_key="key"
        )

        mock_response = Mock()
        mock_response.content = "Réponse"
        mock_chat_instance = mock_chat_openai.return_value
        mock_chat_instance.invoke.return_value = mock_response

        client = LLMClient()
        client.invoke("Question")

        # Vérifier qu'un seul message a été envoyé
        call_args = mock_chat_instance.invoke.call_args[0][0]
        assert len(call_args) == 1

    @patch('app.core.llm.ChatOpenAI')
    @patch('app.core.llm.get_settings')
    def test_invoke_raises_on_error(self, mock_settings, mock_chat_openai):
        """Vérifie la propagation des erreurs."""
        mock_settings.return_value = Mock(
            llm_model="model",
            temperature=0.1,
            max_tokens=1000,
            openai_api_key="key"
        )

        mock_chat_openai.return_value.invoke.side_effect = Exception("API Error")

        client = LLMClient()

        with pytest.raises(Exception) as exc_info:
            client.invoke("Question")

        assert "API Error" in str(exc_info.value)


class TestLLMClientStream:
    """Tests pour la méthode stream."""

    @patch('app.core.llm.ChatOpenAI')
    @patch('app.core.llm.get_settings')
    def test_stream_yields_tokens(self, mock_settings, mock_chat_openai):
        """Vérifie que stream yield les tokens."""
        mock_settings.return_value = Mock(
            llm_model="model",
            temperature=0.1,
            max_tokens=1000,
            openai_api_key="key"
        )

        # Création de chunks mockés
        chunks = [
            Mock(content="Hello"),
            Mock(content=" "),
            Mock(content="World")
        ]
        mock_chat_openai.return_value.stream.return_value = iter(chunks)

        client = LLMClient()
        tokens = list(client.stream("Question"))

        assert tokens == ["Hello", " ", "World"]

    @patch('app.core.llm.ChatOpenAI')
    @patch('app.core.llm.get_settings')
    def test_stream_skips_empty_content(self, mock_settings, mock_chat_openai):
        """Vérifie que les chunks vides sont ignorés."""
        mock_settings.return_value = Mock(
            llm_model="model",
            temperature=0.1,
            max_tokens=1000,
            openai_api_key="key"
        )

        chunks = [
            Mock(content="Token1"),
            Mock(content=""),  # Chunk vide
            Mock(content=None),  # Chunk None
            Mock(content="Token2")
        ]
        mock_chat_openai.return_value.stream.return_value = iter(chunks)

        client = LLMClient()
        tokens = list(client.stream("Question"))

        assert tokens == ["Token1", "Token2"]

    @patch('app.core.llm.ChatOpenAI')
    @patch('app.core.llm.get_settings')
    def test_stream_with_system_prompt(self, mock_settings, mock_chat_openai):
        """Vérifie le streaming avec system prompt."""
        mock_settings.return_value = Mock(
            llm_model="model",
            temperature=0.1,
            max_tokens=1000,
            openai_api_key="key"
        )

        mock_chat_instance = mock_chat_openai.return_value
        mock_chat_instance.stream.return_value = iter([Mock(content="Token")])

        client = LLMClient()
        list(client.stream("Question", system_prompt="System"))

        call_args = mock_chat_instance.stream.call_args[0][0]
        assert len(call_args) == 2

    @patch('app.core.llm.ChatOpenAI')
    @patch('app.core.llm.get_settings')
    def test_stream_raises_on_error(self, mock_settings, mock_chat_openai):
        """Vérifie la propagation des erreurs en streaming."""
        mock_settings.return_value = Mock(
            llm_model="model",
            temperature=0.1,
            max_tokens=1000,
            openai_api_key="key"
        )

        mock_chat_openai.return_value.stream.side_effect = Exception("Stream Error")

        client = LLMClient()

        with pytest.raises(Exception) as exc_info:
            list(client.stream("Question"))

        assert "Stream Error" in str(exc_info.value)


class TestLLMClientChat:
    """Tests pour la méthode chat."""

    @patch('app.core.llm.ChatOpenAI')
    @patch('app.core.llm.get_settings')
    def test_chat_with_conversation(self, mock_settings, mock_chat_openai):
        """Vérifie le chat avec un historique de conversation."""
        mock_settings.return_value = Mock(
            llm_model="model",
            temperature=0.1,
            max_tokens=1000,
            openai_api_key="key"
        )

        mock_response = Mock()
        mock_response.content = "Réponse assistant"
        mock_chat_instance = mock_chat_openai.return_value
        mock_chat_instance.invoke.return_value = mock_response

        messages = [
            {"role": "user", "content": "Bonjour"},
            {"role": "assistant", "content": "Bonjour! Comment puis-je vous aider?"},
            {"role": "user", "content": "Qu'est-ce qu'une SAS?"}
        ]

        client = LLMClient()
        result = client.chat(messages)

        assert result == "Réponse assistant"

        # Vérifier que les messages ont été convertis correctement
        call_args = mock_chat_instance.invoke.call_args[0][0]
        assert len(call_args) == 3

    @patch('app.core.llm.ChatOpenAI')
    @patch('app.core.llm.get_settings')
    def test_chat_with_system_prompt(self, mock_settings, mock_chat_openai):
        """Vérifie le chat avec system prompt."""
        mock_settings.return_value = Mock(
            llm_model="model",
            temperature=0.1,
            max_tokens=1000,
            openai_api_key="key"
        )

        mock_response = Mock()
        mock_response.content = "Réponse"
        mock_chat_instance = mock_chat_openai.return_value
        mock_chat_instance.invoke.return_value = mock_response

        messages = [{"role": "user", "content": "Question"}]

        client = LLMClient()
        client.chat(messages, system_prompt="Tu es un assistant juridique")

        call_args = mock_chat_instance.invoke.call_args[0][0]
        # System + 1 user message
        assert len(call_args) == 2

    @patch('app.core.llm.ChatOpenAI')
    @patch('app.core.llm.get_settings')
    def test_chat_empty_messages(self, mock_settings, mock_chat_openai):
        """Vérifie le chat avec liste de messages vide."""
        mock_settings.return_value = Mock(
            llm_model="model",
            temperature=0.1,
            max_tokens=1000,
            openai_api_key="key"
        )

        mock_response = Mock()
        mock_response.content = "Réponse"
        mock_chat_openai.return_value.invoke.return_value = mock_response

        client = LLMClient()
        result = client.chat([])

        assert result == "Réponse"

    @patch('app.core.llm.ChatOpenAI')
    @patch('app.core.llm.get_settings')
    def test_chat_raises_on_error(self, mock_settings, mock_chat_openai):
        """Vérifie la propagation des erreurs."""
        mock_settings.return_value = Mock(
            llm_model="model",
            temperature=0.1,
            max_tokens=1000,
            openai_api_key="key"
        )

        mock_chat_openai.return_value.invoke.side_effect = Exception("Chat Error")

        client = LLMClient()

        with pytest.raises(Exception) as exc_info:
            client.chat([{"role": "user", "content": "Test"}])

        assert "Chat Error" in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
