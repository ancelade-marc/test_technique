"""
Tests unitaires pour le gestionnaire de conversations.
"""

import pytest
import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.conversation import (
    Message,
    Conversation,
    ConversationManager
)


class TestMessage:
    """Tests pour la dataclass Message."""

    def test_message_creation(self):
        """Vérifie la création d'un message."""
        msg = Message(
            role="user",
            content="Bonjour"
        )
        assert msg.role == "user"
        assert msg.content == "Bonjour"
        assert msg.timestamp is not None
        assert msg.sources == []

    def test_message_with_sources(self):
        """Vérifie la création d'un message avec sources."""
        msg = Message(
            role="assistant",
            content="Réponse",
            sources=["doc1.txt", "doc2.txt"]
        )
        assert len(msg.sources) == 2

    def test_message_timestamp_format(self):
        """Vérifie le format ISO du timestamp."""
        msg = Message(role="user", content="Test")
        # Le timestamp devrait être parseable en datetime
        datetime.fromisoformat(msg.timestamp)


class TestConversation:
    """Tests pour la dataclass Conversation."""

    def test_conversation_creation(self):
        """Vérifie la création d'une conversation."""
        conv = Conversation()
        assert conv.id is not None
        assert conv.title == "Nouvelle conversation"
        assert conv.messages == []
        assert conv.created_at is not None
        assert conv.updated_at is not None

    def test_conversation_with_custom_values(self):
        """Vérifie la création avec valeurs personnalisées."""
        conv = Conversation(
            id="custom-id",
            title="Titre personnalisé"
        )
        assert conv.id == "custom-id"
        assert conv.title == "Titre personnalisé"

    def test_add_message_user(self):
        """Vérifie l'ajout d'un message utilisateur."""
        conv = Conversation()
        initial_updated_at = conv.updated_at

        conv.add_message("user", "Question test")

        assert len(conv.messages) == 1
        assert conv.messages[0].role == "user"
        assert conv.messages[0].content == "Question test"
        # updated_at devrait être mis à jour
        assert conv.updated_at >= initial_updated_at

    def test_add_message_assistant(self):
        """Vérifie l'ajout d'un message assistant."""
        conv = Conversation()

        conv.add_message("assistant", "Réponse", sources=["source.txt"])

        assert len(conv.messages) == 1
        assert conv.messages[0].role == "assistant"
        assert conv.messages[0].sources == ["source.txt"]

    def test_add_message_updates_title_on_first_user_message(self):
        """Vérifie la mise à jour du titre au premier message."""
        conv = Conversation()

        conv.add_message("user", "Qu'est-ce qu'une SAS?")

        assert conv.title == "Qu'est-ce qu'une SAS?"

    def test_title_truncated_if_too_long(self):
        """Vérifie la troncature du titre si trop long."""
        conv = Conversation()

        long_message = "A" * 100
        conv.add_message("user", long_message)

        assert len(conv.title) <= 53  # 50 + "..."
        assert conv.title.endswith("...")

    def test_title_not_updated_on_subsequent_messages(self):
        """Vérifie que le titre n'est pas modifié après le premier message."""
        conv = Conversation()

        conv.add_message("user", "Premier message")
        first_title = conv.title

        conv.add_message("user", "Deuxième message")

        assert conv.title == first_title

    def test_to_dict(self):
        """Vérifie la conversion en dictionnaire."""
        conv = Conversation(id="test-id", title="Test Title")
        conv.add_message("user", "Question")

        data = conv.to_dict()

        assert data["id"] == "test-id"
        # Le titre est mis à jour par le premier message user
        assert data["title"] == "Question"
        assert len(data["messages"]) == 1
        assert "created_at" in data
        assert "updated_at" in data

    def test_from_dict(self):
        """Vérifie la création depuis un dictionnaire."""
        data = {
            "id": "restored-id",
            "title": "Restored Title",
            "messages": [
                {"role": "user", "content": "Test", "timestamp": "2024-01-01T00:00:00", "sources": []}
            ],
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00"
        }

        conv = Conversation.from_dict(data)

        assert conv.id == "restored-id"
        assert conv.title == "Restored Title"
        assert len(conv.messages) == 1

    def test_from_dict_with_defaults(self):
        """Vérifie from_dict avec valeurs manquantes."""
        data = {}

        conv = Conversation.from_dict(data)

        assert conv.id is not None
        assert conv.title == "Conversation"
        assert conv.messages == []


class TestConversationManager:
    """Tests pour la classe ConversationManager."""

    def setup_method(self):
        """Configuration avant chaque test."""
        # Créer un dossier temporaire pour les tests
        self.temp_dir = tempfile.mkdtemp()
        self.conversations_dir = Path(self.temp_dir) / "conversations"
        self.conversations_dir.mkdir(parents=True, exist_ok=True)

        self.paths_patcher = patch('app.services.conversation.get_paths')
        self.mock_paths = self.paths_patcher.start()
        self.mock_paths.return_value = Mock(conversations=self.conversations_dir)

    def teardown_method(self):
        """Nettoyage après chaque test."""
        self.paths_patcher.stop()
        # Nettoyer le dossier temporaire
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init_creates_empty_manager(self):
        """Vérifie l'initialisation avec aucune conversation."""
        manager = ConversationManager()
        assert len(manager._conversations) == 0

    def test_init_loads_existing_conversations(self):
        """Vérifie le chargement des conversations existantes."""
        # Créer un fichier de conversations
        history_data = {
            "conversations": [
                {
                    "id": "existing-id",
                    "title": "Existing",
                    "messages": [],
                    "created_at": "2024-01-01T00:00:00",
                    "updated_at": "2024-01-01T00:00:00"
                }
            ]
        }
        history_path = self.conversations_dir / "conversations.json"
        with open(history_path, "w") as f:
            json.dump(history_data, f)

        manager = ConversationManager()

        assert len(manager._conversations) == 1
        assert "existing-id" in manager._conversations

    def test_create_conversation(self):
        """Vérifie la création d'une conversation."""
        manager = ConversationManager()

        conv = manager.create_conversation()

        assert conv.id in manager._conversations
        assert conv.title == "Nouvelle conversation"

    def test_create_conversation_persists(self):
        """Vérifie que la création persiste sur disque."""
        manager = ConversationManager()

        manager.create_conversation()

        # Vérifier que le fichier existe
        history_path = self.conversations_dir / "conversations.json"
        assert history_path.exists()

    def test_get_conversation_existing(self):
        """Vérifie la récupération d'une conversation existante."""
        manager = ConversationManager()
        created = manager.create_conversation()

        retrieved = manager.get_conversation(created.id)

        assert retrieved is not None
        assert retrieved.id == created.id

    def test_get_conversation_not_found(self):
        """Vérifie le retour None pour conversation inexistante."""
        manager = ConversationManager()

        result = manager.get_conversation("nonexistent-id")

        assert result is None

    def test_add_message(self):
        """Vérifie l'ajout d'un message."""
        manager = ConversationManager()
        conv = manager.create_conversation()

        result = manager.add_message(conv.id, "user", "Question")

        assert result is True
        assert len(manager._conversations[conv.id].messages) == 1

    def test_add_message_to_nonexistent_conversation(self):
        """Vérifie l'ajout à une conversation inexistante."""
        manager = ConversationManager()

        result = manager.add_message("fake-id", "user", "Question")

        assert result is False

    def test_add_message_with_sources(self):
        """Vérifie l'ajout d'un message avec sources."""
        manager = ConversationManager()
        conv = manager.create_conversation()

        manager.add_message(conv.id, "assistant", "Réponse", sources=["doc.txt"])

        msg = manager._conversations[conv.id].messages[0]
        assert msg.sources == ["doc.txt"]

    def test_delete_conversation(self):
        """Vérifie la suppression d'une conversation."""
        manager = ConversationManager()
        conv = manager.create_conversation()

        result = manager.delete_conversation(conv.id)

        assert result is True
        assert conv.id not in manager._conversations

    def test_delete_conversation_not_found(self):
        """Vérifie la suppression d'une conversation inexistante."""
        manager = ConversationManager()

        result = manager.delete_conversation("fake-id")

        assert result is False

    def test_list_conversations_sorted_by_date(self):
        """Vérifie que les conversations sont triées par date."""
        manager = ConversationManager()

        conv1 = manager.create_conversation()
        conv2 = manager.create_conversation()

        # Modifier updated_at de conv1 pour être plus ancien
        manager._conversations[conv1.id].updated_at = "2024-01-01T00:00:00"
        manager._conversations[conv2.id].updated_at = "2024-12-01T00:00:00"

        convs = manager.list_conversations()

        # conv2 devrait être en premier (plus récent)
        assert convs[0].id == conv2.id

    def test_list_conversations_empty(self):
        """Vérifie la liste vide."""
        manager = ConversationManager()

        convs = manager.list_conversations()

        assert convs == []

    def test_get_messages_for_context(self):
        """Vérifie la récupération des messages pour le contexte."""
        manager = ConversationManager()
        conv = manager.create_conversation()

        manager.add_message(conv.id, "user", "Question 1")
        manager.add_message(conv.id, "assistant", "Réponse 1")
        manager.add_message(conv.id, "user", "Question 2")

        messages = manager.get_messages_for_context(conv.id)

        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Question 1"

    def test_get_messages_for_context_limited(self):
        """Vérifie la limite de messages pour le contexte."""
        manager = ConversationManager()
        conv = manager.create_conversation()

        # Ajouter 15 messages
        for i in range(15):
            manager.add_message(conv.id, "user", f"Message {i}")

        messages = manager.get_messages_for_context(conv.id, max_messages=5)

        assert len(messages) == 5
        # Devrait retourner les 5 derniers messages
        assert messages[0]["content"] == "Message 10"

    def test_get_messages_for_context_not_found(self):
        """Vérifie le retour vide pour conversation inexistante."""
        manager = ConversationManager()

        messages = manager.get_messages_for_context("fake-id")

        assert messages == []

    def test_clear_all(self):
        """Vérifie la suppression de toutes les conversations."""
        manager = ConversationManager()

        manager.create_conversation()
        manager.create_conversation()
        manager.create_conversation()

        manager.clear_all()

        assert len(manager._conversations) == 0

    def test_persistence_save_and_load(self):
        """Vérifie la persistance complète."""
        # Créer et sauvegarder
        manager1 = ConversationManager()
        conv = manager1.create_conversation()
        manager1.add_message(conv.id, "user", "Test message")

        # Créer un nouveau manager qui devrait charger les données
        manager2 = ConversationManager()

        assert conv.id in manager2._conversations
        assert len(manager2._conversations[conv.id].messages) == 1


class TestConversationManagerHistoryFile:
    """Tests pour la gestion du fichier d'historique."""

    def setup_method(self):
        """Configuration avant chaque test."""
        self.temp_dir = tempfile.mkdtemp()
        self.conversations_dir = Path(self.temp_dir) / "conversations"
        self.conversations_dir.mkdir(parents=True, exist_ok=True)

        self.paths_patcher = patch('app.services.conversation.get_paths')
        self.mock_paths = self.paths_patcher.start()
        self.mock_paths.return_value = Mock(conversations=self.conversations_dir)

    def teardown_method(self):
        """Nettoyage après chaque test."""
        self.paths_patcher.stop()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_history_file_name(self):
        """Vérifie le nom du fichier d'historique."""
        assert ConversationManager.HISTORY_FILE == "conversations.json"

    def test_load_corrupted_file(self):
        """Vérifie la gestion d'un fichier corrompu."""
        history_path = self.conversations_dir / "conversations.json"
        with open(history_path, "w") as f:
            f.write("not valid json {{{")

        # Ne devrait pas lever d'exception
        manager = ConversationManager()

        assert len(manager._conversations) == 0

    def test_save_creates_file(self):
        """Vérifie que la sauvegarde crée le fichier."""
        manager = ConversationManager()
        manager.create_conversation()

        history_path = self.conversations_dir / "conversations.json"
        assert history_path.exists()

        with open(history_path) as f:
            data = json.load(f)

        assert "conversations" in data
        assert len(data["conversations"]) == 1

    def test_save_preserves_unicode(self):
        """Vérifie la préservation des caractères Unicode."""
        manager = ConversationManager()
        conv = manager.create_conversation()
        manager.add_message(conv.id, "user", "Café résumé été")

        # Recharger
        manager2 = ConversationManager()

        msg = manager2._conversations[conv.id].messages[0]
        assert msg.content == "Café résumé été"


class TestConversationManagerSaveErrors:
    """Tests pour les erreurs de sauvegarde."""

    def setup_method(self):
        """Configuration avant chaque test."""
        self.temp_dir = tempfile.mkdtemp()
        self.conversations_dir = Path(self.temp_dir) / "conversations"
        self.conversations_dir.mkdir(parents=True, exist_ok=True)

        self.paths_patcher = patch('app.services.conversation.get_paths')
        self.mock_paths = self.paths_patcher.start()
        self.mock_paths.return_value = Mock(conversations=self.conversations_dir)

    def teardown_method(self):
        """Nettoyage après chaque test."""
        self.paths_patcher.stop()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_history_exception(self):
        """Vérifie la gestion des erreurs lors de la sauvegarde."""
        manager = ConversationManager()
        manager.create_conversation()

        # Simuler une erreur lors de l'écriture
        with patch('builtins.open', side_effect=PermissionError("Permission denied")):
            # Ne devrait pas lever d'exception
            manager._save_history()


class TestConversationManagerEdgeCases:
    """Tests pour les cas limites."""

    def setup_method(self):
        """Configuration avant chaque test."""
        self.temp_dir = tempfile.mkdtemp()
        self.conversations_dir = Path(self.temp_dir) / "conversations"
        self.conversations_dir.mkdir(parents=True, exist_ok=True)

        self.paths_patcher = patch('app.services.conversation.get_paths')
        self.mock_paths = self.paths_patcher.start()
        self.mock_paths.return_value = Mock(conversations=self.conversations_dir)

    def teardown_method(self):
        """Nettoyage après chaque test."""
        self.paths_patcher.stop()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_empty_message_content(self):
        """Vérifie la gestion d'un message vide."""
        manager = ConversationManager()
        conv = manager.create_conversation()

        result = manager.add_message(conv.id, "user", "")

        assert result is True
        assert manager._conversations[conv.id].messages[0].content == ""

    def test_very_long_message(self):
        """Vérifie la gestion d'un très long message."""
        manager = ConversationManager()
        conv = manager.create_conversation()

        long_content = "A" * 100000  # 100k caractères
        manager.add_message(conv.id, "user", long_content)

        # Recharger pour vérifier la persistance
        manager2 = ConversationManager()
        msg = manager2._conversations[conv.id].messages[0]

        assert len(msg.content) == 100000

    def test_special_characters_in_message(self):
        """Vérifie la gestion des caractères spéciaux."""
        manager = ConversationManager()
        conv = manager.create_conversation()

        special_content = '{"key": "value"}\n<script>alert("test")</script>'
        manager.add_message(conv.id, "user", special_content)

        # Recharger
        manager2 = ConversationManager()
        msg = manager2._conversations[conv.id].messages[0]

        assert msg.content == special_content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
