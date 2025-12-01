"""
Gestionnaire d'historique des conversations.

Gère la persistance et la récupération des conversations
pour permettre un suivi contextuel des échanges.
"""

import json
import uuid
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime

from app.config import get_paths
from app.utils.logger import get_logger


logger = get_logger("conversation")


@dataclass
class Message:
    """
    Message dans une conversation.

    Attributes:
        role: Rôle de l'auteur ('user' ou 'assistant')
        content: Contenu du message
        timestamp: Horodatage du message
        sources: Sources utilisées (pour les réponses assistant)
    """
    role: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    sources: list[str] = field(default_factory=list)


@dataclass
class Conversation:
    """
    Conversation complète avec métadonnées.

    Attributes:
        id: Identifiant unique de la conversation
        title: Titre de la conversation
        messages: Liste des messages
        created_at: Date de création
        updated_at: Date de dernière modification
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = "Nouvelle conversation"
    messages: list[Message] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def add_message(self, role: str, content: str, sources: list[str] = None) -> None:
        """Ajoute un message à la conversation."""
        message = Message(
            role=role,
            content=content,
            sources=sources or []
        )
        self.messages.append(message)
        self.updated_at = datetime.now().isoformat()

        # Mise à jour du titre basé sur le premier message utilisateur
        if role == "user" and len(self.messages) == 1:
            self.title = self._generate_title(content)

    def _generate_title(self, content: str) -> str:
        """Génère un titre basé sur le premier message."""
        # Limite à 50 caractères
        title = content[:50].strip()
        if len(content) > 50:
            title += "..."
        return title

    def to_dict(self) -> dict:
        """Convertit la conversation en dictionnaire."""
        return {
            "id": self.id,
            "title": self.title,
            "messages": [asdict(m) for m in self.messages],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Conversation":
        """Crée une conversation depuis un dictionnaire."""
        messages = [
            Message(**msg) for msg in data.get("messages", [])
        ]
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            title=data.get("title", "Conversation"),
            messages=messages,
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
        )


class ConversationManager:
    """
    Gestionnaire de l'historique des conversations.

    Gère la persistance JSON des conversations et fournit
    des méthodes pour créer, charger et supprimer des conversations.
    """

    HISTORY_FILE = "conversations.json"

    def __init__(self):
        """Initialise le gestionnaire de conversations."""
        self.paths = get_paths()
        self.history_path = self.paths.conversations / self.HISTORY_FILE

        # Chargement de l'historique existant
        self._conversations: dict[str, Conversation] = {}
        self._load_history()

        logger.info(
            f"ConversationManager initialisé "
            f"({len(self._conversations)} conversations)"
        )

    def create_conversation(self) -> Conversation:
        """
        Crée une nouvelle conversation.

        Returns:
            Conversation: Nouvelle conversation vide
        """
        conversation = Conversation()
        self._conversations[conversation.id] = conversation
        self._save_history()

        logger.debug(f"Nouvelle conversation créée: {conversation.id}")
        return conversation

    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """
        Récupère une conversation par son ID.

        Args:
            conversation_id: Identifiant de la conversation

        Returns:
            Optional[Conversation]: Conversation ou None si non trouvée
        """
        return self._conversations.get(conversation_id)

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        sources: list[str] = None
    ) -> bool:
        """
        Ajoute un message à une conversation existante.

        Args:
            conversation_id: ID de la conversation
            role: Rôle ('user' ou 'assistant')
            content: Contenu du message
            sources: Sources utilisées

        Returns:
            bool: True si l'ajout a réussi
        """
        conversation = self._conversations.get(conversation_id)

        if not conversation:
            logger.warning(f"Conversation non trouvée: {conversation_id}")
            return False

        conversation.add_message(role, content, sources)
        self._save_history()

        return True

    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Supprime une conversation.

        Args:
            conversation_id: ID de la conversation à supprimer

        Returns:
            bool: True si la suppression a réussi
        """
        if conversation_id in self._conversations:
            del self._conversations[conversation_id]
            self._save_history()
            logger.info(f"Conversation supprimée: {conversation_id}")
            return True

        return False

    def list_conversations(self) -> list[Conversation]:
        """
        Liste toutes les conversations triées par date.

        Returns:
            list[Conversation]: Conversations triées (plus récentes d'abord)
        """
        return sorted(
            self._conversations.values(),
            key=lambda c: c.updated_at,
            reverse=True
        )

    def get_messages_for_context(
        self,
        conversation_id: str,
        max_messages: int = 10
    ) -> list[dict]:
        """
        Récupère les messages formatés pour le contexte LLM.

        Args:
            conversation_id: ID de la conversation
            max_messages: Nombre max de messages à inclure

        Returns:
            list[dict]: Messages formatés pour le LLM
        """
        conversation = self._conversations.get(conversation_id)

        if not conversation:
            return []

        # Récupération des derniers messages
        messages = conversation.messages[-max_messages:]

        return [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

    def clear_all(self) -> None:
        """Supprime toutes les conversations."""
        self._conversations = {}
        self._save_history()
        logger.info("Toutes les conversations ont été supprimées")

    def _load_history(self) -> None:
        """Charge l'historique depuis le fichier JSON."""
        if not self.history_path.exists():
            return

        try:
            with open(self.history_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for conv_data in data.get("conversations", []):
                conversation = Conversation.from_dict(conv_data)
                self._conversations[conversation.id] = conversation

            logger.debug(f"Historique chargé: {len(self._conversations)} conversations")

        except Exception as e:
            logger.error(f"Erreur chargement historique: {str(e)}")
            self._conversations = {}

    def _save_history(self) -> None:
        """Sauvegarde l'historique dans le fichier JSON."""
        try:
            data = {
                "conversations": [
                    conv.to_dict() for conv in self._conversations.values()
                ]
            }

            with open(self.history_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"Erreur sauvegarde historique: {str(e)}")
