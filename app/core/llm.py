"""
Client LLM pour l'interaction avec l'API OpenAI.

Encapsule la logique d'appel au modèle de langage avec
gestion des erreurs et configuration centralisée.
"""

from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from app.config import get_settings
from app.utils.logger import get_logger


logger = get_logger("llm")


class LLMClient:
    """
    Client pour interagir avec les modèles de langage OpenAI.

    Fournit une interface simplifiée pour l'envoi de requêtes
    au LLM avec gestion des paramètres et des erreurs.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        """
        Initialise le client LLM.

        Args:
            model: Nom du modèle à utiliser (défaut: config)
            temperature: Température pour la génération (défaut: config)
            max_tokens: Nombre max de tokens en sortie (défaut: config)
        """
        settings = get_settings()

        self.model = model or settings.llm_model
        self.temperature = temperature if temperature is not None else settings.temperature
        self.max_tokens = max_tokens or settings.max_tokens

        self._client: Optional[ChatOpenAI] = None
        logger.info(f"Client LLM initialisé avec le modèle {self.model}")

    @property
    def client(self) -> ChatOpenAI:
        """
        Retourne l'instance du client ChatOpenAI (lazy loading).

        Le client est créé à la première utilisation pour éviter
        les erreurs si la clé API n'est pas encore configurée.
        """
        if self._client is None:
            settings = get_settings()

            if not settings.openai_api_key:
                raise ValueError(
                    "La clé API OpenAI n'est pas configurée. "
                    "Veuillez définir OPENAI_API_KEY dans le fichier .env"
                )

            self._client = ChatOpenAI(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                api_key=settings.openai_api_key,
            )
        return self._client

    def invoke(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Envoie une requête au LLM et retourne la réponse.

        Args:
            prompt: Message utilisateur à envoyer
            system_prompt: Instructions système optionnelles

        Returns:
            str: Réponse générée par le modèle

        Raises:
            Exception: En cas d'erreur lors de l'appel API
        """
        try:
            messages = []

            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))

            messages.append(HumanMessage(content=prompt))

            logger.debug(f"Envoi de la requête au LLM ({len(prompt)} caractères)")
            response = self.client.invoke(messages)

            logger.debug(f"Réponse reçue ({len(response.content)} caractères)")
            return response.content

        except Exception as e:
            logger.error(f"Erreur lors de l'appel au LLM: {str(e)}")
            raise

    def stream(self, prompt: str, system_prompt: Optional[str] = None):
        """
        Envoie une requête au LLM et retourne un générateur de tokens.

        Permet l'affichage progressif de la réponse dans l'interface.

        Args:
            prompt: Message utilisateur à envoyer
            system_prompt: Instructions système optionnelles

        Yields:
            str: Tokens générés un par un
        """
        try:
            messages = []

            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))

            messages.append(HumanMessage(content=prompt))

            logger.debug("Début du streaming de la réponse")

            for chunk in self.client.stream(messages):
                if chunk.content:
                    yield chunk.content

        except Exception as e:
            logger.error(f"Erreur lors du streaming: {str(e)}")
            raise

    def chat(
        self,
        messages: list[dict],
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Envoie une conversation complète au LLM.

        Args:
            messages: Liste de messages {"role": "user/assistant", "content": "..."}
            system_prompt: Instructions système optionnelles

        Returns:
            str: Réponse générée par le modèle
        """
        try:
            langchain_messages = []

            if system_prompt:
                langchain_messages.append(SystemMessage(content=system_prompt))

            for msg in messages:
                if msg["role"] == "user":
                    langchain_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    langchain_messages.append(AIMessage(content=msg["content"]))

            response = self.client.invoke(langchain_messages)
            return response.content

        except Exception as e:
            logger.error(f"Erreur lors du chat: {str(e)}")
            raise
