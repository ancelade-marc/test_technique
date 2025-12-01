"""
Gestionnaire d'embeddings pour la vectorisation des documents.

Encapsule la création des vecteurs de représentation textuelle
utilisés pour la recherche sémantique.
"""

from typing import Optional
from langchain_openai import OpenAIEmbeddings

from app.config import get_settings
from app.utils.logger import get_logger


logger = get_logger("embeddings")


class EmbeddingsManager:
    """
    Gestionnaire pour la création d'embeddings via OpenAI.

    Fournit une interface unifiée pour convertir du texte
    en vecteurs numériques pour la recherche sémantique.
    """

    def __init__(self, model: Optional[str] = None):
        """
        Initialise le gestionnaire d'embeddings.

        Args:
            model: Nom du modèle d'embeddings (défaut: config)
        """
        settings = get_settings()
        self.model = model or settings.embedding_model
        self._embeddings: Optional[OpenAIEmbeddings] = None

        logger.info(f"Gestionnaire d'embeddings initialisé avec {self.model}")

    @property
    def embeddings(self) -> OpenAIEmbeddings:
        """
        Retourne l'instance OpenAIEmbeddings (lazy loading).

        L'instance est créée à la première utilisation pour
        permettre une configuration différée de la clé API.
        """
        if self._embeddings is None:
            settings = get_settings()

            if not settings.openai_api_key:
                raise ValueError(
                    "La clé API OpenAI n'est pas configurée. "
                    "Veuillez définir OPENAI_API_KEY dans le fichier .env"
                )

            self._embeddings = OpenAIEmbeddings(
                model=self.model,
                api_key=settings.openai_api_key,
            )

        return self._embeddings

    def embed_text(self, text: str) -> list[float]:
        """
        Génère l'embedding d'un texte unique.

        Args:
            text: Texte à vectoriser

        Returns:
            list[float]: Vecteur de représentation
        """
        try:
            logger.debug(f"Génération de l'embedding pour {len(text)} caractères")
            return self.embeddings.embed_query(text)

        except Exception as e:
            logger.error(f"Erreur lors de la génération de l'embedding: {str(e)}")
            raise

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Génère les embeddings pour une liste de textes.

        Optimisé pour le traitement par lots lors de l'indexation
        de documents volumineux.

        Args:
            texts: Liste de textes à vectoriser

        Returns:
            list[list[float]]: Liste des vecteurs de représentation
        """
        try:
            logger.debug(f"Génération des embeddings pour {len(texts)} textes")
            return self.embeddings.embed_documents(texts)

        except Exception as e:
            logger.error(f"Erreur lors de la génération des embeddings: {str(e)}")
            raise

    def get_langchain_embeddings(self) -> OpenAIEmbeddings:
        """
        Retourne l'objet embeddings LangChain pour intégration directe.

        Utilisé par VectorStoreManager pour la création de la base
        vectorielle ChromaDB.

        Returns:
            OpenAIEmbeddings: Instance LangChain
        """
        return self.embeddings
