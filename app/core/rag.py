"""
Chaîne RAG (Retrieval-Augmented Generation).

Orchestre la récupération de contexte et la génération de réponses
basées exclusivement sur les documents indexés.
"""

from typing import Optional, Generator
from dataclasses import dataclass

from langchain_core.documents import Document

from app.config import get_settings, SYSTEM_PROMPT, NO_CONTEXT_MESSAGE
from app.core.llm import LLMClient
from app.core.vectorstore import VectorStoreManager
from app.utils.logger import get_logger


logger = get_logger("rag")


@dataclass
class RAGResponse:
    """
    Réponse structurée du système RAG.

    Attributes:
        answer: Réponse générée par le LLM
        sources: Documents utilisés comme contexte
        has_context: Indique si des documents pertinents ont été trouvés
    """
    answer: str
    sources: list[Document]
    has_context: bool


class RAGChain:
    """
    Chaîne RAG complète pour la génération de réponses contextuelles.

    Combine la recherche vectorielle et la génération LLM pour
    produire des réponses basées exclusivement sur les documents
    du cabinet.
    """

    # Score minimum de similarité pour considérer un document pertinent
    MIN_RELEVANCE_SCORE = 0.3

    def __init__(
        self,
        vectorstore_manager: Optional[VectorStoreManager] = None,
        llm_client: Optional[LLMClient] = None,
    ):
        """
        Initialise la chaîne RAG.

        Args:
            vectorstore_manager: Gestionnaire de base vectorielle
            llm_client: Client LLM pour la génération
        """
        self.settings = get_settings()
        self.vectorstore = vectorstore_manager or VectorStoreManager()
        self.llm = llm_client or LLMClient()

        logger.info("Chaîne RAG initialisée")

    def query(self, question: str, k: Optional[int] = None) -> RAGResponse:
        """
        Traite une question et génère une réponse basée sur les documents.

        Args:
            question: Question de l'utilisateur
            k: Nombre de documents à utiliser pour le contexte

        Returns:
            RAGResponse: Réponse avec sources et métadonnées
        """
        try:
            k = k or self.settings.retriever_k

            # Récupération des documents pertinents avec scores
            logger.debug(f"Recherche de contexte pour: '{question[:50]}...'")
            results_with_scores = self.vectorstore.search_with_scores(question, k=k)

            # Filtrage par score de pertinence
            relevant_docs = [
                doc for doc, score in results_with_scores
                if score <= self.MIN_RELEVANCE_SCORE  # ChromaDB: score bas = meilleur
            ]

            # Si aucun document pertinent, utiliser quand même les meilleurs résultats
            if not relevant_docs and results_with_scores:
                relevant_docs = [doc for doc, _ in results_with_scores[:k]]

            # Cas sans contexte disponible
            if not relevant_docs:
                logger.warning("Aucun document pertinent trouvé")
                return RAGResponse(
                    answer=NO_CONTEXT_MESSAGE,
                    sources=[],
                    has_context=False
                )

            # Construction du contexte
            context = self._build_context(relevant_docs)

            # Génération de la réponse
            prompt = SYSTEM_PROMPT.format(
                context=context,
                question=question
            )

            logger.debug("Génération de la réponse")
            answer = self.llm.invoke(prompt)

            return RAGResponse(
                answer=answer,
                sources=relevant_docs,
                has_context=True
            )

        except Exception as e:
            logger.error(f"Erreur lors du traitement de la question: {str(e)}")
            raise

    def query_stream(
        self,
        question: str,
        k: Optional[int] = None
    ) -> Generator[str, None, RAGResponse]:
        """
        Traite une question avec streaming de la réponse.

        Permet l'affichage progressif dans l'interface utilisateur.

        Args:
            question: Question de l'utilisateur
            k: Nombre de documents pour le contexte

        Yields:
            str: Tokens de la réponse un par un

        Returns:
            RAGResponse: Réponse complète à la fin du streaming
        """
        try:
            k = k or self.settings.retriever_k

            # Récupération du contexte
            results_with_scores = self.vectorstore.search_with_scores(question, k=k)

            relevant_docs = [
                doc for doc, score in results_with_scores
                if score <= self.MIN_RELEVANCE_SCORE
            ]

            if not relevant_docs and results_with_scores:
                relevant_docs = [doc for doc, _ in results_with_scores[:k]]

            if not relevant_docs:
                yield NO_CONTEXT_MESSAGE
                return RAGResponse(
                    answer=NO_CONTEXT_MESSAGE,
                    sources=[],
                    has_context=False
                )

            context = self._build_context(relevant_docs)
            prompt = SYSTEM_PROMPT.format(
                context=context,
                question=question
            )

            # Streaming de la réponse
            full_response = []
            for token in self.llm.stream(prompt):
                full_response.append(token)
                yield token

            return RAGResponse(
                answer="".join(full_response),
                sources=relevant_docs,
                has_context=True
            )

        except Exception as e:
            logger.error(f"Erreur lors du streaming: {str(e)}")
            raise

    def _build_context(self, documents: list[Document]) -> str:
        """
        Construit le contexte textuel à partir des documents.

        Args:
            documents: Liste des documents pertinents

        Returns:
            str: Contexte formaté pour le prompt
        """
        context_parts = []

        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source_id", "Document inconnu")
            content = doc.page_content

            context_parts.append(
                f"[Document {i} - Source: {source}]\n{content}\n"
            )

        return "\n---\n".join(context_parts)

    def get_sources_summary(self, documents: list[Document]) -> list[dict]:
        """
        Génère un résumé des sources pour l'affichage.

        Args:
            documents: Documents utilisés comme sources

        Returns:
            list[dict]: Liste des informations de source
        """
        sources = []
        seen = set()

        for doc in documents:
            source_id = doc.metadata.get("source_id", "Inconnu")

            if source_id not in seen:
                seen.add(source_id)
                sources.append({
                    "name": source_id,
                    "preview": doc.page_content[:200] + "..."
                    if len(doc.page_content) > 200
                    else doc.page_content
                })

        return sources

    def is_ready(self) -> bool:
        """
        Vérifie si le système RAG est prêt à répondre.

        Returns:
            bool: True si des documents sont indexés
        """
        return self.vectorstore.get_document_count() > 0
