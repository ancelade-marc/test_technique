"""
Gestionnaire de la base vectorielle ChromaDB.

Encapsule toutes les opérations sur la base vectorielle :
création, ajout de documents, recherche et suppression.
"""

from typing import Optional
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config import get_paths, get_settings
from app.core.embeddings import EmbeddingsManager
from app.utils.logger import get_logger


logger = get_logger("vectorstore")


class VectorStoreManager:
    """
    Gestionnaire de la base vectorielle ChromaDB.

    Fournit une interface haut niveau pour les opérations CRUD
    sur la base de documents vectorisés.
    """

    COLLECTION_NAME = "legal_documents"

    def __init__(self, embeddings_manager: Optional[EmbeddingsManager] = None):
        """
        Initialise le gestionnaire de la base vectorielle.

        Args:
            embeddings_manager: Gestionnaire d'embeddings à utiliser
        """
        self.paths = get_paths()
        self.settings = get_settings()
        self.embeddings_manager = embeddings_manager or EmbeddingsManager()
        self._vectorstore: Optional[Chroma] = None

        logger.info(
            f"VectorStoreManager initialisé "
            f"(persistence: {self.paths.vectorstore})"
        )

    @property
    def vectorstore(self) -> Chroma:
        """
        Retourne l'instance ChromaDB (lazy loading avec persistence).

        La base est créée ou chargée depuis le disque selon son existence.
        """
        if self._vectorstore is None:
            persist_directory = str(self.paths.vectorstore)

            self._vectorstore = Chroma(
                collection_name=self.COLLECTION_NAME,
                embedding_function=self.embeddings_manager.get_langchain_embeddings(),
                persist_directory=persist_directory,
            )

            logger.debug("Base vectorielle chargée/créée")

        return self._vectorstore

    def add_documents(
        self,
        documents: list[Document],
        source_id: Optional[str] = None
    ) -> list[str]:
        """
        Ajoute des documents à la base vectorielle.

        Args:
            documents: Liste de documents LangChain à indexer
            source_id: Identifiant optionnel de la source (nom du fichier)

        Returns:
            list[str]: Liste des IDs générés pour les documents
        """
        try:
            # Ajout des métadonnées de source si spécifié
            if source_id:
                for doc in documents:
                    doc.metadata["source_id"] = source_id

            logger.info(f"Ajout de {len(documents)} documents à la base")
            ids = self.vectorstore.add_documents(documents)

            logger.info(f"{len(ids)} documents ajoutés avec succès")
            return ids

        except Exception as e:
            logger.error(f"Erreur lors de l'ajout des documents: {str(e)}")
            raise

    def search(
        self,
        query: str,
        k: Optional[int] = None,
        filter_dict: Optional[dict] = None
    ) -> list[Document]:
        """
        Recherche les documents les plus pertinents pour une requête.

        Args:
            query: Requête de recherche en langage naturel
            k: Nombre de résultats à retourner (défaut: config)
            filter_dict: Filtres optionnels sur les métadonnées

        Returns:
            list[Document]: Documents les plus pertinents
        """
        try:
            k = k or self.settings.retriever_k

            logger.debug(f"Recherche: '{query[:50]}...' (k={k})")

            results = self.vectorstore.similarity_search(
                query=query,
                k=k,
                filter=filter_dict,
            )

            logger.debug(f"{len(results)} documents trouvés")
            return results

        except Exception as e:
            logger.error(f"Erreur lors de la recherche: {str(e)}")
            raise

    def search_with_scores(
        self,
        query: str,
        k: Optional[int] = None
    ) -> list[tuple[Document, float]]:
        """
        Recherche avec scores de similarité.

        Utile pour filtrer les résultats peu pertinents
        ou afficher la confiance de la recherche.

        Args:
            query: Requête de recherche
            k: Nombre de résultats

        Returns:
            list[tuple[Document, float]]: Documents avec leurs scores
        """
        try:
            k = k or self.settings.retriever_k

            results = self.vectorstore.similarity_search_with_score(
                query=query,
                k=k,
            )

            return results

        except Exception as e:
            logger.error(f"Erreur lors de la recherche avec scores: {str(e)}")
            raise

    def delete_by_source(self, source_id: str) -> bool:
        """
        Supprime tous les documents d'une source spécifique.

        Args:
            source_id: Identifiant de la source à supprimer

        Returns:
            bool: True si la suppression a réussi
        """
        try:
            logger.info(f"Suppression des documents de la source: {source_id}")

            # Récupération des IDs des documents à supprimer
            collection = self.vectorstore._collection
            results = collection.get(
                where={"source_id": source_id},
                include=[]
            )

            if results["ids"]:
                collection.delete(ids=results["ids"])
                logger.info(f"{len(results['ids'])} documents supprimés")
                return True

            logger.warning(f"Aucun document trouvé pour la source: {source_id}")
            return False

        except Exception as e:
            logger.error(f"Erreur lors de la suppression: {str(e)}")
            raise

    def get_all_sources(self) -> list[str]:
        """
        Récupère la liste de toutes les sources indexées.

        Returns:
            list[str]: Liste des identifiants de sources uniques
        """
        try:
            collection = self.vectorstore._collection
            results = collection.get(include=["metadatas"])

            sources = set()
            for metadata in results.get("metadatas", []):
                if metadata and "source_id" in metadata:
                    sources.add(metadata["source_id"])

            return sorted(list(sources))

        except Exception as e:
            logger.error(f"Erreur lors de la récupération des sources: {str(e)}")
            return []

    def get_document_count(self) -> int:
        """
        Retourne le nombre total de documents dans la base.

        Returns:
            int: Nombre de documents indexés
        """
        try:
            return self.vectorstore._collection.count()
        except Exception as e:
            logger.error(f"Erreur lors du comptage: {str(e)}")
            return 0

    def get_retriever(self, k: Optional[int] = None):
        """
        Retourne un retriever LangChain pour intégration avec les chaînes.

        Args:
            k: Nombre de documents à récupérer

        Returns:
            VectorStoreRetriever: Retriever configuré
        """
        k = k or self.settings.retriever_k
        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )

    def clear(self) -> bool:
        """
        Supprime tous les documents de la base.

        Returns:
            bool: True si la suppression a réussi
        """
        try:
            logger.warning("Suppression de tous les documents de la base")

            collection = self.vectorstore._collection
            results = collection.get(include=[])

            if results["ids"]:
                collection.delete(ids=results["ids"])
                logger.info(f"{len(results['ids'])} documents supprimés")

            return True

        except Exception as e:
            logger.error(f"Erreur lors de la suppression totale: {str(e)}")
            return False
