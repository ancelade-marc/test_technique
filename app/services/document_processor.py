"""
Pipeline de traitement des documents.

Orchestre le nettoyage, le découpage et la vectorisation
des documents uploadés selon leur format.
"""

from typing import Optional
from pathlib import Path
from dataclasses import dataclass

from bs4 import BeautifulSoup
import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.core.vectorstore import VectorStoreManager
from app.services.file_handler import FileHandler, FileInfo
from app.utils.text_cleaner import TextCleaner
from app.utils.logger import get_logger


logger = get_logger("document_processor")


@dataclass
class ProcessingResult:
    """
    Résultat du traitement d'un document.

    Attributes:
        success: Indique si le traitement a réussi
        source_id: Identifiant de la source dans la base
        chunks_count: Nombre de chunks créés
        message: Message de statut ou d'erreur
    """
    success: bool
    source_id: str
    chunks_count: int
    message: str


class DocumentProcessor:
    """
    Processeur de documents pour le pipeline RAG.

    Gère le cycle complet : extraction du texte, nettoyage,
    découpage en chunks et indexation vectorielle.
    """

    # Séparateurs pour le découpage intelligent du texte
    SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(
        self,
        vectorstore: Optional[VectorStoreManager] = None,
        file_handler: Optional[FileHandler] = None,
    ):
        """
        Initialise le processeur de documents.

        Args:
            vectorstore: Gestionnaire de base vectorielle
            file_handler: Gestionnaire de fichiers
        """
        self.settings = get_settings()
        self.vectorstore = vectorstore or VectorStoreManager()
        self.file_handler = file_handler or FileHandler()

        # Configuration du text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            separators=self.SEPARATORS,
            length_function=len,
        )

        # Configuration du nettoyeur de texte
        self.cleaner = TextCleaner(
            remove_urls=True,
            remove_emails=True,
            normalize_whitespace=True,
            min_line_length=3,
        )

        logger.info(
            f"DocumentProcessor initialisé "
            f"(chunk_size={self.settings.chunk_size}, "
            f"overlap={self.settings.chunk_overlap})"
        )

    def process_file(self, file_info: FileInfo) -> ProcessingResult:
        """
        Traite un fichier complet : extraction, nettoyage et indexation.

        Args:
            file_info: Informations sur le fichier à traiter

        Returns:
            ProcessingResult: Résultat du traitement
        """
        try:
            logger.info(f"Traitement du fichier: {file_info.name}")

            # Extraction du texte selon le format
            raw_text = self._extract_text(file_info)

            if not raw_text or not raw_text.strip():
                return ProcessingResult(
                    success=False,
                    source_id=file_info.name,
                    chunks_count=0,
                    message="Le fichier est vide ou ne contient pas de texte exploitable"
                )

            # Nettoyage du texte
            cleaned_text = self.cleaner.clean(raw_text)

            if len(cleaned_text) < 50:
                return ProcessingResult(
                    success=False,
                    source_id=file_info.name,
                    chunks_count=0,
                    message="Le contenu nettoyé est trop court pour être exploitable"
                )

            # Découpage en chunks
            chunks = self._split_text(cleaned_text, file_info.name)

            if not chunks:
                return ProcessingResult(
                    success=False,
                    source_id=file_info.name,
                    chunks_count=0,
                    message="Impossible de découper le document en chunks"
                )

            # Indexation dans la base vectorielle
            self.vectorstore.add_documents(chunks, source_id=file_info.name)

            logger.info(
                f"Fichier traité avec succès: {file_info.name} "
                f"({len(chunks)} chunks)"
            )

            return ProcessingResult(
                success=True,
                source_id=file_info.name,
                chunks_count=len(chunks),
                message=f"Document indexé avec succès ({len(chunks)} segments)"
            )

        except Exception as e:
            logger.error(f"Erreur traitement {file_info.name}: {str(e)}")
            return ProcessingResult(
                success=False,
                source_id=file_info.name,
                chunks_count=0,
                message=f"Erreur lors du traitement: {str(e)}"
            )

    def remove_document(self, source_id: str) -> bool:
        """
        Supprime un document de l'index et du stockage.

        Args:
            source_id: Identifiant du document (nom du fichier)

        Returns:
            bool: True si la suppression a réussi
        """
        try:
            # Suppression de la base vectorielle
            self.vectorstore.delete_by_source(source_id)

            # Suppression du fichier physique
            self.file_handler.delete_file(source_id)

            logger.info(f"Document supprimé: {source_id}")
            return True

        except Exception as e:
            logger.error(f"Erreur suppression {source_id}: {str(e)}")
            return False

    def _extract_text(self, file_info: FileInfo) -> str:
        """
        Extrait le texte d'un fichier selon son format.

        Args:
            file_info: Informations sur le fichier

        Returns:
            str: Texte extrait
        """
        content = self.file_handler.get_file_content(file_info.name)

        if content is None:
            raise ValueError(f"Impossible de lire le fichier: {file_info.name}")

        # Traitement selon le format
        if file_info.extension == ".html":
            return self._extract_from_html(content)

        elif file_info.extension == ".csv":
            return self._extract_from_csv(file_info.path)

        else:  # .txt et autres formats texte
            return content

    def _extract_from_html(self, html_content: str) -> str:
        """
        Extrait le texte d'un contenu HTML.

        Utilise BeautifulSoup pour parser le HTML et extraire
        le texte de manière propre.

        Args:
            html_content: Contenu HTML brut

        Returns:
            str: Texte extrait
        """
        try:
            soup = BeautifulSoup(html_content, "lxml")

            # Suppression des éléments non textuels
            for element in soup(["script", "style", "meta", "link", "noscript"]):
                element.decompose()

            # Extraction du texte avec gestion des espaces
            text = soup.get_text(separator="\n", strip=True)

            return text

        except Exception as e:
            logger.warning(f"Erreur parsing HTML: {str(e)}")
            # Fallback : suppression simple des balises
            return TextCleaner.remove_html_tags(html_content)

    def _extract_from_csv(self, file_path: Path) -> str:
        """
        Extrait le texte d'un fichier CSV.

        Convertit chaque ligne en texte structuré pour
        faciliter la recherche.

        Args:
            file_path: Chemin vers le fichier CSV

        Returns:
            str: Texte extrait et formaté
        """
        try:
            # Tentative de lecture avec différents encodages
            for encoding in ["utf-8", "latin-1", "cp1252"]:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError("Impossible de décoder le fichier CSV")

            # Conversion en texte structuré
            text_parts = []

            for idx, row in df.iterrows():
                row_text = []
                for col, value in row.items():
                    if pd.notna(value):
                        row_text.append(f"{col}: {value}")

                if row_text:
                    text_parts.append(" | ".join(row_text))

            return "\n\n".join(text_parts)

        except Exception as e:
            logger.error(f"Erreur lecture CSV: {str(e)}")
            raise

    def _split_text(self, text: str, source_name: str) -> list[Document]:
        """
        Découpe le texte en chunks pour l'indexation.

        Args:
            text: Texte nettoyé à découper
            source_name: Nom de la source pour les métadonnées

        Returns:
            list[Document]: Liste des documents LangChain
        """
        chunks = self.text_splitter.split_text(text)

        documents = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": source_name,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                }
            )
            documents.append(doc)

        return documents

    def get_indexed_documents(self) -> list[dict]:
        """
        Récupère la liste des documents indexés avec statistiques.

        Returns:
            list[dict]: Informations sur chaque document indexé
        """
        sources = self.vectorstore.get_all_sources()
        documents_info = []

        for source in sources:
            # Récupération des fichiers correspondants
            files = self.file_handler.list_files()
            file_info = next(
                (f for f in files if f.name == source),
                None
            )

            documents_info.append({
                "name": source,
                "size": file_info.size if file_info else 0,
                "uploaded_at": file_info.uploaded_at if file_info else None,
                "indexed": True
            })

        return documents_info

    def reindex_all(self) -> list[ProcessingResult]:
        """
        Réindexe tous les fichiers du dossier documents.

        Utile après une mise à jour des paramètres de chunking
        ou pour reconstruire l'index.

        Returns:
            list[ProcessingResult]: Résultats pour chaque fichier
        """
        # Suppression de l'index existant
        self.vectorstore.clear()

        # Traitement de chaque fichier
        results = []
        files = self.file_handler.list_files()

        for file_info in files:
            result = self.process_file(file_info)
            results.append(result)

        return results
