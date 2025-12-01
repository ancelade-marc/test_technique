"""
Gestionnaire de fichiers pour l'upload et le stockage.

Gère la validation, le stockage et la suppression des fichiers
uploadés par les utilisateurs.
"""

import os
import hashlib
from pathlib import Path
from typing import Optional, BinaryIO
from dataclasses import dataclass
from datetime import datetime

from app.config import get_settings, get_paths
from app.utils.logger import get_logger


logger = get_logger("file_handler")


@dataclass
class FileInfo:
    """
    Informations sur un fichier uploadé.

    Attributes:
        name: Nom original du fichier
        path: Chemin complet sur le disque
        extension: Extension du fichier
        size: Taille en octets
        hash: Hash MD5 pour détecter les doublons
        uploaded_at: Date d'upload
    """
    name: str
    path: Path
    extension: str
    size: int
    hash: str
    uploaded_at: datetime


class FileHandler:
    """
    Gestionnaire pour les opérations sur les fichiers.

    Valide, stocke et supprime les fichiers uploadés
    en garantissant la sécurité et l'intégrité.
    """

    def __init__(self):
        """Initialise le gestionnaire de fichiers."""
        self.settings = get_settings()
        self.paths = get_paths()

        logger.info(f"FileHandler initialisé (dossier: {self.paths.documents})")

    def validate_file(self, filename: str, file_size: int) -> tuple[bool, str]:
        """
        Valide un fichier avant upload.

        Args:
            filename: Nom du fichier
            file_size: Taille en octets

        Returns:
            tuple[bool, str]: (est_valide, message_erreur)
        """
        # Vérification de l'extension
        extension = Path(filename).suffix.lower()

        if extension not in self.settings.allowed_extensions:
            allowed = ", ".join(self.settings.allowed_extensions)
            return False, f"Extension '{extension}' non autorisée. Formats acceptés: {allowed}"

        # Vérification de la taille
        max_size = self.settings.max_file_size_mb * 1024 * 1024

        if file_size > max_size:
            return False, f"Fichier trop volumineux. Taille max: {self.settings.max_file_size_mb} Mo"

        # Vérification du nom de fichier (sécurité)
        if not self._is_safe_filename(filename):
            return False, "Nom de fichier invalide ou potentiellement dangereux"

        return True, ""

    def save_file(self, file_content: BinaryIO, filename: str) -> FileInfo:
        """
        Sauvegarde un fichier uploadé.

        Args:
            file_content: Contenu binaire du fichier
            filename: Nom original du fichier

        Returns:
            FileInfo: Informations sur le fichier sauvegardé

        Raises:
            ValueError: Si le fichier existe déjà (même hash)
        """
        try:
            # Lecture du contenu
            content = file_content.read()
            file_hash = self._compute_hash(content)

            # Vérification des doublons
            if self._file_exists_by_hash(file_hash):
                raise ValueError(
                    f"Un fichier identique existe déjà dans la base"
                )

            # Sécurisation du nom de fichier
            safe_name = self._sanitize_filename(filename)
            file_path = self.paths.documents / safe_name

            # Gestion des conflits de noms
            file_path = self._get_unique_path(file_path)

            # Sauvegarde
            with open(file_path, "wb") as f:
                f.write(content)

            file_info = FileInfo(
                name=file_path.name,
                path=file_path,
                extension=file_path.suffix.lower(),
                size=len(content),
                hash=file_hash,
                uploaded_at=datetime.now()
            )

            logger.info(f"Fichier sauvegardé: {file_info.name} ({file_info.size} octets)")
            return file_info

        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde: {str(e)}")
            raise

    def delete_file(self, filename: str) -> bool:
        """
        Supprime un fichier du stockage.

        Args:
            filename: Nom du fichier à supprimer

        Returns:
            bool: True si la suppression a réussi
        """
        try:
            file_path = self.paths.documents / filename

            if not file_path.exists():
                logger.warning(f"Fichier non trouvé: {filename}")
                return False

            os.remove(file_path)
            logger.info(f"Fichier supprimé: {filename}")
            return True

        except Exception as e:
            logger.error(f"Erreur lors de la suppression: {str(e)}")
            return False

    def list_files(self) -> list[FileInfo]:
        """
        Liste tous les fichiers uploadés.

        Returns:
            list[FileInfo]: Liste des informations de fichiers
        """
        files = []

        for file_path in self.paths.documents.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.settings.allowed_extensions:
                try:
                    with open(file_path, "rb") as f:
                        content = f.read()
                        file_hash = self._compute_hash(content)

                    stat = file_path.stat()
                    files.append(FileInfo(
                        name=file_path.name,
                        path=file_path,
                        extension=file_path.suffix.lower(),
                        size=stat.st_size,
                        hash=file_hash,
                        uploaded_at=datetime.fromtimestamp(stat.st_mtime)
                    ))
                except Exception as e:
                    logger.warning(f"Erreur lecture fichier {file_path}: {str(e)}")

        return sorted(files, key=lambda f: f.uploaded_at, reverse=True)

    def get_file_content(self, filename: str) -> Optional[str]:
        """
        Lit le contenu textuel d'un fichier.

        Args:
            filename: Nom du fichier

        Returns:
            Optional[str]: Contenu du fichier ou None si erreur
        """
        try:
            file_path = self.paths.documents / filename

            if not file_path.exists():
                return None

            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()

        except UnicodeDecodeError:
            # Tentative avec autre encodage
            try:
                with open(file_path, "r", encoding="latin-1") as f:
                    return f.read()
            except Exception:
                return None

        except Exception as e:
            logger.error(f"Erreur lecture fichier: {str(e)}")
            return None

    def _compute_hash(self, content: bytes) -> str:
        """Calcule le hash MD5 du contenu."""
        return hashlib.md5(content).hexdigest()

    def _file_exists_by_hash(self, file_hash: str) -> bool:
        """Vérifie si un fichier avec le même hash existe."""
        for file_info in self.list_files():
            if file_info.hash == file_hash:
                return True
        return False

    def _is_safe_filename(self, filename: str) -> bool:
        """Vérifie que le nom de fichier est sûr."""
        # Pas de traversée de répertoire
        if ".." in filename or "/" in filename or "\\" in filename:
            return False

        # Pas de caractères spéciaux dangereux
        dangerous_chars = ["<", ">", ":", '"', "|", "?", "*"]
        return not any(char in filename for char in dangerous_chars)

    def _sanitize_filename(self, filename: str) -> str:
        """Nettoie le nom de fichier pour le stockage."""
        # Conservation du nom de base et de l'extension
        name = Path(filename).stem
        ext = Path(filename).suffix.lower()

        # Suppression des caractères spéciaux
        safe_name = "".join(
            c for c in name
            if c.isalnum() or c in ["-", "_", " "]
        )

        # Remplacement des espaces
        safe_name = safe_name.replace(" ", "_")

        return f"{safe_name}{ext}"

    def _get_unique_path(self, file_path: Path) -> Path:
        """Génère un chemin unique si le fichier existe."""
        if not file_path.exists():
            return file_path

        base = file_path.stem
        ext = file_path.suffix
        counter = 1

        while True:
            new_path = file_path.parent / f"{base}_{counter}{ext}"
            if not new_path.exists():
                return new_path
            counter += 1
