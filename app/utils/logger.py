"""
Module de logging structuré.

Fournit un système de logging uniforme pour toute l'application,
avec formatage cohérent et niveaux configurables.
"""

import logging
import sys
from typing import Optional
from functools import lru_cache

from app.config import get_settings


class ColoredFormatter(logging.Formatter):
    """
    Formatter avec coloration pour une meilleure lisibilité en console.

    Utilise les codes ANSI pour colorer les messages selon leur niveau.
    """

    COLORS = {
        logging.DEBUG: "\033[36m",     # Cyan
        logging.INFO: "\033[32m",      # Vert
        logging.WARNING: "\033[33m",   # Jaune
        logging.ERROR: "\033[31m",     # Rouge
        logging.CRITICAL: "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        """Formate le message avec la couleur appropriée."""
        color = self.COLORS.get(record.levelno, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging() -> None:
    """
    Configure le système de logging de l'application.

    Initialise un handler console avec formatage coloré
    et le niveau défini dans la configuration.
    """
    settings = get_settings()

    # Format du message
    log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Configuration du handler console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(
        ColoredFormatter(fmt=log_format, datefmt=date_format)
    )

    # Configuration du logger racine
    root_logger = logging.getLogger("legal_rag")
    root_logger.setLevel(getattr(logging, settings.log_level.upper()))
    root_logger.addHandler(console_handler)

    # Réduction du bruit des bibliothèques tierces
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


@lru_cache(maxsize=32)
def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Récupère un logger configuré pour le module spécifié.

    Args:
        name: Nom du module (utilisé comme suffixe du logger)

    Returns:
        logging.Logger: Logger configuré

    Example:
        >>> logger = get_logger("document_processor")
        >>> logger.info("Traitement du document en cours")
    """
    logger_name = f"legal_rag.{name}" if name else "legal_rag"
    return logging.getLogger(logger_name)
