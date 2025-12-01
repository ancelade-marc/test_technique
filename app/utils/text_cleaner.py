"""
Module de nettoyage de texte.

Fournit des outils pour normaliser et nettoyer le contenu textuel
extrait des différents formats de documents.
"""

import re
import unicodedata
from typing import Optional


class TextCleaner:
    """
    Nettoyeur de texte pour le pré-traitement des documents.

    Applique une série de transformations pour normaliser le texte
    avant la vectorisation : suppression des caractères spéciaux,
    normalisation des espaces, etc.
    """

    # Patterns de nettoyage compilés pour performance
    MULTIPLE_SPACES = re.compile(r"\s+")
    MULTIPLE_NEWLINES = re.compile(r"\n{3,}")
    URL_PATTERN = re.compile(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    )
    EMAIL_PATTERN = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
    SPECIAL_CHARS = re.compile(r"[^\w\s\-.,;:!?()'\"\u00C0-\u017F]")

    def __init__(
        self,
        remove_urls: bool = True,
        remove_emails: bool = True,
        normalize_whitespace: bool = True,
        remove_special_chars: bool = False,
        min_line_length: int = 3,
    ):
        """
        Initialise le nettoyeur avec les options spécifiées.

        Args:
            remove_urls: Supprime les URLs du texte
            remove_emails: Supprime les adresses email
            normalize_whitespace: Normalise les espaces multiples
            remove_special_chars: Supprime les caractères spéciaux
            min_line_length: Longueur minimale d'une ligne à conserver
        """
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.normalize_whitespace = normalize_whitespace
        self.remove_special_chars = remove_special_chars
        self.min_line_length = min_line_length

    def clean(self, text: Optional[str]) -> str:
        """
        Nettoie le texte en appliquant toutes les transformations configurées.

        Args:
            text: Texte brut à nettoyer

        Returns:
            str: Texte nettoyé et normalisé
        """
        if not text:
            return ""

        # Normalisation Unicode (NFC)
        cleaned = unicodedata.normalize("NFC", text)

        # Suppression des URLs si activé
        if self.remove_urls:
            cleaned = self.URL_PATTERN.sub(" ", cleaned)

        # Suppression des emails si activé
        if self.remove_emails:
            cleaned = self.EMAIL_PATTERN.sub("[EMAIL]", cleaned)

        # Suppression des caractères spéciaux si activé
        if self.remove_special_chars:
            cleaned = self.SPECIAL_CHARS.sub(" ", cleaned)

        # Normalisation des espaces
        if self.normalize_whitespace:
            cleaned = self._normalize_whitespace(cleaned)

        # Filtrage des lignes trop courtes
        if self.min_line_length > 0:
            cleaned = self._filter_short_lines(cleaned)

        return cleaned.strip()

    def _normalize_whitespace(self, text: str) -> str:
        """Normalise les espaces et sauts de ligne multiples."""
        # Remplace les espaces multiples par un seul
        text = self.MULTIPLE_SPACES.sub(" ", text)
        # Limite les sauts de ligne consécutifs à 2 maximum
        text = self.MULTIPLE_NEWLINES.sub("\n\n", text)
        return text

    def _filter_short_lines(self, text: str) -> str:
        """Supprime les lignes trop courtes (souvent du bruit)."""
        lines = text.split("\n")
        filtered = [
            line for line in lines
            if len(line.strip()) >= self.min_line_length or not line.strip()
        ]
        return "\n".join(filtered)

    @staticmethod
    def remove_html_tags(text: str) -> str:
        """
        Supprime les balises HTML du texte.

        Utilisé comme pré-traitement avant le nettoyage principal
        pour les fichiers HTML.

        Args:
            text: Texte contenant potentiellement du HTML

        Returns:
            str: Texte sans balises HTML
        """
        clean_pattern = re.compile(r"<[^>]+>")
        return clean_pattern.sub(" ", text)

    @staticmethod
    def extract_sentences(text: str) -> list[str]:
        """
        Split le texte en phrases.

        Args:
            text: Le texte en question

        Returns:
            list[str]: Un array de phrases
        """
        # Pattern pour détecter les fins de phrases
        sentence_pattern = re.compile(r"(?<=[.!?])\s+")
        sentences = sentence_pattern.split(text)
        return [s.strip() for s in sentences if s.strip()]
