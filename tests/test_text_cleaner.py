"""
Tests unitaires pour le module de nettoyage de texte.
"""

import pytest
import sys
from pathlib import Path

# Ajout du chemin du projet pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.utils.text_cleaner import TextCleaner


class TestTextCleaner:
    """Tests pour la classe TextCleaner."""

    def setup_method(self):
        """Configuration avant chaque test."""
        self.cleaner = TextCleaner()

    def test_clean_removes_multiple_spaces(self):
        """Vérifie la normalisation des espaces multiples."""
        text = "Hello    world   test"
        result = self.cleaner.clean(text)
        assert "    " not in result
        assert "Hello world test" == result

    def test_clean_removes_urls(self):
        """Vérifie la suppression des URLs."""
        text = "Voir le site https://example.com pour plus d'infos"
        result = self.cleaner.clean(text)
        assert "https://example.com" not in result

    def test_clean_handles_emails(self):
        """Vérifie le remplacement des emails."""
        text = "Contact: test@example.com"
        result = self.cleaner.clean(text)
        assert "test@example.com" not in result
        assert "[EMAIL]" in result

    def test_clean_empty_string(self):
        """Vérifie le comportement avec une chaîne vide."""
        result = self.cleaner.clean("")
        assert result == ""

    def test_clean_none_input(self):
        """Vérifie le comportement avec None."""
        result = self.cleaner.clean(None)
        assert result == ""

    def test_remove_html_tags(self):
        """Vérifie la suppression des balises HTML."""
        html = "<p>Hello <strong>World</strong></p>"
        result = TextCleaner.remove_html_tags(html)
        assert "<p>" not in result
        assert "<strong>" not in result
        assert "Hello" in result
        assert "World" in result

    def test_extract_sentences(self):
        """Vérifie le découpage en phrases."""
        text = "Première phrase. Deuxième phrase! Troisième phrase?"
        sentences = TextCleaner.extract_sentences(text)
        assert len(sentences) == 3

    def test_clean_with_french_accents(self):
        """Vérifie la préservation des accents français."""
        text = "Café résumé été"
        result = self.cleaner.clean(text)
        assert "Café" in result
        assert "résumé" in result
        assert "été" in result

    def test_normalize_multiple_newlines(self):
        """Vérifie la normalisation des sauts de ligne multiples."""
        text = "Paragraphe 1\n\n\n\n\nParagraphe 2"
        result = self.cleaner.clean(text)
        assert "\n\n\n" not in result


class TestTextCleanerConfiguration:
    """Tests pour les options de configuration du TextCleaner."""

    def test_preserve_urls_when_disabled(self):
        """Vérifie la préservation des URLs quand l'option est désactivée."""
        cleaner = TextCleaner(remove_urls=False)
        text = "Voir https://example.com"
        result = cleaner.clean(text)
        assert "https://example.com" in result

    def test_preserve_emails_when_disabled(self):
        """Vérifie la préservation des emails quand l'option est désactivée."""
        cleaner = TextCleaner(remove_emails=False)
        text = "Contact: test@example.com"
        result = cleaner.clean(text)
        assert "test@example.com" in result

    def test_remove_special_chars_when_enabled(self):
        """Vérifie la suppression des caractères spéciaux."""
        cleaner = TextCleaner(remove_special_chars=True)
        text = "Test @#$% special"
        result = cleaner.clean(text)
        assert "@#$%" not in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
