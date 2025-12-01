"""
Tests unitaires pour le module de logging.
"""

import pytest
import sys
import logging
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from io import StringIO

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.utils.logger import ColoredFormatter, setup_logging, get_logger


class TestColoredFormatter:
    """Tests pour la classe ColoredFormatter."""

    def test_colors_constant(self):
        """Vérifie que les couleurs sont définies."""
        assert logging.DEBUG in ColoredFormatter.COLORS
        assert logging.INFO in ColoredFormatter.COLORS
        assert logging.WARNING in ColoredFormatter.COLORS
        assert logging.ERROR in ColoredFormatter.COLORS
        assert logging.CRITICAL in ColoredFormatter.COLORS

    def test_reset_constant(self):
        """Vérifie la constante RESET."""
        assert ColoredFormatter.RESET == "\033[0m"

    def test_format_debug_message(self):
        """Vérifie le formatage d'un message DEBUG."""
        formatter = ColoredFormatter(fmt="%(levelname)s - %(message)s")

        record = logging.LogRecord(
            name="test",
            level=logging.DEBUG,
            pathname="",
            lineno=0,
            msg="Debug message",
            args=(),
            exc_info=None
        )

        result = formatter.format(record)

        # Devrait contenir la couleur cyan
        assert "\033[36m" in result
        assert "Debug message" in result

    def test_format_info_message(self):
        """Vérifie le formatage d'un message INFO."""
        formatter = ColoredFormatter(fmt="%(levelname)s - %(message)s")

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Info message",
            args=(),
            exc_info=None
        )

        result = formatter.format(record)

        # Devrait contenir la couleur verte
        assert "\033[32m" in result

    def test_format_warning_message(self):
        """Vérifie le formatage d'un message WARNING."""
        formatter = ColoredFormatter(fmt="%(levelname)s - %(message)s")

        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="",
            lineno=0,
            msg="Warning message",
            args=(),
            exc_info=None
        )

        result = formatter.format(record)

        # Devrait contenir la couleur jaune
        assert "\033[33m" in result

    def test_format_error_message(self):
        """Vérifie le formatage d'un message ERROR."""
        formatter = ColoredFormatter(fmt="%(levelname)s - %(message)s")

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="Error message",
            args=(),
            exc_info=None
        )

        result = formatter.format(record)

        # Devrait contenir la couleur rouge
        assert "\033[31m" in result

    def test_format_critical_message(self):
        """Vérifie le formatage d'un message CRITICAL."""
        formatter = ColoredFormatter(fmt="%(levelname)s - %(message)s")

        record = logging.LogRecord(
            name="test",
            level=logging.CRITICAL,
            pathname="",
            lineno=0,
            msg="Critical message",
            args=(),
            exc_info=None
        )

        result = formatter.format(record)

        # Devrait contenir la couleur magenta
        assert "\033[35m" in result

    def test_format_unknown_level(self):
        """Vérifie le formatage d'un niveau inconnu."""
        formatter = ColoredFormatter(fmt="%(levelname)s - %(message)s")

        record = logging.LogRecord(
            name="test",
            level=99,  # Niveau inconnu
            pathname="",
            lineno=0,
            msg="Unknown level message",
            args=(),
            exc_info=None
        )

        result = formatter.format(record)

        # Devrait utiliser RESET comme couleur par défaut
        assert "Unknown level message" in result

    def test_format_includes_reset(self):
        """Vérifie que le RESET est inclus dans le formatage."""
        formatter = ColoredFormatter(fmt="%(levelname)s - %(message)s")

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Message",
            args=(),
            exc_info=None
        )

        result = formatter.format(record)

        assert "\033[0m" in result


class TestSetupLogging:
    """Tests pour la fonction setup_logging."""

    def setup_method(self):
        """Nettoyage du logger avant chaque test."""
        # Supprimer les handlers existants
        logger = logging.getLogger("legal_rag")
        logger.handlers = []

    @patch('app.utils.logger.get_settings')
    def test_setup_logging_creates_handler(self, mock_settings):
        """Vérifie que setup_logging crée un handler."""
        mock_settings.return_value = Mock(log_level="INFO")

        setup_logging()

        logger = logging.getLogger("legal_rag")
        assert len(logger.handlers) >= 1

    @patch('app.utils.logger.get_settings')
    def test_setup_logging_sets_level_info(self, mock_settings):
        """Vérifie le niveau INFO."""
        mock_settings.return_value = Mock(log_level="INFO")

        setup_logging()

        logger = logging.getLogger("legal_rag")
        assert logger.level == logging.INFO

    @patch('app.utils.logger.get_settings')
    def test_setup_logging_sets_level_debug(self, mock_settings):
        """Vérifie le niveau DEBUG."""
        mock_settings.return_value = Mock(log_level="DEBUG")

        setup_logging()

        logger = logging.getLogger("legal_rag")
        assert logger.level == logging.DEBUG

    @patch('app.utils.logger.get_settings')
    def test_setup_logging_sets_level_warning(self, mock_settings):
        """Vérifie le niveau WARNING."""
        mock_settings.return_value = Mock(log_level="WARNING")

        setup_logging()

        logger = logging.getLogger("legal_rag")
        assert logger.level == logging.WARNING

    @patch('app.utils.logger.get_settings')
    def test_setup_logging_uses_colored_formatter(self, mock_settings):
        """Vérifie l'utilisation du ColoredFormatter."""
        mock_settings.return_value = Mock(log_level="INFO")

        setup_logging()

        logger = logging.getLogger("legal_rag")
        handler = logger.handlers[0]
        assert isinstance(handler.formatter, ColoredFormatter)

    @patch('app.utils.logger.get_settings')
    def test_setup_logging_reduces_third_party_noise(self, mock_settings):
        """Vérifie la réduction du bruit des bibliothèques tierces."""
        mock_settings.return_value = Mock(log_level="DEBUG")

        setup_logging()

        # Vérifier que les loggers tiers sont à WARNING
        assert logging.getLogger("httpx").level == logging.WARNING
        assert logging.getLogger("chromadb").level == logging.WARNING
        assert logging.getLogger("openai").level == logging.WARNING


class TestGetLogger:
    """Tests pour la fonction get_logger."""

    def test_get_logger_with_name(self):
        """Vérifie la création d'un logger avec nom."""
        logger = get_logger("test_module")

        assert logger.name == "legal_rag.test_module"

    def test_get_logger_without_name(self):
        """Vérifie la création d'un logger sans nom."""
        logger = get_logger()

        assert logger.name == "legal_rag"

    def test_get_logger_with_none(self):
        """Vérifie la création d'un logger avec None."""
        logger = get_logger(None)

        assert logger.name == "legal_rag"

    def test_get_logger_returns_logger_instance(self):
        """Vérifie que get_logger retourne un Logger."""
        logger = get_logger("test")

        assert isinstance(logger, logging.Logger)

    def test_get_logger_cached(self):
        """Vérifie que les loggers sont mis en cache."""
        logger1 = get_logger("cached_test")
        logger2 = get_logger("cached_test")

        assert logger1 is logger2

    def test_get_logger_different_names(self):
        """Vérifie que des noms différents donnent des loggers différents."""
        logger1 = get_logger("module1")
        logger2 = get_logger("module2")

        assert logger1 is not logger2
        assert logger1.name != logger2.name

    def test_get_logger_can_log(self):
        """Vérifie qu'on peut logger avec le logger retourné."""
        logger = get_logger("functional_test")

        # Ne devrait pas lever d'exception
        logger.debug("Debug test")
        logger.info("Info test")
        logger.warning("Warning test")
        logger.error("Error test")


class TestLoggerIntegration:
    """Tests d'intégration pour le système de logging."""

    def setup_method(self):
        """Configuration avant chaque test."""
        # Reset le logger
        logger = logging.getLogger("legal_rag")
        logger.handlers = []
        logger.setLevel(logging.NOTSET)

    @patch('app.utils.logger.get_settings')
    def test_full_logging_pipeline(self, mock_settings):
        """Test du pipeline complet de logging."""
        mock_settings.return_value = Mock(log_level="DEBUG")

        # Setup
        setup_logging()

        # Obtenir un logger
        logger = get_logger("integration_test")

        # Capturer la sortie
        string_io = StringIO()
        handler = logging.StreamHandler(string_io)
        handler.setFormatter(ColoredFormatter(fmt="%(levelname)s - %(message)s"))
        logger.addHandler(handler)

        # Logger un message
        logger.info("Test message")

        # Vérifier la sortie
        output = string_io.getvalue()
        assert "Test message" in output

    @patch('app.utils.logger.get_settings')
    def test_multiple_loggers_same_root(self, mock_settings):
        """Vérifie que plusieurs loggers partagent la même config."""
        mock_settings.return_value = Mock(log_level="WARNING")

        setup_logging()

        logger1 = get_logger("service1")
        logger2 = get_logger("service2")

        # Les deux devraient être des enfants de legal_rag
        assert logger1.name.startswith("legal_rag.")
        assert logger2.name.startswith("legal_rag.")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
