"""
Tests unitaires pour le gestionnaire de fichiers.
"""

import pytest
import sys
import os
import tempfile
import shutil
from pathlib import Path
from io import BytesIO
from datetime import datetime
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.file_handler import FileHandler, FileInfo


class TestFileHandler:
    """Tests pour la classe FileHandler."""

    def setup_method(self):
        """Configuration avant chaque test."""
        self.handler = FileHandler()

    def test_validate_file_valid_txt(self):
        """Vérifie la validation d'un fichier .txt valide."""
        is_valid, error = self.handler.validate_file("document.txt", 1024)
        assert is_valid is True
        assert error == ""

    def test_validate_file_valid_csv(self):
        """Vérifie la validation d'un fichier .csv valide."""
        is_valid, error = self.handler.validate_file("data.csv", 2048)
        assert is_valid is True
        assert error == ""

    def test_validate_file_valid_html(self):
        """Vérifie la validation d'un fichier .html valide."""
        is_valid, error = self.handler.validate_file("page.html", 4096)
        assert is_valid is True
        assert error == ""

    def test_validate_file_invalid_extension(self):
        """Vérifie le rejet d'une extension non autorisée."""
        is_valid, error = self.handler.validate_file("document.pdf", 1024)
        assert is_valid is False
        assert "non autorisée" in error

    def test_validate_file_too_large(self):
        """Vérifie le rejet d'un fichier trop volumineux."""
        large_size = 100 * 1024 * 1024  # 100 Mo
        is_valid, error = self.handler.validate_file("document.txt", large_size)
        assert is_valid is False
        assert "volumineux" in error

    def test_validate_file_path_traversal(self):
        """Vérifie la protection contre le path traversal."""
        is_valid, error = self.handler.validate_file("../etc/passwd", 100)
        assert is_valid is False
        assert "invalide" in error or "non autorisée" in error

    def test_validate_file_unsafe_filename(self):
        """Vérifie la validation d'un nom de fichier non sûr."""
        is_valid, error = self.handler.validate_file("file<script>.txt", 100)
        assert is_valid is False
        assert "invalide" in error

    def test_sanitize_filename(self):
        """Vérifie le nettoyage des noms de fichiers."""
        safe_name = self.handler._sanitize_filename("Mon Document (v2).txt")
        assert "/" not in safe_name
        assert "\\" not in safe_name
        assert " " not in safe_name  # Espaces remplacés par _

    def test_sanitize_filename_preserves_extension(self):
        """Vérifie la préservation de l'extension."""
        safe_name = self.handler._sanitize_filename("document.TXT")
        assert safe_name.endswith(".txt")  # Extension en minuscule

    def test_is_safe_filename_rejects_path_separators(self):
        """Vérifie le rejet des séparateurs de chemin."""
        assert self.handler._is_safe_filename("normal.txt") is True
        assert self.handler._is_safe_filename("../file.txt") is False
        assert self.handler._is_safe_filename("folder/file.txt") is False
        assert self.handler._is_safe_filename("folder\\file.txt") is False

    def test_is_safe_filename_rejects_dangerous_chars(self):
        """Vérifie le rejet des caractères dangereux."""
        assert self.handler._is_safe_filename("file<test>.txt") is False
        assert self.handler._is_safe_filename("file:test.txt") is False
        assert self.handler._is_safe_filename('file"test.txt') is False
        assert self.handler._is_safe_filename("file|test.txt") is False
        assert self.handler._is_safe_filename("file?test.txt") is False
        assert self.handler._is_safe_filename("file*test.txt") is False


class TestFileHandlerCompute:
    """Tests pour les fonctions de calcul du FileHandler."""

    def setup_method(self):
        """Configuration avant chaque test."""
        self.handler = FileHandler()

    def test_compute_hash_same_content(self):
        """Vérifie que le même contenu produit le même hash."""
        content = b"Test content"
        hash1 = self.handler._compute_hash(content)
        hash2 = self.handler._compute_hash(content)
        assert hash1 == hash2

    def test_compute_hash_different_content(self):
        """Vérifie que des contenus différents produisent des hash différents."""
        hash1 = self.handler._compute_hash(b"Content 1")
        hash2 = self.handler._compute_hash(b"Content 2")
        assert hash1 != hash2

    def test_compute_hash_is_md5(self):
        """Vérifie que le hash est un MD5 (32 caractères hex)."""
        hash_result = self.handler._compute_hash(b"Test")
        assert len(hash_result) == 32
        assert all(c in '0123456789abcdef' for c in hash_result)


class TestFileHandlerWithTempDir:
    """Tests nécessitant un dossier temporaire."""

    def setup_method(self):
        """Configuration avant chaque test."""
        self.temp_dir = tempfile.mkdtemp()

        self.paths_patcher = patch('app.services.file_handler.get_paths')
        self.settings_patcher = patch('app.services.file_handler.get_settings')

        self.mock_paths = self.paths_patcher.start()
        self.mock_settings = self.settings_patcher.start()

        mock_paths = Mock()
        mock_paths.documents = Path(self.temp_dir) / "documents"
        mock_paths.documents.mkdir(parents=True, exist_ok=True)

        mock_settings = Mock()
        mock_settings.allowed_extensions = [".txt", ".csv", ".html"]
        mock_settings.max_file_size_mb = 10

        self.mock_paths.return_value = mock_paths
        self.mock_settings.return_value = mock_settings

        self.handler = FileHandler()

    def teardown_method(self):
        """Nettoyage après chaque test."""
        self.paths_patcher.stop()
        self.settings_patcher.stop()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_file_creates_file(self):
        """Vérifie que save_file crée le fichier."""
        content = BytesIO(b"Contenu du fichier")

        file_info = self.handler.save_file(content, "test.txt")

        assert file_info.path.exists()
        assert file_info.name == "test.txt"

    def test_save_file_returns_correct_info(self):
        """Vérifie les informations retournées par save_file."""
        content = b"Contenu test"
        file_io = BytesIO(content)

        file_info = self.handler.save_file(file_io, "info_test.txt")

        assert file_info.size == len(content)
        assert file_info.extension == ".txt"
        assert file_info.hash is not None
        assert file_info.uploaded_at is not None

    def test_save_file_duplicate_raises(self):
        """Vérifie la détection des doublons."""
        content1 = BytesIO(b"Contenu identique")
        content2 = BytesIO(b"Contenu identique")

        self.handler.save_file(content1, "original.txt")

        with pytest.raises(ValueError) as exc_info:
            self.handler.save_file(content2, "duplicate.txt")

        assert "identique" in str(exc_info.value)

    def test_save_file_unique_path_conflict(self):
        """Vérifie la gestion des conflits de noms."""
        # Créer un fichier existant
        content1 = BytesIO(b"Contenu 1")
        self.handler.save_file(content1, "conflict.txt")

        # Essayer de sauvegarder avec le même nom mais contenu différent
        content2 = BytesIO(b"Contenu different 2")
        file_info = self.handler.save_file(content2, "conflict.txt")

        # Devrait avoir un nom différent
        assert file_info.name != "conflict.txt"
        assert "conflict" in file_info.name

    def test_delete_file_success(self):
        """Vérifie la suppression d'un fichier."""
        content = BytesIO(b"A supprimer")
        file_info = self.handler.save_file(content, "to_delete.txt")

        result = self.handler.delete_file(file_info.name)

        assert result is True
        assert not file_info.path.exists()

    def test_delete_file_not_found(self):
        """Vérifie la suppression d'un fichier inexistant."""
        result = self.handler.delete_file("inexistant.txt")

        assert result is False

    def test_delete_file_exception(self):
        """Vérifie la gestion des erreurs de suppression."""
        content = BytesIO(b"Test")
        file_info = self.handler.save_file(content, "error.txt")

        # Simuler une erreur en rendant le fichier non supprimable
        with patch('os.remove') as mock_remove:
            mock_remove.side_effect = PermissionError("Permission denied")
            result = self.handler.delete_file(file_info.name)

        assert result is False

    def test_list_files_empty(self):
        """Vérifie le listage d'un dossier vide."""
        files = self.handler.list_files()

        assert files == []

    def test_list_files_with_files(self):
        """Vérifie le listage avec des fichiers."""
        self.handler.save_file(BytesIO(b"File 1"), "file1.txt")
        self.handler.save_file(BytesIO(b"File 2 content"), "file2.txt")

        files = self.handler.list_files()

        assert len(files) == 2
        names = [f.name for f in files]
        assert "file1.txt" in names
        assert "file2.txt" in names

    def test_list_files_sorted_by_date(self):
        """Vérifie le tri par date de modification."""
        self.handler.save_file(BytesIO(b"Older"), "older.txt")
        import time
        time.sleep(0.1)
        self.handler.save_file(BytesIO(b"Newer content"), "newer.txt")

        files = self.handler.list_files()

        # Le plus récent devrait être en premier
        assert files[0].name == "newer.txt"

    def test_list_files_ignores_wrong_extensions(self):
        """Vérifie que les extensions non autorisées sont ignorées."""
        self.handler.save_file(BytesIO(b"Valid"), "valid.txt")

        # Créer manuellement un fichier avec extension non autorisée
        bad_file = self.mock_paths.return_value.documents / "invalid.pdf"
        bad_file.write_bytes(b"Invalid")

        files = self.handler.list_files()

        assert len(files) == 1
        assert files[0].name == "valid.txt"

    def test_get_file_content_utf8(self):
        """Vérifie la lecture de contenu UTF-8."""
        content = "Contenu avec accents: éàü"
        self.handler.save_file(BytesIO(content.encode('utf-8')), "utf8.txt")

        result = self.handler.get_file_content("utf8.txt")

        assert result == content

    def test_get_file_content_not_found(self):
        """Vérifie la lecture d'un fichier inexistant."""
        result = self.handler.get_file_content("inexistant.txt")

        assert result is None

    def test_get_file_content_latin1_fallback(self):
        """Vérifie le fallback vers latin-1 pour les encodages non-UTF8."""
        # Créer un fichier avec encodage latin-1
        content = "Contenu latin-1: café"
        file_path = self.mock_paths.return_value.documents / "latin1.txt"
        file_path.write_bytes(content.encode('latin-1'))

        result = self.handler.get_file_content("latin1.txt")

        assert "caf" in result

    def test_get_file_content_unreadable(self):
        """Vérifie la gestion des fichiers illisibles."""
        # Créer un fichier binaire qui ne peut pas être lu comme texte
        file_path = self.mock_paths.return_value.documents / "binary.txt"
        file_path.write_bytes(bytes([0x80, 0x81, 0x82, 0x83] * 100))

        # Patcher open pour simuler une erreur
        with patch('builtins.open') as mock_open:
            mock_open.side_effect = Exception("Read error")
            result = self.handler.get_file_content("binary.txt")

        assert result is None

    def test_get_unique_path_no_conflict(self):
        """Vérifie _get_unique_path sans conflit."""
        path = self.mock_paths.return_value.documents / "unique.txt"

        result = self.handler._get_unique_path(path)

        assert result == path

    def test_get_unique_path_with_conflicts(self):
        """Vérifie _get_unique_path avec conflits."""
        base_path = self.mock_paths.return_value.documents / "conflict.txt"

        # Créer des fichiers existants
        base_path.write_text("Original")
        (self.mock_paths.return_value.documents / "conflict_1.txt").write_text("Conflict 1")

        result = self.handler._get_unique_path(base_path)

        assert result.name == "conflict_2.txt"

    def test_file_exists_by_hash_true(self):
        """Vérifie _file_exists_by_hash quand le fichier existe."""
        content = b"Test content"
        self.handler.save_file(BytesIO(content), "existing.txt")

        file_hash = self.handler._compute_hash(content)
        result = self.handler._file_exists_by_hash(file_hash)

        assert result is True

    def test_file_exists_by_hash_false(self):
        """Vérifie _file_exists_by_hash quand le fichier n'existe pas."""
        result = self.handler._file_exists_by_hash("nonexistent_hash")

        assert result is False


class TestFileInfo:
    """Tests pour la dataclass FileInfo."""

    def test_file_info_creation(self):
        """Vérifie la création d'un FileInfo."""
        file_info = FileInfo(
            name="test.txt",
            path=Path("/tmp/test.txt"),
            extension=".txt",
            size=1024,
            hash="abc123",
            uploaded_at=datetime.now()
        )

        assert file_info.name == "test.txt"
        assert file_info.extension == ".txt"
        assert file_info.size == 1024


class TestFileHandlerListFilesErrors:
    """Tests pour les erreurs dans list_files."""

    def setup_method(self):
        """Configuration avant chaque test."""
        self.temp_dir = tempfile.mkdtemp()

        self.paths_patcher = patch('app.services.file_handler.get_paths')
        self.settings_patcher = patch('app.services.file_handler.get_settings')

        self.mock_paths = self.paths_patcher.start()
        self.mock_settings = self.settings_patcher.start()

        mock_paths = Mock()
        mock_paths.documents = Path(self.temp_dir) / "documents"
        mock_paths.documents.mkdir(parents=True, exist_ok=True)

        mock_settings = Mock()
        mock_settings.allowed_extensions = [".txt", ".csv", ".html"]
        mock_settings.max_file_size_mb = 10

        self.mock_paths.return_value = mock_paths
        self.mock_settings.return_value = mock_settings

        self.handler = FileHandler()

    def teardown_method(self):
        """Nettoyage après chaque test."""
        self.paths_patcher.stop()
        self.settings_patcher.stop()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_list_files_handles_file_read_error(self):
        """Vérifie la gestion des erreurs lors de la lecture de fichiers."""
        # Créer un fichier valide
        valid_file = self.mock_paths.return_value.documents / "valid.txt"
        valid_file.write_text("Valid content")

        # Créer un fichier qui va provoquer une erreur
        error_file = self.mock_paths.return_value.documents / "error.txt"
        error_file.write_text("Error content")

        # Mock open pour lever une exception sur le fichier error.txt
        original_open = open

        def mock_open_func(path, mode='r', **kwargs):
            if "error.txt" in str(path) and 'rb' in mode:
                raise PermissionError("Cannot read file")
            return original_open(path, mode, **kwargs)

        with patch('builtins.open', mock_open_func):
            files = self.handler.list_files()

        # Seul le fichier valide devrait être retourné
        assert len(files) == 1
        assert files[0].name == "valid.txt"


class TestFileHandlerGetContentFallbackError:
    """Tests pour les erreurs de fallback dans get_file_content."""

    def setup_method(self):
        """Configuration avant chaque test."""
        self.temp_dir = tempfile.mkdtemp()

        self.paths_patcher = patch('app.services.file_handler.get_paths')
        self.settings_patcher = patch('app.services.file_handler.get_settings')

        self.mock_paths = self.paths_patcher.start()
        self.mock_settings = self.settings_patcher.start()

        mock_paths = Mock()
        mock_paths.documents = Path(self.temp_dir) / "documents"
        mock_paths.documents.mkdir(parents=True, exist_ok=True)

        mock_settings = Mock()
        mock_settings.allowed_extensions = [".txt"]
        mock_settings.max_file_size_mb = 10

        self.mock_paths.return_value = mock_paths
        self.mock_settings.return_value = mock_settings

        self.handler = FileHandler()

    def teardown_method(self):
        """Nettoyage après chaque test."""
        self.paths_patcher.stop()
        self.settings_patcher.stop()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_get_file_content_latin1_fallback_fails(self):
        """Vérifie la gestion quand le fallback latin-1 échoue aussi."""
        # Créer un fichier avec du contenu binaire invalide
        file_path = self.mock_paths.return_value.documents / "binary.txt"
        file_path.write_bytes(bytes([0x80, 0x81, 0x82, 0x83] * 100))

        # Simuler l'échec de la lecture UTF-8 ET latin-1
        call_count = [0]
        original_open = open

        def mock_open_func(path, mode='r', **kwargs):
            if str(path).endswith("binary.txt") and 'r' in mode:
                call_count[0] += 1
                if call_count[0] == 1:
                    # Premier appel (UTF-8) - lever UnicodeDecodeError
                    raise UnicodeDecodeError('utf-8', b'', 0, 1, 'test')
                else:
                    # Deuxième appel (latin-1) - lever une autre exception
                    raise IOError("Cannot read file")
            return original_open(path, mode, **kwargs)

        with patch('builtins.open', mock_open_func):
            result = self.handler.get_file_content("binary.txt")

        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
