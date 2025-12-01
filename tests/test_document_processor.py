"""
Tests unitaires pour le processeur de documents.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from io import StringIO

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.documents import Document

from app.services.document_processor import DocumentProcessor, ProcessingResult
from app.services.file_handler import FileInfo


class TestProcessingResult:
    """Tests pour la dataclass ProcessingResult."""

    def test_processing_result_success(self):
        """Vérifie la création d'un résultat de succès."""
        result = ProcessingResult(
            success=True,
            source_id="doc.txt",
            chunks_count=5,
            message="Document indexé avec succès"
        )
        assert result.success is True
        assert result.source_id == "doc.txt"
        assert result.chunks_count == 5

    def test_processing_result_failure(self):
        """Vérifie la création d'un résultat d'échec."""
        result = ProcessingResult(
            success=False,
            source_id="doc.txt",
            chunks_count=0,
            message="Erreur lors du traitement"
        )
        assert result.success is False
        assert result.chunks_count == 0


class TestDocumentProcessorInit:
    """Tests pour l'initialisation du DocumentProcessor."""

    @patch('app.services.document_processor.get_settings')
    @patch('app.services.document_processor.VectorStoreManager')
    @patch('app.services.document_processor.FileHandler')
    def test_init_with_defaults(self, mock_file_handler, mock_vectorstore, mock_settings):
        """Vérifie l'initialisation avec les valeurs par défaut."""
        mock_settings.return_value = Mock(
            chunk_size=1000,
            chunk_overlap=200
        )

        processor = DocumentProcessor()

        assert processor.vectorstore is not None
        assert processor.file_handler is not None

    @patch('app.services.document_processor.get_settings')
    def test_init_with_custom_dependencies(self, mock_settings):
        """Vérifie l'initialisation avec des dépendances personnalisées."""
        mock_settings.return_value = Mock(
            chunk_size=500,
            chunk_overlap=100
        )

        mock_vectorstore = Mock()
        mock_file_handler = Mock()

        processor = DocumentProcessor(
            vectorstore=mock_vectorstore,
            file_handler=mock_file_handler
        )

        assert processor.vectorstore == mock_vectorstore
        assert processor.file_handler == mock_file_handler

    @patch('app.services.document_processor.get_settings')
    def test_separators_constant(self, mock_settings):
        """Vérifie les séparateurs pour le découpage."""
        mock_settings.return_value = Mock(chunk_size=1000, chunk_overlap=200)

        assert DocumentProcessor.SEPARATORS == ["\n\n", "\n", ". ", " ", ""]


class TestDocumentProcessorProcessFile:
    """Tests pour la méthode process_file."""

    def setup_method(self):
        """Configuration avant chaque test."""
        self.settings_patcher = patch('app.services.document_processor.get_settings')
        self.mock_settings = self.settings_patcher.start()
        self.mock_settings.return_value = Mock(chunk_size=1000, chunk_overlap=200)

        self.mock_vectorstore = Mock()
        self.mock_file_handler = Mock()

    def teardown_method(self):
        """Nettoyage après chaque test."""
        self.settings_patcher.stop()

    def test_process_file_success(self):
        """Vérifie le traitement réussi d'un fichier."""
        processor = DocumentProcessor(
            vectorstore=self.mock_vectorstore,
            file_handler=self.mock_file_handler
        )

        file_info = FileInfo(
            name="test.txt",
            path=Path("/tmp/test.txt"),
            extension=".txt",
            size=1000,
            hash="abc123",
            uploaded_at=datetime.now()
        )

        self.mock_file_handler.get_file_content.return_value = "Contenu du document juridique avec beaucoup de texte pour tester."
        self.mock_vectorstore.add_documents.return_value = ["id1"]

        result = processor.process_file(file_info)

        assert result.success is True
        assert result.source_id == "test.txt"
        assert result.chunks_count >= 1

    def test_process_file_empty_content(self):
        """Vérifie le comportement avec un fichier vide."""
        processor = DocumentProcessor(
            vectorstore=self.mock_vectorstore,
            file_handler=self.mock_file_handler
        )

        file_info = FileInfo(
            name="empty.txt",
            path=Path("/tmp/empty.txt"),
            extension=".txt",
            size=0,
            hash="empty",
            uploaded_at=datetime.now()
        )

        self.mock_file_handler.get_file_content.return_value = ""

        result = processor.process_file(file_info)

        assert result.success is False
        assert "vide" in result.message

    def test_process_file_whitespace_only(self):
        """Vérifie le comportement avec uniquement des espaces."""
        processor = DocumentProcessor(
            vectorstore=self.mock_vectorstore,
            file_handler=self.mock_file_handler
        )

        file_info = FileInfo(
            name="spaces.txt",
            path=Path("/tmp/spaces.txt"),
            extension=".txt",
            size=10,
            hash="spaces",
            uploaded_at=datetime.now()
        )

        self.mock_file_handler.get_file_content.return_value = "   \n\n   \t   "

        result = processor.process_file(file_info)

        assert result.success is False

    def test_process_file_too_short_after_cleaning(self):
        """Vérifie le comportement avec contenu trop court après nettoyage."""
        processor = DocumentProcessor(
            vectorstore=self.mock_vectorstore,
            file_handler=self.mock_file_handler
        )

        file_info = FileInfo(
            name="short.txt",
            path=Path("/tmp/short.txt"),
            extension=".txt",
            size=20,
            hash="short",
            uploaded_at=datetime.now()
        )

        self.mock_file_handler.get_file_content.return_value = "AB"

        result = processor.process_file(file_info)

        assert result.success is False
        assert "trop court" in result.message

    def test_process_file_handles_exception(self):
        """Vérifie la gestion des exceptions."""
        processor = DocumentProcessor(
            vectorstore=self.mock_vectorstore,
            file_handler=self.mock_file_handler
        )

        file_info = FileInfo(
            name="error.txt",
            path=Path("/tmp/error.txt"),
            extension=".txt",
            size=100,
            hash="error",
            uploaded_at=datetime.now()
        )

        self.mock_file_handler.get_file_content.side_effect = Exception("Read Error")

        result = processor.process_file(file_info)

        assert result.success is False
        assert "Erreur" in result.message


class TestDocumentProcessorExtraction:
    """Tests pour l'extraction de texte."""

    def setup_method(self):
        """Configuration avant chaque test."""
        self.settings_patcher = patch('app.services.document_processor.get_settings')
        self.mock_settings = self.settings_patcher.start()
        self.mock_settings.return_value = Mock(chunk_size=1000, chunk_overlap=200)

        self.mock_vectorstore = Mock()
        self.mock_file_handler = Mock()

        self.processor = DocumentProcessor(
            vectorstore=self.mock_vectorstore,
            file_handler=self.mock_file_handler
        )

    def teardown_method(self):
        """Nettoyage après chaque test."""
        self.settings_patcher.stop()

    def test_extract_text_from_txt(self):
        """Vérifie l'extraction depuis un fichier .txt."""
        file_info = FileInfo(
            name="doc.txt",
            path=Path("/tmp/doc.txt"),
            extension=".txt",
            size=100,
            hash="txt",
            uploaded_at=datetime.now()
        )

        self.mock_file_handler.get_file_content.return_value = "Contenu texte brut"

        result = self.processor._extract_text(file_info)

        assert result == "Contenu texte brut"

    def test_extract_text_from_html(self):
        """Vérifie l'extraction depuis un fichier .html."""
        file_info = FileInfo(
            name="doc.html",
            path=Path("/tmp/doc.html"),
            extension=".html",
            size=200,
            hash="html",
            uploaded_at=datetime.now()
        )

        html_content = """
        <html>
        <head><title>Test</title></head>
        <body>
            <h1>Titre</h1>
            <p>Paragraphe de contenu.</p>
            <script>var x = 1;</script>
        </body>
        </html>
        """
        self.mock_file_handler.get_file_content.return_value = html_content

        result = self.processor._extract_text(file_info)

        assert "Titre" in result
        assert "Paragraphe de contenu" in result
        assert "script" not in result.lower() or "var x" not in result

    def test_extract_from_html_removes_scripts(self):
        """Vérifie la suppression des scripts du HTML."""
        html = "<script>alert('test')</script><p>Contenu</p>"

        result = self.processor._extract_from_html(html)

        assert "alert" not in result
        assert "Contenu" in result

    def test_extract_from_html_removes_styles(self):
        """Vérifie la suppression des styles du HTML."""
        html = "<style>.class { color: red; }</style><p>Contenu</p>"

        result = self.processor._extract_from_html(html)

        assert "color" not in result
        assert "Contenu" in result

    def test_extract_text_raises_if_file_not_found(self):
        """Vérifie l'erreur si fichier non trouvé."""
        file_info = FileInfo(
            name="missing.txt",
            path=Path("/tmp/missing.txt"),
            extension=".txt",
            size=0,
            hash="missing",
            uploaded_at=datetime.now()
        )

        self.mock_file_handler.get_file_content.return_value = None

        with pytest.raises(ValueError) as exc_info:
            self.processor._extract_text(file_info)

        assert "Impossible de lire" in str(exc_info.value)


class TestDocumentProcessorCSVExtraction:
    """Tests pour l'extraction depuis CSV."""

    def setup_method(self):
        """Configuration avant chaque test."""
        self.settings_patcher = patch('app.services.document_processor.get_settings')
        self.mock_settings = self.settings_patcher.start()
        self.mock_settings.return_value = Mock(chunk_size=1000, chunk_overlap=200)

        self.mock_vectorstore = Mock()
        self.mock_file_handler = Mock()

        self.processor = DocumentProcessor(
            vectorstore=self.mock_vectorstore,
            file_handler=self.mock_file_handler
        )

    def teardown_method(self):
        """Nettoyage après chaque test."""
        self.settings_patcher.stop()

    @patch('app.services.document_processor.pd.read_csv')
    def test_extract_from_csv(self, mock_read_csv):
        """Vérifie l'extraction depuis un CSV."""
        import pandas as pd

        df = pd.DataFrame({
            'Nom': ['Alice', 'Bob'],
            'Age': [25, 30]
        })
        mock_read_csv.return_value = df

        result = self.processor._extract_from_csv(Path("/tmp/data.csv"))

        assert "Nom: Alice" in result
        assert "Age: 25" in result
        assert "Nom: Bob" in result

    @patch('app.services.document_processor.pd.read_csv')
    def test_extract_from_csv_handles_nan(self, mock_read_csv):
        """Vérifie la gestion des valeurs NaN."""
        import pandas as pd
        import numpy as np

        df = pd.DataFrame({
            'Col1': ['Value1', np.nan],
            'Col2': [np.nan, 'Value2']
        })
        mock_read_csv.return_value = df

        result = self.processor._extract_from_csv(Path("/tmp/data.csv"))

        # Les NaN ne devraient pas apparaître dans le texte
        assert "nan" not in result.lower()


class TestDocumentProcessorSplitting:
    """Tests pour le découpage en chunks."""

    def setup_method(self):
        """Configuration avant chaque test."""
        self.settings_patcher = patch('app.services.document_processor.get_settings')
        self.mock_settings = self.settings_patcher.start()
        self.mock_settings.return_value = Mock(chunk_size=100, chunk_overlap=20)

        self.mock_vectorstore = Mock()
        self.mock_file_handler = Mock()

        self.processor = DocumentProcessor(
            vectorstore=self.mock_vectorstore,
            file_handler=self.mock_file_handler
        )

    def teardown_method(self):
        """Nettoyage après chaque test."""
        self.settings_patcher.stop()

    def test_split_text_creates_documents(self):
        """Vérifie la création de documents LangChain."""
        text = "Premier paragraphe.\n\nDeuxième paragraphe.\n\nTroisième paragraphe."

        docs = self.processor._split_text(text, "source.txt")

        assert len(docs) >= 1
        assert all(isinstance(doc, Document) for doc in docs)

    def test_split_text_adds_metadata(self):
        """Vérifie l'ajout des métadonnées aux chunks."""
        text = "Contenu du document."

        docs = self.processor._split_text(text, "source.txt")

        for doc in docs:
            assert "source" in doc.metadata
            assert "chunk_index" in doc.metadata
            assert "total_chunks" in doc.metadata
            assert doc.metadata["source"] == "source.txt"

    def test_split_text_preserves_chunk_order(self):
        """Vérifie que l'ordre des chunks est préservé."""
        text = "A" * 500  # Assez long pour plusieurs chunks

        docs = self.processor._split_text(text, "source.txt")

        for i, doc in enumerate(docs):
            assert doc.metadata["chunk_index"] == i


class TestDocumentProcessorRemove:
    """Tests pour la suppression de documents."""

    def setup_method(self):
        """Configuration avant chaque test."""
        self.settings_patcher = patch('app.services.document_processor.get_settings')
        self.mock_settings = self.settings_patcher.start()
        self.mock_settings.return_value = Mock(chunk_size=1000, chunk_overlap=200)

        self.mock_vectorstore = Mock()
        self.mock_file_handler = Mock()

        self.processor = DocumentProcessor(
            vectorstore=self.mock_vectorstore,
            file_handler=self.mock_file_handler
        )

    def teardown_method(self):
        """Nettoyage après chaque test."""
        self.settings_patcher.stop()

    def test_remove_document_success(self):
        """Vérifie la suppression réussie."""
        self.mock_vectorstore.delete_by_source.return_value = True
        self.mock_file_handler.delete_file.return_value = True

        result = self.processor.remove_document("doc.txt")

        assert result is True
        self.mock_vectorstore.delete_by_source.assert_called_once_with("doc.txt")
        self.mock_file_handler.delete_file.assert_called_once_with("doc.txt")

    def test_remove_document_handles_error(self):
        """Vérifie la gestion des erreurs de suppression."""
        self.mock_vectorstore.delete_by_source.side_effect = Exception("Delete Error")

        result = self.processor.remove_document("doc.txt")

        assert result is False


class TestDocumentProcessorGetIndexed:
    """Tests pour la récupération des documents indexés."""

    def setup_method(self):
        """Configuration avant chaque test."""
        self.settings_patcher = patch('app.services.document_processor.get_settings')
        self.mock_settings = self.settings_patcher.start()
        self.mock_settings.return_value = Mock(chunk_size=1000, chunk_overlap=200)

        self.mock_vectorstore = Mock()
        self.mock_file_handler = Mock()

        self.processor = DocumentProcessor(
            vectorstore=self.mock_vectorstore,
            file_handler=self.mock_file_handler
        )

    def teardown_method(self):
        """Nettoyage après chaque test."""
        self.settings_patcher.stop()

    def test_get_indexed_documents(self):
        """Vérifie la récupération des documents indexés."""
        self.mock_vectorstore.get_all_sources.return_value = ["doc1.txt", "doc2.txt"]

        file_info1 = FileInfo(
            name="doc1.txt",
            path=Path("/tmp/doc1.txt"),
            extension=".txt",
            size=100,
            hash="hash1",
            uploaded_at=datetime.now()
        )
        self.mock_file_handler.list_files.return_value = [file_info1]

        docs = self.processor.get_indexed_documents()

        assert len(docs) == 2
        assert docs[0]["name"] == "doc1.txt"
        assert docs[0]["indexed"] is True

    def test_get_indexed_documents_missing_file(self):
        """Vérifie la gestion des fichiers manquants."""
        self.mock_vectorstore.get_all_sources.return_value = ["missing.txt"]
        self.mock_file_handler.list_files.return_value = []

        docs = self.processor.get_indexed_documents()

        assert len(docs) == 1
        assert docs[0]["size"] == 0


class TestDocumentProcessorReindex:
    """Tests pour la réindexation."""

    def setup_method(self):
        """Configuration avant chaque test."""
        self.settings_patcher = patch('app.services.document_processor.get_settings')
        self.mock_settings = self.settings_patcher.start()
        self.mock_settings.return_value = Mock(chunk_size=1000, chunk_overlap=200)

        self.mock_vectorstore = Mock()
        self.mock_file_handler = Mock()

        self.processor = DocumentProcessor(
            vectorstore=self.mock_vectorstore,
            file_handler=self.mock_file_handler
        )

    def teardown_method(self):
        """Nettoyage après chaque test."""
        self.settings_patcher.stop()

    def test_reindex_all_clears_and_reprocesses(self):
        """Vérifie que reindex_all vide et retraite."""
        file_info = FileInfo(
            name="doc.txt",
            path=Path("/tmp/doc.txt"),
            extension=".txt",
            size=100,
            hash="hash",
            uploaded_at=datetime.now()
        )
        self.mock_file_handler.list_files.return_value = [file_info]

        # Mock process_file pour éviter le traitement réel
        with patch.object(self.processor, 'process_file') as mock_process:
            mock_process.return_value = ProcessingResult(
                success=True,
                source_id="doc.txt",
                chunks_count=1,
                message="OK"
            )

            results = self.processor.reindex_all()

        self.mock_vectorstore.clear.assert_called_once()
        assert len(results) == 1

    def test_reindex_all_empty_folder(self):
        """Vérifie reindex_all avec dossier vide."""
        self.mock_file_handler.list_files.return_value = []

        results = self.processor.reindex_all()

        self.mock_vectorstore.clear.assert_called_once()
        assert results == []


class TestDocumentProcessorNoChunks:
    """Tests pour le cas où le chunking échoue."""

    def setup_method(self):
        """Configuration avant chaque test."""
        self.settings_patcher = patch('app.services.document_processor.get_settings')
        self.mock_settings = self.settings_patcher.start()
        self.mock_settings.return_value = Mock(chunk_size=1000, chunk_overlap=200)

        self.mock_vectorstore = Mock()
        self.mock_file_handler = Mock()

    def teardown_method(self):
        """Nettoyage après chaque test."""
        self.settings_patcher.stop()

    def test_process_file_no_chunks(self):
        """Vérifie le comportement quand le chunking ne produit rien."""
        processor = DocumentProcessor(
            vectorstore=self.mock_vectorstore,
            file_handler=self.mock_file_handler
        )

        file_info = FileInfo(
            name="empty_chunks.txt",
            path=Path("/tmp/empty_chunks.txt"),
            extension=".txt",
            size=100,
            hash="hash",
            uploaded_at=datetime.now()
        )

        # Le contenu doit avoir plus de 50 caractères après nettoyage pour atteindre la ligne de chunks
        long_content = "Contenu valide pour test qui doit être suffisamment long pour passer la vérification de longueur minimum de 50 caractères"
        self.mock_file_handler.get_file_content.return_value = long_content

        # Mock _split_text pour retourner une liste vide
        with patch.object(processor, '_split_text') as mock_split:
            mock_split.return_value = []
            result = processor.process_file(file_info)

        assert result.success is False
        # Le message devrait mentionner "chunks" car le texte est assez long
        assert "chunks" in result.message.lower()


class TestDocumentProcessorCSVErrors:
    """Tests pour les erreurs CSV."""

    def setup_method(self):
        """Configuration avant chaque test."""
        self.settings_patcher = patch('app.services.document_processor.get_settings')
        self.mock_settings = self.settings_patcher.start()
        self.mock_settings.return_value = Mock(chunk_size=1000, chunk_overlap=200)

        self.mock_vectorstore = Mock()
        self.mock_file_handler = Mock()

        self.processor = DocumentProcessor(
            vectorstore=self.mock_vectorstore,
            file_handler=self.mock_file_handler
        )

    def teardown_method(self):
        """Nettoyage après chaque test."""
        self.settings_patcher.stop()

    @patch('app.services.document_processor.pd.read_csv')
    def test_extract_from_csv_all_encodings_fail(self, mock_read_csv):
        """Vérifie le comportement quand tous les encodages échouent."""
        mock_read_csv.side_effect = UnicodeDecodeError('utf-8', b'', 0, 1, 'test')

        with pytest.raises(ValueError) as exc_info:
            self.processor._extract_from_csv(Path("/tmp/bad.csv"))

        assert "décoder" in str(exc_info.value).lower()

    @patch('app.services.document_processor.pd.read_csv')
    def test_extract_from_csv_read_error(self, mock_read_csv):
        """Vérifie la gestion des erreurs de lecture CSV."""
        mock_read_csv.side_effect = Exception("CSV Error")

        with pytest.raises(Exception) as exc_info:
            self.processor._extract_from_csv(Path("/tmp/error.csv"))

        assert "CSV Error" in str(exc_info.value)


class TestDocumentProcessorHTMLFallback:
    """Tests pour le fallback HTML."""

    def setup_method(self):
        """Configuration avant chaque test."""
        self.settings_patcher = patch('app.services.document_processor.get_settings')
        self.mock_settings = self.settings_patcher.start()
        self.mock_settings.return_value = Mock(chunk_size=1000, chunk_overlap=200)

        self.mock_vectorstore = Mock()
        self.mock_file_handler = Mock()

        self.processor = DocumentProcessor(
            vectorstore=self.mock_vectorstore,
            file_handler=self.mock_file_handler
        )

    def teardown_method(self):
        """Nettoyage après chaque test."""
        self.settings_patcher.stop()

    @patch('app.services.document_processor.BeautifulSoup')
    def test_extract_from_html_fallback(self, mock_bs):
        """Vérifie le fallback quand BeautifulSoup échoue."""
        mock_bs.side_effect = Exception("Parse error")

        html = "<p>Simple text</p>"
        result = self.processor._extract_from_html(html)

        # Devrait utiliser le fallback remove_html_tags
        assert "Simple text" in result


class TestDocumentProcessorExtractText:
    """Tests pour l'extraction de texte selon le format."""

    def setup_method(self):
        """Configuration avant chaque test."""
        self.settings_patcher = patch('app.services.document_processor.get_settings')
        self.mock_settings = self.settings_patcher.start()
        self.mock_settings.return_value = Mock(chunk_size=1000, chunk_overlap=200)

        self.mock_vectorstore = Mock()
        self.mock_file_handler = Mock()

        self.processor = DocumentProcessor(
            vectorstore=self.mock_vectorstore,
            file_handler=self.mock_file_handler
        )

    def teardown_method(self):
        """Nettoyage après chaque test."""
        self.settings_patcher.stop()

    def test_extract_text_csv_calls_extract_from_csv(self):
        """Vérifie que _extract_text appelle _extract_from_csv pour les fichiers CSV."""
        file_info = FileInfo(
            name="data.csv",
            path=Path("/tmp/data.csv"),
            extension=".csv",
            size=100,
            hash="hash",
            uploaded_at=datetime.now()
        )

        self.mock_file_handler.get_file_content.return_value = "some,csv,content"

        with patch.object(self.processor, '_extract_from_csv') as mock_csv:
            mock_csv.return_value = "Extracted CSV content"
            result = self.processor._extract_text(file_info)

        mock_csv.assert_called_once_with(file_info.path)
        assert result == "Extracted CSV content"

    def test_extract_text_returns_no_chunk_error(self):
        """Vérifie le retour d'erreur quand aucun chunk n'est produit."""
        processor = DocumentProcessor(
            vectorstore=self.mock_vectorstore,
            file_handler=self.mock_file_handler
        )

        file_info = FileInfo(
            name="empty.txt",
            path=Path("/tmp/empty.txt"),
            extension=".txt",
            size=100,
            hash="hash",
            uploaded_at=datetime.now()
        )

        self.mock_file_handler.get_file_content.return_value = "Contenu suffisamment long pour passer la vérification de longueur"

        # Mock _split_text pour retourner une liste vide après la vérification de longueur
        with patch.object(processor, '_split_text', return_value=[]):
            result = processor.process_file(file_info)

        assert result.success is False
        assert result.chunks_count == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
