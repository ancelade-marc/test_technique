"""
Configuration centralisée de l'application.

Ce module gère toutes les variables de configuration via Pydantic Settings,
permettant une validation automatique et un chargement sécurisé depuis
les variables d'environnement.
"""

from pathlib import Path
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """
    Configuration principale de l'application.

    Les valeurs sont chargées depuis les variables d'environnement
    ou le fichier .env à la racine du projet.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # OpenAI Configuration
    openai_api_key: str = Field(
        default="",
        description="Clé API OpenAI pour l'accès aux modèles"
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="Modèle LLM à utiliser pour la génération"
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Modèle d'embeddings pour la vectorisation"
    )

    # RAG Configuration
    chunk_size: int = Field(
        default=1000,
        description="Taille des chunks pour le découpage des documents"
    )
    chunk_overlap: int = Field(
        default=200,
        description="Chevauchement entre les chunks"
    )
    retriever_k: int = Field(
        default=4,
        description="Nombre de documents à récupérer pour le contexte"
    )

    # LLM Parameters
    temperature: float = Field(
        default=0.1,
        description="Température du modèle (créativité)"
    )
    max_tokens: int = Field(
        default=2048,
        description="Nombre maximum de tokens en sortie"
    )

    # Application Settings
    debug: bool = Field(
        default=False,
        description="Mode debug"
    )
    log_level: str = Field(
        default="INFO",
        description="Niveau de logging"
    )

    # Supported file types
    allowed_extensions: list[str] = Field(
        default=[".txt", ".csv", ".html"],
        description="Extensions de fichiers autorisées"
    )
    max_file_size_mb: int = Field(
        default=10,
        description="Taille maximale des fichiers en Mo"
    )


class Paths:
    """
    Gestionnaire des chemins de l'application.

    Centralise tous les chemins utilisés pour garantir
    la cohérence et faciliter les modifications.
    """

    def __init__(self):
        self.root = Path(__file__).parent.parent
        self.app = self.root / "app"
        self.data = self.root / "data"
        self.documents = self.data / "documents"
        self.vectorstore = self.data / "vectorstore"
        self.conversations = self.data / "conversations"

        # Création automatique des dossiers nécessaires
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Crée les dossiers s'ils n'existent pas."""
        for path in [self.documents, self.vectorstore, self.conversations]:
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """
    Récupère l'instance de configuration (singleton).

    Utilise le cache LRU pour éviter de recharger
    la configuration à chaque appel.

    Returns:
        Settings: Instance de configuration
    """
    return Settings()


@lru_cache
def get_paths() -> Paths:
    """
    Récupère l'instance des chemins (singleton).

    Returns:
        Paths: Instance du gestionnaire de chemins
    """
    return Paths()


# Constantes de l'interface
APP_TITLE = "Assistant Juridique"
APP_ICON = "balance_scale"
PAGE_CHAT = "Chat"
PAGE_DOCUMENTS = "Documents"

# Messages système
SYSTEM_PROMPT = """Tu es un assistant juridique expert travaillant pour le cabinet Parenti & Associés,
spécialisé en droit des affaires. Tu réponds uniquement en te basant sur les documents fournis dans le contexte.

Directives:
1. Base tes réponses EXCLUSIVEMENT sur les documents fournis
2. Si l'information n'est pas dans les documents, indique-le clairement
3. Cite les sources pertinentes dans ta réponse
4. Adopte un ton professionnel et précis
5. Structure ta réponse de manière claire

Contexte documentaire:
{context}

Question du collaborateur: {question}

Réponse:"""

NO_CONTEXT_MESSAGE = """Je n'ai pas trouvé d'information pertinente dans la base documentaire
pour répondre à cette question. Veuillez reformuler votre question ou vérifier que les documents
nécessaires ont été indexés."""
