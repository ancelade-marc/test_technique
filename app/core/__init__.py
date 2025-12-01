"""
Module core - Composants fondamentaux du système RAG.

Ce module contient les briques de base pour le fonctionnement
du système de Retrieval-Augmented Generation :
- Client LLM
- Gestionnaire d'embeddings
- Interface avec la base vectorielle
- Chaîne RAG complète
"""

from .llm import LLMClient
from .embeddings import EmbeddingsManager
from .vectorstore import VectorStoreManager
from .rag import RAGChain

__all__ = ["LLMClient", "EmbeddingsManager", "VectorStoreManager", "RAGChain"]
