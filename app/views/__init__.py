"""
Module views - Interfaces utilisateur Streamlit.

Contient les composants de l'interface :
- Page Chat : Interface conversationnelle
- Page Documents : Gestion des documents
"""

from .chat import render_chat_page, render_sidebar
from .documents import render_documents_page

__all__ = ["render_chat_page", "render_documents_page", "render_sidebar"]
