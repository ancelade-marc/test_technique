# Assistant Juridique RAG

Application de chat intelligent basée sur RAG (Retrieval-Augmented Generation) pour le cabinet d'avocats Parenti & Associés.

## Fonctionnalités

- **Chat conversationnel** : Conversation en langage naturel avec historique persistant
- **Réponses sourcées** : Réponses basées exclusivement sur les documents indexés avec citation des sources
- **Gestion documentaire** : Upload, visualisation et suppression de documents (.txt, .csv, .html)
- **Streaming** : Réponses affichées en temps réel

## Architecture

### Structure du projet

```
POC/
├── app/                           # Code source principal
│   ├── __init__.py
│   ├── main.py                    # Point d'entrée Streamlit (UI multipage)
│   ├── config.py                  # Configuration centralisée (Pydantic Settings)
│   ├── core/                      # Composants RAG
│   │   ├── __init__.py
│   │   ├── llm.py                 # Client OpenAI (chat + streaming)
│   │   ├── embeddings.py          # Gestionnaire d'embeddings OpenAI
│   │   ├── vectorstore.py         # Interface ChromaDB (CRUD + recherche)
│   │   └── rag.py                 # Chaîne RAG (retrieval + génération)
│   ├── services/                  # Services métier
│   │   ├── __init__.py
│   │   ├── document_processor.py  # Pipeline de traitement documentaire
│   │   ├── file_handler.py        # Validation et gestion des fichiers
│   │   └── conversation.py        # Historique des conversations (JSON)
│   ├── utils/                     # Utilitaires
│   │   ├── __init__.py
│   │   ├── text_cleaner.py        # Nettoyage de texte (regex)
│   │   └── logger.py              # Logging coloré (colorlog)
│   └── views/                     # Pages Streamlit
│       ├── __init__.py
│       ├── chat.py                # Interface de conversation
│       └── documents.py           # Gestion des documents
│
├── data/                          # Données persistantes
│   ├── documents/                 # Fichiers uploadés
│   ├── vectorstore/               # Base ChromaDB (embeddings)
│   └── conversations/             # Historique JSON des conversations
│
├── tests/                         # Tests unitaires
│   ├── __init__.py
│   ├── test_config.py             # Tests configuration
│   ├── test_llm.py                # Tests client LLM
│   ├── test_embeddings.py         # Tests embeddings
│   ├── test_vectorstore.py        # Tests ChromaDB
│   ├── test_rag.py                # Tests chaîne RAG
│   ├── test_document_processor.py # Tests pipeline documents
│   ├── test_file_handler.py       # Tests gestionnaire fichiers
│   ├── test_conversation.py       # Tests historique
│   ├── test_text_cleaner.py       # Tests nettoyage texte
│   ├── test_logger.py             # Tests logging
│   └── test_integration.py        # Tests d'intégration
│
├── documents_test/                # Documents juridiques fictifs (Je l'ai ai créer avec ChatGPT en speed pour faire des tests) (12 fichiers)
│   ├── 01_contrat_prestation_services.txt
│   ├── 02_proces_verbal_reunion_lancement.txt
│   ├── 03_lettre_mise_en_demeure.txt
│   ├── 04_assignation_tribunal_commerce.txt
│   ├── 05_rapport_expertise_technique.txt
│   ├── 06_conclusions_avocat.txt
│   ├── 07_ordonnance_refere.txt
│   ├── 08_rapport_expertise_judiciaire.txt
│   ├── 09_jugement_tribunal_commerce.txt
│   ├── 10_protocole_transactionnel.txt
│   ├── 11_factures_honoraires.txt
│   └── 12_correspondances_emails.txt
│
├── .env                           # Variables d'environnement (pour la prod)
├── .env.example                   # Template de configuration (le model a utilisé, mais sans secret)
├── .gitignore
├── requirements.txt               # Dépendances Python (Normalement j'ai rien oublié)
└── README.md
```

### Diagramme des composants

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              STREAMLIT UI                                    │
│  ┌─────────────────────────────┐    ┌─────────────────────────────────────┐ │
│  │       Page Chat             │    │       Page Documents                │ │
│  │   (views/chat.py)           │    │   (views/documents.py)              │ │
│  └──────────────┬──────────────┘    └──────────────┬──────────────────────┘ │
└─────────────────┼──────────────────────────────────┼────────────────────────┘
                  │                                  │
                  ▼                                  ▼
┌─────────────────────────────────┐    ┌─────────────────────────────────────┐
│          RAGChain               │    │      DocumentProcessor              │
│      (core/rag.py)              │    │  (services/document_processor.py)   │
│                                 │    │                                     │
│  ┌───────────┐ ┌─────────────┐  │    │  1. Extraction (txt/csv/html)       │
│  │ Retrieval │ │ Generation  │  │    │  2. Nettoyage (TextCleaner)         │
│  └─────┬─────┘ └──────┬──────┘  │    │  3. Chunking (LangChain)            │
│        │              │         │    │  4. Indexation                      │
└────────┼──────────────┼─────────┘    │                                     │
         │              │              └──────────────┬──────────────────────┘
         ▼              ▼                             │
┌─────────────────┐ ┌─────────────────┐               │
│ VectorStore     │ │   LLMClient     │               │
│ (ChromaDB)      │ │   (OpenAI)      │               │
│                 │ │                 │               │
│ - search()      │ │ - chat()        │               │
│ - add_documents │ │ - stream()      │◄──────────────┘
│ - delete()      │ │                 │
└────────┬────────┘ └────────┬────────┘
         │                   │
         ▼                   ▼
┌─────────────────┐ ┌─────────────────┐
│ EmbeddingManager│ │  OpenAI API     │
│ (embeddings.py) │ │  (externe)      │
└────────┬────────┘ └─────────────────┘
         │
         ▼
┌─────────────────┐
│  OpenAI API     │
│  (embeddings)   │
└─────────────────┘
```

## Installation

### Prérequis

- Python 3.10+
- pip

### Installation


```bash
git clone https://github.com/AI-Sisters/test_technique
pip install -r requirements.txt
cp .env.example .env
#Éditer le fichier `.env` (Voir l'onglet configuration)
```

## Utilisation

### Lancer l'application

```bash
python -m streamlit run app/main.py
```

L'application sera accessible à l'adresse : http://localhost:8501 (Le port peux changé si vous ouvrez plusieurs streamlit en méme temps)


## Configuration

Les paramètres sont configurables via le fichier `.env` ou les variables d'environnement :

| Variable | Description | Exemple                |
|----------|-------------|------------------------|
| `OPENAI_API_KEY` | Clé API OpenAI | sk-proj-0JB7*****               |
| `LLM_MODEL` | Modèle de génération | gpt-4o-mini            |
| `EMBEDDING_MODEL` | Modèle d'embeddings | text-embedding-3-small |
| `CHUNK_SIZE` | Taille des chunks | 1000                   |
| `CHUNK_OVERLAP` | Chevauchement des chunks | 200                    |
| `RETRIEVER_K` | Nombre de documents contextuels | 4                      |
| `TEMPERATURE` | Température du LLM | 0.1                    |
| `LOG_LEVEL` | Niveau de logging | INFO                   |

## Stack Technique

- **Framework UI** : Streamlit
- **LLM** : OpenAI gpt-5-mini-2025-08-07
- **Embeddings** : OpenAI text-embedding-3-large
- **Base vectorielle** : ChromaDB pour la vécto en local
- **Orchestration RAG** : LangChain
- **Parsing HTML** : BeautifulSoup4
- **Configuration** : Pydantic Settings


## Tests

```bash
# Lancer tous les tests
pytest tests/ -v
```

Il y a encore moyen d’améliorer les tests ; j’ai seulement réalisé des tests assez basiques étant donné qu’il s’agit d’un POC.

## Licence

Projet développé exclusivement dans le cadre d’un test technique.  
La société Ancelade détient l’intégralité des droits patrimoniaux et moraux sur le contenu du présent dépôt.  
Toute reproduction, utilisation ou diffusion est strictement interdite sans autorisation écrite, à l’exception unique de son utilisation dans le cadre de l’évaluation de son auteur.



# Cahier des charges original ---

## **1. Contexte**

Emilia Parenti dirige un **cabinet d’avocats en droit des affaires**, situé à Paris.

Son équipe traite quotidiennement des documents confidentiels : contrats, litiges, notes internes, jurisprudences, etc. Emilia souhaite mettre en place un **chatbot interne sécurisé** pour faciliter l’accès à l'information juridique tout en garantissant la confidentialité.

Pour cette **preuve de concept (PoC)**, les documents utilisés sont **anonymisés** avec de faux noms, et le modèle de langage devra être **appelé via une API** sécurisée.

---

## **2. Objectif fonctionnel**

Le but du test est de concevoir une **application Streamlit** intégrant un système de **RAG (Retrieval-Augmented Generation)** basé sur des documents juridiques uploadés manuellement. L’objectif est de tester :

- ta capacité à **intégrer un LLM à une interface personnalisée**
- ta rigueur dans le **pré-traitement et vectorisation des documents**
- la qualité de ton **architecture logicielle**

### **2.1 Page 1 – Interface Chatbot**

Cette page permet à un collaborateur de :

- Poser une question à l’IA via une interface de chat
- Recevoir une réponse basée exclusivement sur les documents internes
- Créer une nouvelle conversation (💬 bonus : gestion d’un historique de conversations)

Toutes les réponses doivent être générées à partir des **documents vectorisés** (pas de génération hors corpus).

### **2.2 Page 2 – Gestion des documents**

Cette page permet à l’utilisateur de :

- **Uploader** des documents (`.txt`, `.csv`, `.html`)
- **Supprimer** des documents existants
- Automatiquement :
    - **Nettoyer les fichiers**
    - **Vectoriser** le contenu pour la base RAG

L’ensemble des documents doit être indexé pour que le modèle puisse s’y référer via un moteur vectoriel (type FAISS, Chroma, etc.).

---

## **3. Livrables & Environnement de Test**

### **3.1 Setup minimal**

Avant de commencer :

- Créer un environnement Python dédié
- Installer les dépendances nécessaires (ex : `streamlit`, `langchain`, `openai`, `chromadb`, etc.)
- Utiliser un modèle LLM disponible via API (`OpenAI (clef fournit)`, `Mistral`, `Claude`, etc.)
- Créer un dossier local ou une base vectorielle pour stocker les embeddings

### **3.2 Livrables attendus**

| Élément | Détail attendu |
| --- | --- |
| 💻 Application | Interface Streamlit fonctionnelle avec deux pages |
| 📦 Gestion de fichiers | Upload / delete + vectorisation automatisée |
| 🔗 Intégration LLM | API propre, sécurisé, réponse contrôlée via RAG |
| 🧹 Nettoyage des données | Pipeline de preprocessing simple et efficace |
| 📜 Historique (bonus) | Gestion conversationnelle avec suivi des échanges |
| 📁 README | Instructions claires pour exécuter le projet en local |
| 🔗 GitHub | Repo : https://github.com/AI-Sisters/test_technique |

---

## **4. Évaluation**

| Critère | Éléments attendus | Points |
| --- | --- | --- |
| ⚙️ Fonctionnalité | Upload, RAG, interface chat, vectorisation | 150pt |
| 🧱 Architecture | Structure du projet claire, code modulaire | 100pt |
| 🤖 Intégration IA | API LLM bien utilisée, réponses cohérentes | 75pt |
| 🧼 Données | Pipeline de nettoyage fiable et simple | 50pt |
| 🧪 Robustesse | Gestion des erreurs, logs, stabilité | 50pt |
| 🎯 UX | Interface fluide, logique d’usage claire | 50pt |
| 🎁 Bonus | Historique, logs, sécurité, documentation | +10 à +50pt |
| **Total** |  |  |

> 🧠 Tu peux utiliser tous les outils d’IA à disposition (ChatGPT, Copilot, etc.), mais la rigueur et la qualité de ton code primeront.
> 

---

## **5. Conclusion**

Ce test a pour but de valider :

- Ta capacité à **prototyper un outil complet en autonomie**
- Ton aisance avec les concepts de **RAG, vectorisation, et intégration LLM**
- Ta **rigueur technique** (structure, propreté du code, gestion des erreurs)
- Ton **agilité** : apprendre vite, aller à l’essentiel, mais proprement

Tu es libre dans tes choix techniques tant que tu **justifies ton raisonnement**, que ton code est **complet et maintenable**, et que le prototype **fonctionne avec fluidité**.