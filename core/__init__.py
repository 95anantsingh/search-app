"""
The 'core' package contains modules and utilities for creating and managing a database of offer data,
performing different types of searches on the database, and combining multiple search methods in a hybrid
approach.

Modules:
- 'data_processor': Provides functionality to create a database from CSV files with brand, category, and offer data.
- 'offers_db': Contains the OfferDBSession class and OffersTable model for database interaction.
- 'tfidf': Implements TF-IDF search for textual data.
- 'bm25': Implements BM25 search for textual data.
- 'neural': Provides a NeuralSearch class for searching using neural models and defines related constants.
- 'hybrid': Introduces a HybridSearch class for combining multiple search methods and defines related constants.

For detailed documentation of each module and its functionalities, refer to the respective module's docstrings.
"""

from .data_processor import create_db
from .offers_db import OfferDBSession, OffersTable
from .tfidf import TFIDFSearch
from .bm25 import BM25Search
from .neural import NeuralSearch, RETRIEVAL_MODELS, SCORE_TYPES
from .hybrid import HybridSearch, MEAN_TYPES, NORM_TYPES
