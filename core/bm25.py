import os
import re
from typing import List
from pandas import DataFrame
from pickle import dump, load
from rank_bm25 import BM25Okapi
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from .base_search import BaseSearch
from .offers_db import OfferDBSession


class BM25Search(BaseSearch):
    """
    BM25Search is a class that implements a search algorithm based on BM25 similarity.

    Attributes:
    - session (OfferDBSession): An optional database session for offer data.
    - model_cache (str): The path to the cache file where the BM25 model is stored.
    - model (BM25Okapi): The BM25 model for similarity scoring.

    Methods:
    - __init__(session: OfferDBSession | None = None, cache: str = "vectors/bm25") -> None: Constructor to initialize the BM25Search object.
    - session (property) -> OfferDBSession: Property to access the database session.
    - session (setter) -> None: Setter method for the database session.
    - tokenize(text: str) -> List[str]: Tokenize a given text into a list of words.
    - create_model() -> BM25Okapi: Create and store a BM25 model for similarity scoring.
    - get_scores(query: str) -> DataFrame: Calculate similarity scores between a query and offers in the database.
    """

    def __init__(
        self,
        session: OfferDBSession | None = None,
        cache: str = "vectors/bm25",
    ) -> None:
        """
        Initialize the BM25Search object.

        Args:
        - session (OfferDBSession | None, optional): An optional database session for offer data. Defaults to None.
        - cache (str, optional): The path to the cache file where the BM25 model is stored. Defaults to "vectors/bm25".
        """
        self._session = session if session else OfferDBSession()
        self.model_cache = os.path.join(cache, "model.pickle")
        os.makedirs(cache, exist_ok=True)
        if os.path.exists(self.model_cache):
            with open(self.model_cache, "rb") as file:
                self.model = load(file)
        else:
            self.model = self.create_model()

    @property
    def session(self) -> OfferDBSession:
        """
        Get the database session for offer data.

        Returns:
        OfferDBSession: An instance of the database session for offer data.
        """
        return self._session

    @session.setter
    def session(self, value: OfferDBSession) -> None:
        """
        Set the database session for offer data.

        Args:
        - value (OfferDBSession): An instance of the database session.
        """
        self._session = value

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize a given text into a list of words.

        Args:
        - text (str): The text to tokenize.

        Returns:
        List[str]: A list of tokenized words.
        """
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
        tokens = word_tokenize(text)
        ps = PorterStemmer()
        stop_words = set(stopwords.words("english"))
        tokens = [ps.stem(word) for word in tokens if not word in stop_words]

        return tokens

    def create_model(self) -> BM25Okapi:
        """
        Create and store a BM25 model for similarity scoring.

        Returns:
        BM25Okapi: A BM25 model for similarity scoring.
        """
        targets = self.session.get_targets()
        targets = [self.tokenize(target) for target in targets]
        model = BM25Okapi(targets)
        with open(self.model_cache, "wb") as file:
            dump(model, file)
        return model

    def get_scores(self, query: str) -> DataFrame:
        """
        Calculate similarity scores between a query and offers in the database.

        Args:
        - query (str): The query string for similarity scoring.

        Returns:
        DataFrame: A DataFrame containing similarity scores for offers.
        """
        # Tokenize
        query = self.tokenize(query.lower())

        # Calculate similarity between user input and each offer
        scores = self.model.get_scores(query)

        results = DataFrame.from_dict({"SCORE": scores})
        results = results.sort_values(by="SCORE", ascending=False)
        return results
