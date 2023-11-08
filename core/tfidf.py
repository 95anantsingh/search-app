import os
import re
from pandas import DataFrame
from pickle import dump, load
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import linear_kernel
from scipy.sparse import save_npz, load_npz, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from .base_search import BaseSearch
from .offers_db import OfferDBSession


class TFIDFSearch(BaseSearch):
    """
    This class provides a search algorithm using TF-IDF (Term Frequency-Inverse Document Frequency) to find relevant offers.

    Attributes:
    - session (OfferDBSession): An instance of the database session for offer data.
    - matrix_cache (str): The path to the cache file for the TF-IDF matrix.
    - vectorizer_cache (str): The path to the cache file for the TF-IDF vectorizer.
    - vectorizer (TfidfVectorizer): The TF-IDF vectorizer used for text transformation.
    - matrix (csr_matrix): The TF-IDF matrix representing the offer data.

    Methods:
    - __init__(session: OfferDBSession | None = None, cache: str = "vectors/tfidf") -> None: Initialize the TFIDFSearch instance.
    - create_tfidf_matrix() -> csr_matrix: Create the TF-IDF matrix and cache it.
    - tokenize(text: str) -> str: Tokenize and preprocess the input text for TF-IDF processing.
    - get_scores(query: str) -> DataFrame: Get similarity scores for offers based on the provided query.

    """

    def __init__(
        self,
        session: OfferDBSession | None = None,
        cache: str = "vectors/tfidf",
    ) -> None:
        """
        Initialize the TFIDFSearch instance.

        Args:
        - session (OfferDBSession | None, optional): An instance of the database session for offer data. Defaults to None.
        - cache (str, optional): The path to the cache directory for storing TF-IDF related files. Defaults to "vectors/tfidf".
        """
        self._session = session if session else OfferDBSession()
        self.matrix_cache = os.path.join(cache, "matrix.npz")
        self.vectorizer_cache = os.path.join(cache, "vectorizer.pickle")

        os.makedirs(cache, exist_ok=True)
        if os.path.exists(self.matrix_cache) and os.path.exists(self.vectorizer_cache):
            with open(self.vectorizer_cache, "rb") as file:
                self.vectorizer = load(file)
            self.matrix = load_npz(self.matrix_cache)
        else:
            self.vectorizer = TfidfVectorizer()
            self.matrix = self.create_tfidf_matrix()

    @property
    def session(self) -> OfferDBSession:
        """
        An attribute representing the database session for offer data.

        Returns:
        OfferDBSession: An instance of the database session for offer data.
        """
        return self._session

    @session.setter
    def session(self, value: OfferDBSession) -> None:
        """
        Set the database session for offer data.

        Args:
        - value (OfferDBSession): An instance of the database session for offer data.
        """
        self._session = value

    def create_tfidf_matrix(self) -> csr_matrix:
        """
        Create the TF-IDF matrix and cache it.

        Returns:
        csr_matrix: The TF-IDF matrix containing offer representations.
        """
        targets = self.session.get_targets()
        targets = [self.tokenize(target) for target in targets]
        self.vectorizer.fit(targets)
        matrix = self.vectorizer.transform(targets)
        save_npz(self.matrix_cache, matrix)
        with open(self.vectorizer_cache, "wb") as file:
            dump(self.vectorizer, file)

        return matrix

    def tokenize(self, text: str) -> str:
        """
        Tokenize and preprocess the input text for TF-IDF processing.

        Args:
        - text (str): The input text to tokenize and preprocess.

        Returns:
        str: The preprocessed and tokenized text.
        """
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
        tokens = word_tokenize(text)
        ps = PorterStemmer()
        stop_words = set(stopwords.words("english"))
        tokens = [ps.stem(word) for word in tokens if not word in stop_words]
        text = " ".join(tokens)
        return text

    def get_scores(self, query: str) -> DataFrame:
        """
        Get similarity scores for offers based on the provided query.

        Args:
        - query (str): The query string to match against offers.

        Returns:
        DataFrame: A DataFrame containing similarity scores for offers.
        """
        # Tokenize
        query = self.tokenize(query.lower())

        # Calculate the TF-IDF vector
        vector = self.vectorizer.transform([query])

        # Calculate similarity between user input and each offer
        # scores = cosine_similarity(vector, self.matrix)
        scores = linear_kernel(vector, self.matrix)

        results = DataFrame.from_dict({"SCORE": scores[0]})
        results = results.sort_values(by="SCORE", ascending=False)
        return results