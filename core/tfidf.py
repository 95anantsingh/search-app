import os
import re
from nltk.stem import PorterStemmer
from pandas import DataFrame
from pickle import dump, load
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from scipy.sparse import save_npz, load_npz, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from .base_search import BaseSearch
from .offers_db import OfferDBSession


class TFIDFSearch(BaseSearch):
    def __init__(
        self,
        session: OfferDBSession | None = None,
        cache: str = "models/tfidf",
    ) -> None:
        self._session = session if session else OfferDBSession()
        self.matrix_cache = os.path.join(cache, "matrix.npz")
        self.vectorizer_cache = os.path.join(cache, "vectorizer.pickle")

        os.makedirs(cache, exist_ok=True)
        if os.path.exists(self.matrix_cache) and os.path.exists(self.vectorizer_cache):
            # print("Loading TF-IDF matrix from file...")
            with open(self.vectorizer_cache, "rb") as file:
                self.vectorizer = load(file)
            self.matrix = load_npz(self.matrix_cache)
        else:
            self.vectorizer = TfidfVectorizer()
            self.matrix = self.create_tfidf_matrix()

    @property
    def session(self) -> OfferDBSession:
        return self._session

    @session.setter
    def session(self, value: OfferDBSession) -> None:
        self._session = value

    def create_tfidf_matrix(self) -> csr_matrix:
        targets = self.session.get_targets()
        targets = [self.tokenize(target) for target in targets]
        self.vectorizer.fit(targets)
        matrix = self.vectorizer.transform(targets)
        save_npz(self.matrix_cache, matrix)
        with open(self.vectorizer_cache, "wb") as file:
            dump(self.vectorizer, file)

        return matrix

    def tokenize(self, text: str):
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
        tokens = word_tokenize(text)
        ps = PorterStemmer()
        stop_words = set(stopwords.words("english"))
        tokens = [ps.stem(word) for word in tokens if not word in stop_words]
        text = " ".join(tokens)
        return text

    def get_scores(self, query: str) -> DataFrame:
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
