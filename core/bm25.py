import os
import re
from pickle import dump, load
from pandas import DataFrame
from rank_bm25 import BM25Okapi
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from .base_search import BaseSearch
from .offers_db import OfferDBSession


class BM25Search(BaseSearch):

    def __init__(
        self,
        session: OfferDBSession | None = None,
        cache: str = "models/bm25",
    ) -> None:
        self._session = session if session else OfferDBSession()
        self.model_cache = os.path.join(cache, "model.pickle")
        os.makedirs(cache, exist_ok=True)
        if os.path.exists(self.model_cache):
            # print("Loading BM25 Model from file...")
            with open(self.model_cache, "rb") as file:
                self.model = load(file)
        else:
            self.model = self.create_model()

    @property
    def session(self) -> OfferDBSession:
        return self._session

    @session.setter
    def session(self, value: OfferDBSession) -> None:
        self._session = value

    def tokenize(self, text: str):
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
        tokens = word_tokenize(text)
        ps = PorterStemmer()
        stop_words = set(stopwords.words("english"))
        tokens = [ps.stem(word) for word in tokens if not word in stop_words]

        return tokens

    def create_model(self) -> BM25Okapi:
        targets = self.session.get_targets()
        targets = [self.tokenize(target) for target in targets]
        model = BM25Okapi(targets)
        with open(self.model_cache, "wb") as file:
            dump(model, file)
        return model

    def get_scores(self, query: str) -> DataFrame:
        # Tokenize
        query = self.tokenize(query.lower())

        # Calculate similarity between user input and each offer
        scores = self.model.get_scores(query)

        results = DataFrame.from_dict({"SCORE": scores})
        results = results.sort_values(by="SCORE", ascending=False)
        return results
