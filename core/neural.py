import os
import torch
import faiss
from pandas import DataFrame, Index
from sentence_transformers import SentenceTransformer

from .base_search import BaseSearch
from .offers_db import OfferDBSession


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RETRIVAL_MODELS = [
    "BAAI/bge-base-en-v1.5",
    "thenlper/gte-large",
    "llmrails/ember-v1",
    "thenlper/gte-base",
    "all-distilroberta-v1",
    "msmarco-distilbert-base-v4",
    "msmarco-MiniLM-L-6-v3",
    "msmarco-MiniLM-L-12-v3",
]

SCORE_TYPES = ["Cosine Similarity"]


class NeuralSearch(BaseSearch):
    def __init__(
        self,
        model: str = RETRIVAL_MODELS[0],
        session: OfferDBSession | None = None,
        cache: str = "models/neural",
    ) -> None:
        self.cache = cache
        self._session = session if session else OfferDBSession()
        self.retrieval_index_cache = None
        self.index = None
        self.model = None
        # os.makedirs(cache+"/retrieval", exist_ok=True)
        self.load(model=model)

    @property
    def session(self) -> OfferDBSession:
        return self._session

    @session.setter
    def session(self, value: OfferDBSession) -> None:
        self._session = value

    def load(self, model: str) -> None:
        model = model if model else RETRIVAL_MODELS[0]
        self.retrieval_index_cache = os.path.join(
            self.cache, "retrieval", f"{model.split('/')[-1]}.index"
        )
        self.model = SentenceTransformer(model)
        self.model.to(DEVICE)
        if os.path.exists(self.retrieval_index_cache):
            # print("Loading Faiss index from file...")
            self.index = faiss.read_index(self.retrieval_index_cache)
        else:
            self.index = self.create_index()

    def create_index(self):
        targets = self.session.get_targets()
        text_embeddings = self.model.encode(targets)
        index = faiss.IndexFlatL2(text_embeddings.shape[1])
        index.metric_type = faiss.METRIC_INNER_PRODUCT
        index.add(text_embeddings)  # pylint: disable=no-value-for-parameter
        faiss.write_index(index, self.retrieval_index_cache)

        return index

    def get_scores(self, query: str) -> DataFrame:
        # Calculate the embedding
        embedding = self.model.encode(query.lower()).reshape(1, -1)

        # Calculate similarity between user input and each offer
        scores, indices = self.index.search(embedding, self.index.ntotal)

        results = DataFrame.from_dict({"SCORE": scores[0]})
        results = results.set_index(Index(indices[0]))
        return results
