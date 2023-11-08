import os
import torch
import faiss
from pandas import DataFrame, Index
from sentence_transformers import SentenceTransformer

from .base_search import BaseSearch
from .offers_db import OfferDBSession


if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

RETRIVAL_MODELS = [
    "msmarco-distilbert-base-tas-b",
    "msmarco-distilbert-base-v4",
    "msmarco-MiniLM-L-6-v3",
    "msmarco-MiniLM-L-12-v3",
    "BAAI/bge-base-en-v1.5",
    "thenlper/gte-large",
    "llmrails/ember-v1",
    "thenlper/gte-base",
    "all-distilroberta-v1",
]

SCORE_TYPES = ["Dot Product", "Cosine Similarity"]


class NeuralSearch(BaseSearch):
    def __init__(
        self,
        model: str = RETRIVAL_MODELS[0],
        score_type: str = SCORE_TYPES[0],
        session: OfferDBSession | None = None,
        cache: str = "vectors/neural",
    ) -> None:
        self.cache = cache
        self._session = session if session else OfferDBSession()
        self.retrieval_index_cache = None
        self.index = None
        self.model = None
        self.score_type = None
        # os.makedirs(cache+"/retrieval", exist_ok=True)
        self.load(model=model, score_type=score_type)

    @property
    def session(self) -> OfferDBSession:
        return self._session

    @session.setter
    def session(self, value: OfferDBSession) -> None:
        self._session = value

    def load(self, model: str, score_type: str) -> None:
        model = model if model else RETRIVAL_MODELS[0]
        self.score_type = score_type if score_type else SCORE_TYPES[0]
        self.retrieval_index_cache = os.path.join(
            self.cache,
            "retrieval",
            f"{model.split('/')[-1]}_{''.join([w[0] for w in score_type.split(' ')])}.index",
        )
        self.model = SentenceTransformer(model)
        self.model.to(DEVICE)
        if os.path.exists(self.retrieval_index_cache):
            self.index = faiss.read_index(self.retrieval_index_cache)
        else:
            self.index = self.create_index()

    def create_index(self):
        targets = self.session.get_targets()
        embeddings = self.model.encode(targets)
        if self.score_type == "Cosine Similarity":
            faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.metric_type = faiss.METRIC_INNER_PRODUCT

        index.add(embeddings)  # pylint: disable=no-value-for-parameter
        faiss.write_index(index, self.retrieval_index_cache)

        return index

    def get_scores(self, query: str) -> DataFrame:
        # Calculate the embedding
        embedding = self.model.encode(query.lower()).reshape(1, -1)
        if self.score_type == "Cosine Similarity":
            faiss.normalize_L2(embedding)
        # Calculate similarity between user input and each offer
        scores, indices = self.index.search(embedding, self.index.ntotal)

        results = DataFrame.from_dict({"SCORE": scores[0]})
        results = results.set_index(Index(indices[0]))
        return results
