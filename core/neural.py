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
    """
    A class for implementing neural search using pre-trained sentence embeddings.

    Args:
    - model (str, optional): The name or identifier of the retrieval model to use. Defaults to the first model in RETRIVAL_MODELS.
    - score_type (str, optional): The type of similarity scoring to use. Defaults to "Dot Product".
    - session (OfferDBSession, optional): An optional OfferDBSession for database interaction. Defaults to None, creating a new session if not provided.
    - cache (str, optional): The path to the cache directory for storing retrieval indices. Defaults to "vectors/neural".

    Attributes:
    - cache (str): The path to the cache directory.
    - session (OfferDBSession): The database session for offer data.
    - retrieval_index_cache (str | None): The path to the retrieval index cache file, or None if not created yet.
    - index: The retrieval index created for similarity search.
    - model: The pre-trained sentence embedding model for encoding text.
    - score_type (str): The type of similarity scoring being used.

    Methods:
    - load(model: str, score_type: str) -> None: Load the retrieval model and set the score type.
    - create_index() -> faiss.Index: Create the retrieval index from embeddings.
    - get_scores(query: str) -> DataFrame: Get similarity scores for offers based on the provided query.
    """

    def __init__(
        self,
        model: str = RETRIVAL_MODELS[0],
        score_type: str = SCORE_TYPES[0],
        session: OfferDBSession | None = None,
        cache: str = "vectors/neural",
    ) -> None:
        """
        Initialize a NeuralSearch instance with the specified parameters.

        Args:
        - model (str, optional): The name or identifier of the retrieval model to use. Defaults to the first model in RETRIVAL_MODELS.
        - score_type (str, optional): The type of similarity scoring to use. Defaults to "Dot Product".
        - session (OfferDBSession, optional): An optional OfferDBSession for database interaction. Defaults to None, creating a new session if not provided.
        - cache (str, optional): The path to the cache directory for storing retrieval indices. Defaults to "vectors/neural".
        """
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
        """
        Get the database session for offer data.

        Returns:
        OfferDBSession: The database session for offer data.
        """
        return self._session

    @session.setter
    def session(self, value: OfferDBSession) -> None:
        """
        Set the database session for offer data.

        Args:
        - value (OfferDBSession): The database session to set.
        """
        self._session = value

    def load(self, model: str, score_type: str) -> None:
        """
        Load the retrieval model and set the score type.

        Args:
        - model (str): The name or identifier of the retrieval model to use.
        - score_type (str): The type of similarity scoring to use.
        """
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

    def create_index(self) -> faiss.IndexFlatL2:
        """
        Create the retrieval index from embeddings.

        Returns:
        faiss.IndexFlatL2: The created retrieval index.
        """
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
        """
        Get similarity scores for offers based on the provided query.

        Args:
        - query (str): The query string to match against offers.

        Returns:
        DataFrame: A DataFrame containing similarity scores for offers.
        """
        # Calculate the embedding
        embedding = self.model.encode(query.lower()).reshape(1, -1)
        if self.score_type == "Cosine Similarity":
            faiss.normalize_L2(embedding)
        # Calculate similarity between user input and each offer
        scores, indices = self.index.search(embedding, self.index.ntotal)

        results = DataFrame.from_dict({"SCORE": scores[0]})
        results = results.set_index(Index(indices[0]))
        return results
