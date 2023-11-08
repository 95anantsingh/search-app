from pandas import DataFrame, Index
from scipy.stats import gmean, hmean
from sklearn.preprocessing import MinMaxScaler, normalize as l2_norm

from .bm25 import BM25Search
from .neural import NeuralSearch
from .offers_db import OfferDBSession
from .base_search import BaseSearch

NORM_TYPES = ["L2", "Min-Max"]
MEAN_TYPES = ["Arithmetic", "Geometric", "Harmonic"]


class HybridSearch(BaseSearch):
    """
    A class for hybrid search that combines BM25-based search and Neural search methods.

    Args:
    - model (str | None, optional): The retrieval model to use for Neural search. Defaults to None.
    - score_type (str | None, optional): The type of similarity scoring to use for Neural search. Defaults to None.
    - session (OfferDBSession | None, optional): An optional OfferDBSession for database interaction. Defaults to None, creating a new session if not provided.

    Attributes:
    - session (OfferDBSession): The database session for offer data.
    - bm25 (BM25Search): An instance of BM25Search for BM25-based search.
    - neural (NeuralSearch): An instance of NeuralSearch for Neural search.

    Methods:
    - load(model: str, score_type: str) -> None: Load the retrieval model and set the score type for Neural search.
    - get_scores(query: str, mean_type=MEAN_TYPES[0], norm_type: str | None = NORM_TYPES[0]) -> DataFrame: Get similarity scores for offers using both BM25 and Neural methods and combine them based on specified mean and normalization types.
    - search(query: str, mean_type=MEAN_TYPES[0], norm_type: str | None = NORM_TYPES[0], top_k=50, threshold=0.05, dis_threshold=0.35, e_top_k=False, e_threshold=False, e_cluster=False) -> DataFrame: Execute a search for offers based on the provided query and optional parameters.
    """

    def __init__(
        self,
        model: str | None = None,
        score_type: str | None = None,
        session: OfferDBSession | None = None,
    ) -> None:
        """
        Initialize a HybridSearch instance.

        Args:
        - model (str | None, optional): The retrieval model to use for Neural search. Defaults to None.
        - score_type (str | None, optional): The type of similarity scoring to use for Neural search. Defaults to None.
        - session (OfferDBSession | None, optional): An optional OfferDBSession for database interaction. Defaults to None, creating a new session if not provided.
        """
        self._session = session if session else OfferDBSession()
        self.bm25 = BM25Search(session=session)
        self.neural = NeuralSearch(session=session)
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
        Load the retrieval model and set the score type for Neural search.

        Args:
        - model (str): The name or identifier of the retrieval model to use.
        - score_type (str): The type of similarity scoring to use.
        """
        self.neural.load(model=model, score_type=score_type)

    def get_scores(
        self,
        query: str,
        mean_type=MEAN_TYPES[0],
        norm_type: str | None = NORM_TYPES[0],
    ) -> DataFrame:
        """
        Get similarity scores for offers using both BM25 and Neural methods and combine them based on specified mean and normalization types.

        Args:
        - query (str): The query string to match against offers.
        - mean_type (str, optional): The type of mean to use for combining scores. Defaults to "Arithmetic".
        - norm_type (str | None, optional): The type of normalization to use for combining scores. Defaults to "L2" if not specified.

        Returns:
        DataFrame: A DataFrame containing combined similarity scores for offers.
        """
        bm25_scores = self.bm25.get_scores(query)
        bm25_scores.loc[:,"index"] = bm25_scores.index.values
        bm25_scores.rename(columns={"SCORE": "BM25_SCORE"}, inplace=True)

        neural_scores = self.neural.get_scores(query)
        neural_scores.loc[:,"index"] = neural_scores.index.values
        neural_scores.rename(columns={"SCORE": "NEURAL_SCORE"}, inplace=True)
        scores = bm25_scores.merge(neural_scores, on="index", how="left")

        match norm_type:
            case "L2":
                scores["BM25_SCORE"] = l2_norm(
                    scores[["BM25_SCORE"]], norm="l2", axis=0
                )
                scores["NEURAL_SCORE"] = l2_norm(
                    scores[["NEURAL_SCORE"]], norm="l2", axis=0
                )
            case "Min-Max":
                scaler = MinMaxScaler()
                scores["BM25_SCORE"] = scaler.fit_transform(scores[["BM25_SCORE"]])
                scores["NEURAL_SCORE"] = scaler.fit_transform(scores[["NEURAL_SCORE"]])

        match mean_type:
            case "Arithmetic":
                scores["SCORE"] = (scores["BM25_SCORE"] + scores["NEURAL_SCORE"]) / 2
            case "Geometric":
                if not norm_type or norm_type == "L2":
                    scores["NEURAL_SCORE"] = scores["NEURAL_SCORE"].clip(lower=0)
                scores["SCORE"] = gmean(scores[["BM25_SCORE", "NEURAL_SCORE"]], axis=1)
            case "Harmonic":
                if not norm_type or norm_type == "L2":
                    scores["NEURAL_SCORE"] = scores["NEURAL_SCORE"].clip(lower=0)
                scores["SCORE"] = hmean(scores[["BM25_SCORE", "NEURAL_SCORE"]], axis=1)

        results = DataFrame.from_dict({"SCORE": scores["SCORE"]})
        results = results.set_index(Index(scores["index"].to_list()))
        results = results.sort_values(by="SCORE", ascending=False)
        return results

    def search(
        self,
        query: str,
        mean_type=MEAN_TYPES[0],
        norm_type: str | None = NORM_TYPES[0],
        top_k=50,
        threshold=0.05,
        dis_threshold=0.35,
        e_top_k=False,
        e_threshold=False,
        e_cluster=False,
    ) -> DataFrame:
        """
        Execute a search for offers based on the provided query and optional parameters.

        Args:
        - query (str): The query string to match against offers.
        - mean_type (str, optional): The type of mean to use for combining scores. Defaults to "Arithmetic".
        - norm_type (str | None, optional): The type of normalization to use for combining scores. Defaults to "L2" if not specified.
        - top_k (int, optional): The number of top offers to retrieve. Defaults to 50.
        - threshold (float, optional): The similarity score threshold for filtering offers. Defaults to 0.05.
        - dis_threshold (float, optional): The threshold for Euclidean distance used in clustering. Defaults to 0.35.
        - e_top_k (bool, optional): Whether to apply a top-k filter. Defaults to False.
        - e_threshold (bool, optional): Whether to apply a threshold filter. Defaults to False.
        - e_cluster (bool, optional): Whether to perform clustering on similarity scores. Defaults to False.

        Returns:
        DataFrame: A DataFrame containing relevant offers based on the provided query and criteria.
        """
        scores = self.get_scores(query=query, mean_type=mean_type, norm_type=norm_type)

        offers = self.get_offers(
            scores=scores,
            top_k=top_k,
            threshold=threshold,
            dis_threshold=dis_threshold,
            e_top_k=e_top_k,
            e_threshold=e_threshold,
            e_cluster=e_cluster,
        )

        return offers
