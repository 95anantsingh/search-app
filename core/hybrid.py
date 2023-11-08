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
    def __init__(
        self,
        model: str | None = None,
        score_type: str | None = None,
        session: OfferDBSession | None = None,
    ) -> None:
        self._session = session if session else OfferDBSession()
        self.bm25 = BM25Search(session=session)
        self.neural = NeuralSearch(session=session)
        self.load(model=model, score_type=score_type)

    @property
    def session(self) -> OfferDBSession:
        return self._session

    @session.setter
    def session(self, value: OfferDBSession) -> None:
        self._session = value

    def load(self, model: str, score_type: str) -> None:
        self.neural.load(model=model, score_type=score_type)

    def get_scores(
        self,
        query: str,
        mean_type=MEAN_TYPES[0],
        norm_type: str | None = NORM_TYPES[0],
    ) -> DataFrame:
        bm25_scores = self.bm25.get_scores(query)
        bm25_scores["index"] = bm25_scores.index.values
        bm25_scores.rename(columns={"SCORE": "BM25_SCORE"}, inplace=True)

        neural_scores = self.neural.get_scores(query)
        neural_scores["index"] = neural_scores.index.values
        neural_scores.rename(columns={"SCORE": "NEURAL_SCORE"}, inplace=True)
        scores = bm25_scores.merge(neural_scores, on="index", how="left")

        if norm_type:
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
                    scores["NEURAL_SCORE"] = scaler.fit_transform(
                        scores[["NEURAL_SCORE"]]
                    )

        match mean_type:
            case "Arithmetic":
                scores["SCORE"] = (scores["BM25_SCORE"] + scores["NEURAL_SCORE"]) / 2
            case "Geometric":
                scores["SCORE"] = gmean(scores[["BM25_SCORE", "NEURAL_SCORE"]], axis=1)
            case "Harmonic":
                scores["SCORE"] = hmean(scores[["BM25_SCORE", "NEURAL_SCORE"]], axis=1)

        results = DataFrame.from_dict({"SCORE": scores["SCORE"]})
        results = results.set_index(Index(scores["index"].to_list()))
        results = results.sort_values(by="SCORE", ascending=False)
        return results

    def search(
        self,
        query: str,
        mean_type=MEAN_TYPES[0],
        norm_type:str | None =NORM_TYPES[0],
        top_k=50,
        threshold=0.05,
        dis_threshold=0.35,
        e_top_k=False,
        e_threshold=False,
        e_cluster=False,
    ) -> DataFrame:
        scores = self.get_scores(
            query=query, mean_type=mean_type, norm_type=norm_type
        )

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
