import json
from abc import abstractmethod
import numpy as np
from sklearn.cluster import KMeans
from pandas import DataFrame, options

from .offers_db import OfferDBSession

options.mode.copy_on_write = False


class BaseSearch:
    def __init__(self) -> None:
        pass

    @property
    @abstractmethod
    def session(self) -> OfferDBSession:
        pass

    @abstractmethod
    def get_scores(self, query: str) -> DataFrame:
        pass

    def load(self, model: str) -> None:
        pass

    def get_offers(
        self,
        scores: DataFrame,
        top_k=50,
        threshold=0.05,
        dis_threshold=0.35,
        e_top_k=False,
        e_threshold=False,
        e_cluster=False,
    ) -> DataFrame:
        # Sort offers by similarity and return the results above the threshold
        results = scores
        if e_threshold:
            results = results[results["SCORE"] > threshold]
        # results = results.sort_values(by="SCORE", ascending=False)
        if e_top_k:
            results = results.head(top_k)

        # Apply K-Means clustering to the similarity scores
        if e_cluster and results.shape[0] > 0:
            n_clusters = 2
            kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
            results["CLUSTER"] = kmeans.fit_predict(results[["SCORE"]])
            cluster_centers = kmeans.cluster_centers_

            # Find the lowest point in the cluster with the higher center
            # lowest_point_cluster_higher_center = results[
            #     results["CLUSTER"] == np.argmax(cluster_centers)
            # ]["SCORE"].min()

            # # Find the highest point in the other cluster
            # highest_point_other_cluster = results[
            #     results["CLUSTER"] != np.argmax(cluster_centers)
            # ]["SCORE"].max()

            # Calculate the Euclidean distance
            # distance = np.abs(
            #     lowest_point_cluster_higher_center - highest_point_other_cluster
            # )

            # Calculate distances between cluster centers
            cluster_0_center = cluster_centers[0]
            cluster_1_center = cluster_centers[1]
            distance = np.linalg.norm(cluster_0_center - cluster_1_center)

            if distance >= dis_threshold:
                results = results[results["CLUSTER"] == np.argmax(cluster_centers)]
                print("Distance:", distance)

        else:
            results["CLUSTER"] = np.nan

        # Pull data from Database
        data = DataFrame(
            self.session.get_rows(indices=results.index.to_list(), as_dict=True)
        )
        columns = [
            "index",
            "SCORE",
            "OFFER",
            "RETAILER",
            "BRAND",
            "CATEGORIES",
            "SUPER_CATEGORIES",
            "CLUSTER",
        ]

        if results.shape[0] > 0:
            results["index"] = results.index.values
            results = results.merge(data, on="index", how="left")
            results.CATEGORIES = results.CATEGORIES.apply(json.loads)
            results.SUPER_CATEGORIES = results.SUPER_CATEGORIES.apply(json.loads)
        else:
            results = DataFrame.from_dict({col: [] for col in columns})

        return results[columns]

    def search(
        self,
        query: str,
        mean_type: str | None = None,
        normalize:bool | None = None,
        norm_type: str | None = None,
        score_type: str | None = None,
        top_k=50,
        threshold=0.05,
        dis_threshold=0.35,
        e_top_k=False,
        e_threshold=False,
        e_cluster=False,
    ) -> DataFrame:
        scores = self.get_scores(query=query)

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
