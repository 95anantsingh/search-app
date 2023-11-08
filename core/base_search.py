import json
from abc import abstractmethod
import numpy as np
from sklearn.cluster import KMeans
from pandas import DataFrame, options

from .offers_db import OfferDBSession

options.mode.copy_on_write = False


class BaseSearch:
    """
    This is a base class for implementing a search algorithm for offers.

    Attributes:
    - session (OfferDBSession): An abstract property that needs to be implemented in derived classes. It represents the database session for offer data.

    Methods:
    - get_scores(query: str) -> DataFrame: Get similarity scores for offers based on the provided query.
    - load(model: str) -> None: Load a model for use in the search algorithm.
    - get_offers(scores: DataFrame, top_k=50, threshold=0.05, dis_threshold=0.35, e_top_k=False, e_threshold=False, e_cluster=False) -> DataFrame: Get a DataFrame of relevant offers based on the provided similarity scores and optional parameters.
    - search(query: str, mean_type: str | None = None, norm_type: str | None = None, top_k=50, threshold=0.05, dis_threshold=0.35, e_top_k=False, e_threshold=False, e_cluster=False) -> DataFrame: Execute a search for offers based on the provided query and optional parameters.
    """

    def __init__(self) -> None:
        pass

    @property
    @abstractmethod
    def session(self) -> OfferDBSession:
        """
        An abstract property representing the database session for offer data.

        Returns:
        OfferDBSession: An instance of the database session for offer data.
        """

    @abstractmethod
    def get_scores(self, query: str) -> DataFrame:
        """
        Get similarity scores for offers based on the provided query.

        Args:
        - query (str): The query string to match against offers.

        Returns:
        DataFrame: A DataFrame containing similarity scores for offers.
        """

    def load(self, model: str) -> None:
        """
        Load a model for use in the search algorithm.

        Args:
        - model (str): The name or identifier of the model to be loaded.
        """

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
        """
        Get a DataFrame of relevant offers based on the provided similarity scores and optional parameters.

        Args:
        - scores (DataFrame): A DataFrame containing similarity scores for offers.
        - top_k (int, optional): The number of top offers to retrieve. Defaults to 50.
        - threshold (float, optional): The similarity score threshold for filtering offers. Defaults to 0.05.
        - dis_threshold (float, optional): The threshold for Euclidean distance used in clustering. Defaults to 0.35.
        - e_top_k (bool, optional): Whether to apply a top-k filter. Defaults to False.
        - e_threshold (bool, optional): Whether to apply a threshold filter. Defaults to False.
        - e_cluster (bool, optional): Whether to perform clustering on similarity scores. Defaults to False.

        Returns:
        DataFrame: A DataFrame containing relevant offers based on the provided criteria.
        """

        # Sort offers by similarity and return the results above the threshold
        results = scores
        if e_threshold:
            results = results[results["SCORE"] > threshold]
        # results = results.sort_values(by="SCORE", ascending=False)
        if e_top_k:
            results = results.head(top_k)

        # Apply K-Means clustering to the similarity scores
        if e_cluster and results.shape[0] > 1:
            n_clusters = 2
            kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
            results = results.assign(CLUSTER=kmeans.fit_predict(results[["SCORE"]]))
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
        else:
            results = results.assign(CLUSTER=np.nan)

        # Pull data from Database
        data = DataFrame(self.session.get_rows(indices=results.index.to_list()))
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
            results = results.assign(index=results.index.values)
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
        norm_type: str | None = None,
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
        - mean_type (str | None, optional): The type of mean to use for similarity scoring. Defaults to None.
        - norm_type (str | None, optional): The type of normalization to use for similarity scoring. Defaults to None.
        - top_k (int, optional): The number of top offers to retrieve. Defaults to 50.
        - threshold (float, optional): The similarity score threshold for filtering offers. Defaults to 0.05.
        - dis_threshold (float, optional): The threshold for Euclidean distance used in clustering. Defaults to 0.35.
        - e_top_k (bool, optional): Whether to apply a top-k filter. Defaults to False.
        - e_threshold (bool, optional): Whether to apply a threshold filter. Defaults to False.
        - e_cluster (bool, optional): Whether to perform clustering on similarity scores. Defaults to False.

        Returns:
        DataFrame: A DataFrame containing relevant offers based on the provided query and criteria.
        """
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
