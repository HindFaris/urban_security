"""
Created on 15/02/2024
@author: Hind FARIS, Jérôme Sioc'han de Kersabiec

useful functions to analyse air quality dataset
"""
import pandas as pd
import os
import geopandas as gpd
import numpy as np
import seaborn as sns
from typing import Tuple
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import MinMaxScaler


def best_nb_cluster(
        dataframe: pd.DataFrame,
        columns_to_cluster: list,
        nb_cluster_min: int,
        nb_cluster_max: int,
        show_plot_silhouette_scores: bool = False,
        title: str = None
) -> int:
    """
        determine the optimal number of clusters based on silhouette score
        :param dataframe: dataframe containing the data to cluster
        :param columns_to_cluster: columns to cluster from the dataframe
        :param nb_cluster_min: minimum number of clusters to test
        :param nb_cluster_max: maximal number of clusters to test
        :param show_plot_silhouette_scores: plot the evolution of the silhouette score by number of clusters
        :param title: title of the map showing clusters
        :return: optimal number of clusters based on the silhouette score
    """

    data_to_cluster = dataframe[columns_to_cluster].values

    silhouette_scores = []
    for n_clusters in range(nb_cluster_min, nb_cluster_max):
        kmeans = KMeans(n_clusters=n_clusters)
        cluster_labels = kmeans.fit_predict(data_to_cluster)
        silhouette = silhouette_score(data_to_cluster, cluster_labels)
        silhouette_scores.append(silhouette)

    if show_plot_silhouette_scores and title is not None:
        plt.figure(figsize=(12, 9))
        plt.plot(range(nb_cluster_min, nb_cluster_max), silhouette_scores, marker='o')
        plt.xlabel('Number of clusters', fontsize=10)
        plt.xticks(range(nb_cluster_min, nb_cluster_max))
        plt.ylabel('Silhouette score', fontsize=10)
        plt.title(title, fontsize=15)
        plt.show()

    if show_plot_silhouette_scores and title is None:
        raise ValueError("To show the plot of evolution of silhouette score, a title must be provided.")

    list_nb_cluster = list(range(nb_cluster_min, nb_cluster_max + 1))
    index_best_nb_cluster = silhouette_scores.index(max(silhouette_scores))

    return list_nb_cluster[index_best_nb_cluster]


def clustering(dataframe: pd.DataFrame,
               columns_to_cluster: list,
               best_nb_clusters: int,
               order: bool = False
               ) -> Tuple[dict, np.ndarray]:
    """
    Cluster the data using K-Means and the number of clusters given
    :param dataframe: dataframe containing the data to cluster
    :param columns_to_cluster: columns to cluster from the dataframe
    :param best_nb_clusters: optimal number of clusters
    :param order: if True, return the dictionary that assigns each cluster number to its order
    :return: numpy array containing the cluster assignments for each data point,
    and dictionary that assigns each cluster number to its order (if order is True)
    """

    # applying k-means algorithm
    kmeans = KMeans(n_clusters=best_nb_clusters)
    cluster_labels = kmeans.fit_predict(dataframe[columns_to_cluster])

    if order:
        # ordering the clusters
        cluster_means = pd.DataFrame(dataframe[columns_to_cluster]).groupby(cluster_labels).mean()
        dic_reorder = dict(zip(np.argsort(cluster_means.values.flatten()), range(best_nb_clusters)))
        return dic_reorder, cluster_labels
    else:
        return cluster_labels