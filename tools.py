"""
Created on 15/02/2024
@author: Hind FARIS, Jérôme Sioc'han de Kersabiec

useful functions to analyse air quality dataset
"""
import pandas as pd
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import geopandas as gpd


def best_nb_cluster(
        dataframe: pd.DataFrame,
        columns_to_cluster: list,
        nb_cluster_min: int,
        nb_cluster_max: int,
        show_plot_silhouette_scores: bool = False,
        title: str = None,
        step_for_plot: int = 2
) -> int:
    """
        determine the optimal number of clusters based on silhouette score
        :param step_for_plot: the step used for the x-axis of the plot
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
        plt.figure(figsize=(12, 7))
        plt.plot(range(nb_cluster_min, nb_cluster_max), silhouette_scores, marker='o')
        plt.xticks(np.arange(nb_cluster_min, nb_cluster_max, step_for_plot))
        plt.xlabel('Number of clusters', fontsize=10)
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
        cluster_means = pd.DataFrame(dataframe[columns_to_cluster[0]]).groupby(cluster_labels).mean()
        dic_reorder = dict(zip(np.argsort(cluster_means.values.flatten()), range(best_nb_clusters)))
        return dic_reorder, cluster_labels
    else:
        return cluster_labels


def visualize_clusters_map(
        cluster_labels: np.ndarray,
        title: str,
        dic_reorder: dict = None
):
    """
    Visualize the clusters of UHF42 on the map ordered (if dic_reorder i not None)
    :param cluster_labels: cluster corresponding to the data
    :param title: title of the visualization
    :param dic_reorder: visualize the clusters from less important values to more important ones using cmap
    """

    # reading file containing the geodata used for visualization
    file = 'departements.geojson'
    geo_data = gpd.read_file(file)
    # geo_data.drop(0, inplace=True)
    # adding the cluster number
    geo_data['Cluster'] = cluster_labels

    # ordering the clusters
    if dic_reorder is not None:
        geo_data['Cluster'] = geo_data['Cluster'].map(dic_reorder)

    # visualizing the clusters on map
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    geo_data.plot(column='Cluster', legend=True, edgecolor='black', ax=ax, cmap='Wistia',
                  categorical=True,
                  legend_kwds={'fontsize': 8, 'title': "Cluster Number", 'title_fontsize': 10, 'loc': 'upper left'})
    plt.title(title, fontsize=15)

    # adding the numbers of the UHF42 to the map
    for idx, row in geo_data.iterrows():
        ax.annotate(text=row['id'], xy=row['geometry'].centroid.coords[0], ha='center', fontsize=8, color='black')
    ax.set_axis_off()
