# Branch urban_security

This branch aims to study the evolution of criminality in France. To do so, we clusterise the departements according to criminality values on two periods (respectively 2000-20013 and 2014-2020).

To do so, we first apply a PCA on criminality columns. By ploting the percentage of the explained variance of the data, we kept the 3 first columns of the PCA as it explains already 83% of the variance.

Finding the optimal number of cluster was a bit complicated. Firstly by ploting the silhouette score we saw that the optimal number of cluster should be 2. However 2 clusters limited the interpretation. Thus we tried to find a higher number of clusters. We noticed that there is a rebound for the number of clusters equal to 5 on the silouhette score. To be sure, we ploted the elbow score to find out if there is a break. It comes out that a small break is observed for 5 clusters. So for the rest of the study we kept this number of clusters.

Overall we can see that the departments that have changed cluster are departments that are known to have seen a recent increase in crime. For example, there has been an increase in crime in the departments of Nantes, Bordeaux and Grenoble. It should be noted that the average values for clusters 2 and 3 are very close. So the change of cluster for Corsica is not so relevant.
