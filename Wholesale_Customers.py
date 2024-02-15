# Item details brought by customers in various regions is the data we have taken here.
# It is an unlabelled dataset and hence we are going to implement clustering algorithm here.
# Agglomerative Heirarchical Clustering is the algorithm that will be used for this dataset.

# 1. Start by importing all necessary modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as mtp
from sklearn.metrics import silhouette_score


# 2. Import dataset
data = pd.read_csv('Wholesale_customers_.csv')
df=pd.DataFrame(data)
print(df.to_string())


# 3. Select features for x
x = df.iloc[:, 2:].values


# 4. #Finding the optimal number of clusters using the Dendrogram
import scipy.cluster.hierarchy as shc
dendro = shc.dendrogram(shc.linkage(x, method="ward"))
mtp.title("Dendrogram Plot")
mtp.ylabel("Euclidean Distances")
mtp.xlabel("Customers")
mtp.show()


# 5. Training the hierarchical model on dataset
from sklearn.cluster import AgglomerativeClustering
hc= AgglomerativeClustering(n_clusters=4,metric='minkowski', linkage='single')
y_pred= hc.fit_predict(x)
print(y_pred)

# To assess the clustering performance, we use a metric known as Silhouette Score
# 6. Finding Silhouette score
print(silhouette_score(x, hc.fit_predict(x)))

# We have observed that the silhouette score of this model is 0.72 which is considered to be strong and hence our clustering result is well defined.
