# This dataset is about the information regarding various countries.
# It is an unlabelled dataset and hence we are going to implement clustering algorithm here
# K-Means Clustering is the algorithm that will be used for this dataset.

# 1. Start by importing all necessary modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as mtp
from sklearn.metrics import silhouette_score


# 2. Import dataset
data = pd.read_csv('Country-data.csv')
df=pd.DataFrame(data)
print(df.to_string())


# 3. Select features for x
x = df.iloc[:, 1:].values


# 4. Finding optimal number of clusters using the elbow method
from sklearn.cluster import KMeans
wcss_list= []
#Initializing the list for the values of WCSS
#Using for loop for iterations from 1 to 10.

for i in range(1, 11):
   kmeans = KMeans(n_clusters=i, init='k-means++', random_state= 42)
   kmeans.fit(x)
   wcss_list.append(kmeans.inertia_)
print(wcss_list)
mtp.plot(range(1, 11), wcss_list)
mtp.title('The Elbow Method Graph')
mtp.xlabel('Number of clusters(k)')
mtp.ylabel('wcss_list')
mtp.show()

# From the Elbow Method, we observed a sharp bent at the value 3, Hence we take K as 3.


# 5. Training the K-means model on a dataset
kmeans = KMeans(n_clusters=3, init='k-means++', random_state= 42)
y_predict= kmeans.fit_predict(x)
print(y_predict)


# To assess the clustering performance, we use a metric known as Silhouette Score
# 6. Finding Silhouette score
print(silhouette_score(x, kmeans.fit_predict(x)))

# We have observed that the silhouette score of this model is 0.7 which is considered to be strong and hence our clustering result is well defined.
