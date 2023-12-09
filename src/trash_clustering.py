import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from config import config
from ingredients_selection import IngredientSelector

selector = IngredientSelector()
ingredients = selector.select_ingredients()

df = pd.read_csv(os.path.join(config.NEW_DATA_PATH, "nodes_191120.csv"))
ids = []
for ing in ingredients:
    filtered_df = df[df["name"].str.contains(ing, na=False)]
    if filtered_df.empty:
        filtered_df = df[df["name"].str.contains(ing.split()[1], na=False)]
    if not filtered_df.empty:
        ids.append(filtered_df.iloc[0].node_id)
    else:
        ids.append(None)

with open(os.path.join("./src/data/", "flavorgraph_embedding.pickle"), "rb") as file:
    data = pickle.load(file)
embeddings = np.array([data[str(i)] if i else np.zeros((len(data["1"]))) for i in ids])

with open(
    os.path.join(config.NEW_DATA_PATH, config.INGREDIENTS_EMBEDDINGS_FILE), "rb"
) as file:
    data = pickle.load(file)
embeddings1 = []

for ing in ingredients:
    idx = np.where(data["entities"] == ing)[0]
    embeddings1.append(data["embeddings"][idx][0])

embeddings = np.concatenate((embeddings, embeddings1), axis=1)
scaler = StandardScaler()
embeddings = scaler.fit_transform(embeddings)

############################################################### PCA
# pca = PCA(n_components=2)
# pca_result = pca.fit_transform(embeddings)

# plt.scatter(pca_result[:, 0], pca_result[:, 1])
# for i, entity_name in enumerate(ingredients):
#     plt.annotate(entity_name, (pca_result[i, 0] - 1, pca_result[i, 1]))

# plt.title("PCA Visualization of ingridients using mixed embeddings", fontsize=15)
# plt.show()
# plt.savefig("./src/plots/pca.png", dpi=300)


############################################################### ELBOW
# def calculate_ssd(embeddings, k_range):
#     ssd = []
#     for k in k_range:
#         kmeans = KMeans(n_clusters=k, random_state=42)
#         kmeans.fit(embeddings)
#         ssd.append(kmeans.inertia_)
#     return ssd


# def elbow_point(ssd, k_range, threshold=0.1):
#     rate_of_change = np.diff(ssd) / ssd[:-1]
#     elbow = np.argwhere(rate_of_change < -threshold).flatten()
#     return k_range[elbow[0]] if len(elbow) > 0 else k_range[-1]


# def cluster_and_print(embeddings, entities, n_clusters):
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#     kmeans.fit(embeddings)
#     clusters = {i: [] for i in range(n_clusters)}
#     for i, label in enumerate(kmeans.labels_):
#         clusters[label].append(entities[i])
#     for cluster, members in clusters.items():
#         print(f"Cluster {cluster}: {members}")


# ssd = calculate_ssd(embeddings, range(1, len(ingredients)))

# plt.plot(range(1, len(ingredients)), np.array(ssd) / 500, "bo-")
# plt.xlabel("Number of clusters (k)")
# plt.ylabel("Sum of Squared Distances")
# plt.title("Elbow Method for mixed embeddings", fontsize=15)
# plt.savefig("./src/plots/elbow_mixed.png", dpi=300)
# plt.show()

# optimal_k = elbow_point(ssd, range(1, len(ingredients)))
# print(f"Optimal number of clusters: {optimal_k}")

# # Cluster and print the results
# cluster_and_print(embeddings, ingredients, optimal_k)


# kmeans = KMeans(n_clusters=3, random_state=0)

# # Fit the model to the embeddings
# kmeans.fit(embeddings)

# # Get the cluster labels for each entity
# cluster_labels = kmeans.labels_

# # Create a dictionary to map entities to clusters
# entity_cluster_mapping = {}
# for entity, cluster in zip(ingredients, cluster_labels):
#     if cluster not in entity_cluster_mapping:
#         entity_cluster_mapping[cluster] = []
#     entity_cluster_mapping[cluster].append(entity)

# # Print the entities in each cluster
# for cluster, entities_in_cluster in entity_cluster_mapping.items():
#     print(f"Cluster {cluster + 1}: {entities_in_cluster}")
