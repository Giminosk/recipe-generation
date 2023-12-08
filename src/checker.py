import os
import pickle
import pandas as pd
import ast
from tqdm import tqdm
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import config
from ingredients_selection import IngredientSelector


class Checker:
    """
    Find most similar recipe to selected ingredients using tf-idf and visualizes recipe.
    """

    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.ingredient_selector = IngredientSelector()
        self.ingredients = self.ingredient_selector.select_ingredients()
        self.connecter = lambda x: ". ".join(ast.literal_eval(x))
        self.__load_data()

    def find_closest_recipe_id(self):
        ing_tfidf_vector = self.vectorizer.transform([", ".join(self.ingredients)])
        similarities = cosine_similarity(ing_tfidf_vector, self.tfidf_matrix)[0]
        indeces = np.argsort(similarities)[::-1]
        most_similar_document_ids = [self.recipe_idx[i] for i in indeces[:5]]
        return most_similar_document_ids[0]
        # df = pd.read_csv(self.csv_path)
        # ing_set = set(self.ingredients)
        # for i in indeces:
        #     recipe = df[df.id == self.recipe_idx[i]].squeeze()
        #     if set(ast.literal_eval(recipe.ingredients)).issubset(ing_set):
        #         return self.recipe_idx[i]

    def get_graph_by_id(self, id_):
        with open(
            os.path.join(config.NEW_DATA_PATH, config.FLOW_GRAPS_FILE), "rb"
        ) as file:
            data = pickle.load(file)

        G = nx.DiGraph()
        for e in data[str(id_)]["edges"]:
            G.add_edge(e[0], e[1])
        return G

    @staticmethod
    def simple_visualize(G):
        pos = nx.nx_pydot.graphviz_layout(G, prog="dot")  # dot or neato format
        plt.figure(1, figsize=(11, 11))

        nx.draw_networkx(G, pos, node_size=2000)
        node_labels = {}
        for n in G.nodes():
            if n[:5] == "pred_":
                node_labels[n] = n.split("_")[1]
            else:
                node_labels[n] = n
        nx.draw_networkx_labels(G, pos, labels=node_labels)
        plt.show()

    def __load_data(self):
        with open(os.path.join(config.NEW_DATA_PATH, config.TFIDF_FILE), "rb") as file:
            data = pickle.load(file)
            self.vectorizer = data["vectorizer"]
            self.tfidf_matrix = data["tfidf_matrix"]
            self.recipe_idx = data["recipe_ids"]


if __name__ == "__main__":
    checker = Checker(csv_path="./eat_pim/data/RAW_recipes.csv")
    id_ = checker.find_closest_recipe_id()
    print(f"Closest recipe id: {id_}")
    graph = checker.get_graph_by_id(id_)
    checker.simple_visualize(graph)
