import os
import pickle
import pandas as pd
import ast
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

from config import config


class TfIdf:
    """
    Process all recipes used in embeddings by tf-idf in order to be able
    to find most similar recipes to our set of ingredients.
    """

    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.connecter = lambda x: ". ".join(ast.literal_eval(x))
        self.vectorizer = self.__get_vectorizer()
        self.existing_ids = self.__get_existing_ids()

    def process(self):
        df = pd.read_csv(self.csv_path)

        ids = []
        corpus = []

        for i in tqdm(range(df.shape[0])):
            row = df.loc[i]
            if row.id in self.existing_ids:
                ids.append(row.id)
                # corpus.append(self.connecter(row.steps))
                corpus.append(self.connecter(row.ingredients))

        tfidf_matrix = self.vectorizer.fit_transform(corpus)

        data = {
            "vectorizer": self.vectorizer,
            "tfidf_matrix": tfidf_matrix,
            "recipe_ids": ids,
        }
        with open(os.path.join(config.NEW_DATA_PATH, config.TFIDF_FILE), "wb") as file:
            pickle.dump(data, file)

    def __get_existing_ids(self):
        with open(
            os.path.join(config.NEW_DATA_PATH, config.FLOW_GRAPS_FILE), "rb"
        ) as file:
            data = pickle.load(file)
        return set(map(int, data.keys()))

    def __get_vectorizer(self):
        ...
        return TfidfVectorizer()


if __name__ == "__main__":
    x = TfIdf(csv_path="./eat_pim/data/RAW_recipes.csv")
    x.process()
