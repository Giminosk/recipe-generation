import os
import json
import pickle
import numpy as np
from config import config


class IngredientSelector:
    """
    Select most valueble ingredients from set of ingredients.
    """

    def __init__(self):
        self.data_path = config.NEW_DATA_PATH
        self.load_data()

    def load_data(self):
        with open(os.path.join(self.data_path, config.INGREDIENTS_FILE)) as json_file:
            self.ingredients = json.load(json_file)
            assert isinstance(self.ingredients, list)

        with open(os.path.join(self.data_path, config.COOC_FILE), "rb") as file:
            self.cooc = pickle.load(file)
            self.cooc_matrix = self.cooc["ing_cooc_matrix"].toarray()

        with open(os.path.join(self.data_path, config.FULL_SHORT_FILE), "rb") as file:
            self.full_short_matches = pickle.load(file)

    def ingredients_existing_in_cooc(self, ingredients: list[str]):
        """
        Filters and returns a list of ingredients that exist in the co-occurrence dataset.
        For ingredients not directly found, it attempts to find a short form match.
        If a match is found in the co-occurrence dataset, it is included in the returned list.
        """
        existing_ingredients = []
        for ing in ingredients:
            if ing in self.cooc["ing_to_index"]:
                existing_ingredients.append(ing)
            else:
                short_ing = self.full_short_matches.get(ing, None)[0]
                if short_ing is not None and short_ing in self.cooc["ing_to_index"]:
                    existing_ingredients.append(short_ing)
        return existing_ingredients

    def determine_threshold(self, ingredients: list[str], percentile=65):
        """
        Determines a threshold value for ingredient selection based on the co-occurrence matrix.
        Percentile selected manually and should be adjusted.
        """
        relevant_indices = [self.cooc["ing_to_index"][ing] for ing in ingredients]
        relevant_scores = self.cooc["sim_matrix"][
            np.ix_(relevant_indices, relevant_indices)
        ].flatten()
        threshold = np.percentile(relevant_scores, percentile)
        if threshold < 0:
            raise ValueError(
                f"threshold cannot be non-positive, threshold: {threshold}"
            )
        return threshold

    def select(self, ingredients: list[str]):
        """
        Selects a subset of ingredients based on their co-occurrence and similarity scores which computed also from coocs.
        It starts by selecting the most common ingredient, then adds ingredients to meet
        a minimum count, and continues adding ingredients based on similarity scores until a threshold is met.
        """
        if len(ingredients) < config.MIN_INGREDIENTS:
            raise ValueError(
                f"At least {config.MIN_INGREDIENTS} ingredients are required."
            )

        selected_ingredients = []
        most_common_ing, highest_score = None, -float("inf")
        for ing in ingredients:
            score = sum(
                self.cooc_matrix[self.cooc["ing_to_index"][ing]][
                    self.cooc["ing_to_index"][ing2]
                ]
                for ing2 in ingredients
            )
            if score > highest_score:
                highest_score = score
                most_common_ing = ing
        print(f"First ingredient is {most_common_ing}")
        selected_ingredients.append(most_common_ing)

        for _ in range(config.MIN_INGREDIENTS - 1):
            best_ing, highest_score = None, -float("inf")
            for ing in ingredients:
                if ing not in selected_ingredients:
                    score = sum(
                        self.cooc["sim_matrix"][self.cooc["ing_to_index"][ing]][
                            self.cooc["ing_to_index"][selected_ing]
                        ]
                        for selected_ing in selected_ingredients
                    )
                    if score > highest_score:
                        highest_score = score
                        best_ing = ing
            selected_ingredients.append(best_ing)
        print(f"Minimal needed ingredients are {selected_ingredients}")

        threshold = self.determine_threshold(ingredients)
        print(f"Treshold is {threshold}")

        while True:
            best_ing, highest_score = None, -float("inf")
            for ing in ingredients:
                if ing not in selected_ingredients:
                    score = sum(
                        self.cooc["sim_matrix"][self.cooc["ing_to_index"][ing]][
                            self.cooc["ing_to_index"][selected_ing]
                        ]
                        for selected_ing in selected_ingredients
                    ) / len(selected_ingredients)
                    if score > highest_score:
                        highest_score = score
                        best_ing = ing
            if highest_score < threshold:
                break
            selected_ingredients.append(best_ing)
        print(
            f"Selected {len(selected_ingredients)} ingredients from {len(ingredients)}, they are {selected_ingredients}"
        )

        return selected_ingredients

    def select_ingredients(self):
        ingredients = self.ingredients_existing_in_cooc(self.ingredients)
        return self.select(ingredients)


if __name__ == "__main__":
    selector = IngredientSelector()
    selected_ingredients = selector.select_ingredients()
    print(selected_ingredients)
