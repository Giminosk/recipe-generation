import os
import json
import pickle
import numpy as np

from config import config


data_path = config.NEW_DATA_PATH


def ingredients_existing_in_cooc(
    ingredients: list[str], cooc_ingredients: dict, full_short_matches: dict
):
    """
    Filters and returns a list of ingredients that exist in the co-occurrence dataset.
    For ingredients not directly found, it attempts to find a short form match.
    If a match is found in the co-occurrence dataset, it is included in the returned list.
    """
    existing_ingredients = []
    for ing in ingredients:
        if ing in cooc_ingredients:
            existing_ingredients.append(ing)
        else:
            short_ing = full_short_matches.get(ing, None)[0]
            if short_ing is not None and short_ing in cooc_ingredients:
                existing_ingredients.append(short_ing)
    return existing_ingredients


def determine_threshold(
    cooc_matrix: list[list[int]],
    ing_to_idx: dict,
    ingredients: list[str],
    percentile=65,
):
    """
    Determines a threshold value for ingredient selection based on the co-occurrence matrix.
    Percentile selected manually and should be adjusted.
    """
    relevant_indices = [ing_to_idx[ing] for ing in ingredients]
    relevant_scores = cooc_matrix[np.ix_(relevant_indices, relevant_indices)].flatten()
    threshold = np.percentile(relevant_scores, percentile)
    if threshold < 0:
        raise ValueError(f"threshold cannot be non-positive, threshold: {threshold}")
    return threshold


def select(
    ingredients: list[str],
    cooc_matrix: list[list[int]],
    sim_matrix: list[list[float]],
    ing_to_idx: dict,
):
    """
    Selects a subset of ingredients based on their co-occurrence and similarity scores which computed also from coocs.
    It starts by selecting the most common ingredient, then adds ingredients to meet
    a minimum count, and continues adding ingredients based on similarity scores until a threshold is met.
    """
    if len(ingredients) < config.MIN_INGREDIENTS:
        raise ValueError(f"At least {config.MIN_INGREDIENTS} ingredients are required.")

    selected_ingredients = []

    # Choose first ingredient as the most common among all
    most_common_ing, highest_score = None, -float("inf")
    for ing in ingredients:
        score = sum(
            cooc_matrix[ing_to_idx[ing]][ing_to_idx[ing2]] for ing2 in ingredients
        )
        if score > highest_score:
            highest_score = score
            most_common_ing = ing
    print(f"First ingredient is {most_common_ing}")
    selected_ingredients.append(most_common_ing)

    # Choose other ingredients needed to reach the minimum number
    for _ in range(config.MIN_INGREDIENTS - 1):
        best_ing, highest_score = None, -float("inf")
        for ing in ingredients:
            if ing not in selected_ingredients:
                score = sum(
                    sim_matrix[ing_to_idx[ing]][ing_to_idx[selected_ing]]
                    for selected_ing in selected_ingredients
                )
                if score > highest_score:
                    highest_score = score
                    best_ing = ing
        selected_ingredients.append(best_ing)
    print(f"Minimal needed ingredients are {selected_ingredients}")

    threshold = determine_threshold(sim_matrix, ing_to_idx, ingredients)
    print(f"Treshold is {threshold}")

    # Iteratively add more ingredients until threshold is met
    while True:
        best_ing, highest_score = None, -float("inf")
        for ing in ingredients:
            if ing not in selected_ingredients:
                score = sum(
                    sim_matrix[ing_to_idx[ing]][ing_to_idx[selected_ing]]
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


def select_ingredients():
    """
    Main function to select ingredients. It loads ingredients, co-occurrence data, and
    similarity matrices from files. It then filters and selects ingredients based on
    their co-occurrence and similarity. The final selected ingredients are returned.
    """
    print("Selecting ingredients ...")

    with open(os.path.join(data_path, config.INGREDIENTS_FILE)) as json_file:
        ingredients = json.load(json_file)
        assert isinstance(ingredients, list)

    with open(os.path.join(data_path, config.COOC_FILE), "rb") as file:
        cooc = pickle.load(file)

    with open(os.path.join(data_path, config.FULL_SHORT_FILE), "rb") as file:
        full_short_matches = pickle.load(file)

    ingredients = ingredients_existing_in_cooc(
        ingredients, cooc["ing_to_index"], full_short_matches
    )
    ingredients = select(
        ingredients,
        cooc["ing_cooc_matrix"].toarray(),
        cooc["sim_matrix"],
        cooc["ing_to_index"],
    )

    return ingredients


if __name__ == "__main__":
    select_ingredients()
