import os
import json
import pickle
import numpy as np
import shutil

from config import config


old_data_path = config.OLD_DATA_PATH
new_data_path = config.NEW_DATA_PATH


def parse_matches():
    """
    Parses matches of ingredients and actions with FOODON and WIKIDATA entities.
    Createds 2 files with dicts which keys are ingredients/actions and values are lists with (almost always) one entity
    """
    with open(
        os.path.join(old_data_path, "word_cleanup_linking.json"), "r"
    ) as json_file:
        data = json.load(json_file)

    with open(os.path.join(new_data_path, config.FULL_SHORT_FILE), "wb") as pickle_file:
        pickle.dump(data["ing_to_ing"], pickle_file)

    foodon_matches = data["ing_to_foodon"].copy()
    foodon_matches.update(data["obj_to_foodon"])
    with open(
        os.path.join(new_data_path, config.FOODON_MATCHES_FILE), "wb"
    ) as pickle_file:
        pickle.dump(foodon_matches, pickle_file)

    with open(
        os.path.join(new_data_path, config.WIKIDATA_MATCHES_FILE), "wb"
    ) as pickle_file:
        pickle.dump(data["verb_to_preparations"], pickle_file)


def parse_flow_graphs():
    """
    Parses flow graphs
    """
    with open(os.path.join(old_data_path, "recipe_tree_data.json"), "r") as json_file:
        data = json.load(json_file)

    with open(os.path.join(new_data_path, config.FLOW_GRAPS_FILE), "wb") as pickle_file:
        pickle.dump(data, pickle_file)


def parse_entities_embeddings():
    """
    Parses ingredients and their entities embeddings. (ingridient and its FOODON match have different embeddings is separate spaces).
    Create 2 files, each with list of ingredients/entities and np matrix with embeddings
    """
    with open(
        os.path.join(old_data_path, "eatpim_triple_data/entities.dict"), "r"
    ) as file:
        data = {}
        for line in file:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                key, value = parts
                data[key] = value
        data = np.asarray(list(data.values()))

    embeddings = np.load(
        os.path.join(old_data_path, "models/result_model/entity_embedding.npy")
    )

    assert data.shape[0] == embeddings.shape[0]

    mask = [not i.startswith("RECIPE_OUTPUT") for i in data]
    data = data[mask]
    embeddings = embeddings[mask]

    assert data.shape[0] == embeddings.shape[0]

    mask = np.array([not i.startswith("http") for i in data])
    data_ing = data[mask]
    embeddings_ing = embeddings[mask]
    data_ent = data[~mask]
    embeddings_ent = embeddings[~mask]

    assert (
        data_ing.shape[0] == embeddings_ing.shape[0]
        and data_ent.shape[0] == embeddings_ent.shape[0]
    )

    normalized_embeddings = (
        embeddings_ing / np.linalg.norm(embeddings_ing, axis=1)[:, np.newaxis]
    )
    cosine_sim_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
    data = {
        "entities": data_ing,
        "embeddings": embeddings_ing,
        "cosine_matrix": cosine_sim_matrix,
    }
    with open(
        os.path.join(new_data_path, config.INGREDIENTS_EMBEDDINGS_FILE), "wb"
    ) as pickle_file:
        pickle.dump(data, pickle_file)

    normalized_embeddings = (
        embeddings_ent / np.linalg.norm(embeddings_ent, axis=1)[:, np.newaxis]
    )
    cosine_sim_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
    data = {
        "entities": data_ent,
        "embeddings": embeddings_ent,
        "cosine_matrix": cosine_sim_matrix,
    }
    with open(
        os.path.join(new_data_path, config.FOODON_ENTITIES_EMBEDDINGS_FILE), "wb"
    ) as pickle_file:
        pickle.dump(data, pickle_file)


def parse_actions_embeddings():
    """
    Parses actions embeddings. Create a file with list of actions and np matrix of embeddings
    """
    with open(
        os.path.join(old_data_path, "eatpim_triple_data/relations.dict"), "r"
    ) as file:
        data = {}
        for line in file:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                key, value = parts
                data[key] = value
        data = list(data.values())

    data = [i.replace("pred ", "") for i in data]

    embeddings = np.load(
        os.path.join(old_data_path, "models/result_model/relation_embedding.npy")
    )

    assert len(data) == embeddings.shape[0]
    data = {"actions": data, "embeddings": embeddings}

    with open(
        os.path.join(new_data_path, config.ACTIONS_EMBEDDING_FILE), "wb"
    ) as pickle_file:
        pickle.dump(data, pickle_file)


def parse_cooc():
    """
    Parses co-occurencies. Create file with cooc matrix and similarity matrix
    """
    with open(os.path.join(old_data_path, "ing_occ_data.pkl"), "rb") as file:
        data = pickle.load(file)

    ingredients = list(data["ing_to_index"].keys())
    co_occurrence_matrix = data["ing_cooc_matrix"].toarray()

    epsilon = 1e-8
    row_norms = np.linalg.norm(co_occurrence_matrix, axis=1)
    normalized_co_occurrence_matrix = co_occurrence_matrix / (
        row_norms[:, np.newaxis] + epsilon
    )
    cosine_sim_matrix = np.dot(
        normalized_co_occurrence_matrix, normalized_co_occurrence_matrix.T
    )

    assert (
        (len(ingredients), len(ingredients))
        == co_occurrence_matrix.shape
        == cosine_sim_matrix.shape
    )

    data = {
        "ing_to_index": data["ing_to_index"],
        "ing_cooc_matrix": data["ing_cooc_matrix"],
    }
    data["sim_matrix"] = cosine_sim_matrix

    with open(os.path.join(new_data_path, config.COOC_FILE), "wb") as pickle_file:
        pickle.dump(data, pickle_file)


def main():
    parse_matches()
    parse_flow_graphs()
    parse_entities_embeddings()
    parse_actions_embeddings()
    parse_cooc()


if __name__ == "__main__":
    if os.path.exists(new_data_path) and os.path.isdir(new_data_path):
        for file in os.listdir(new_data_path):
            file_path = os.path.join(new_data_path, file)
            if file != config.INGREDIENTS_FILE:
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                    print(f"Deleted: {file}")
                except Exception as e:
                    print(f"Error deleting {file}: {str(e)}")
    else:
        print(f"The folder '{new_data_path}' does not exist. Creating ...")
        os.makedirs(new_data_path)

    main()
