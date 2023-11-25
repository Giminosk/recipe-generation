import os
import json
import pickle
import numpy as np
import shutil


old_data_path = "eat_pim/data/result/"
new_data_path = "src/data/"


def parse_matches():
    with open(
        os.path.join(old_data_path, "word_cleanup_linking.json"), "r"
    ) as json_file:
        data = json.load(json_file)

    with open(
        os.path.join(new_data_path, "full_short_ingridients_matches.pkl"), "wb"
    ) as pickle_file:
        pickle.dump(data["ing_to_ing"], pickle_file)

    foodon_matches = data["ing_to_foodon"].copy()
    foodon_matches.update(data["obj_to_foodon"])
    with open(os.path.join(new_data_path, "foodon_matches.pkl"), "wb") as pickle_file:
        pickle.dump(foodon_matches, pickle_file)

    with open(os.path.join(new_data_path, "wikidata_matches.pkl"), "wb") as pickle_file:
        pickle.dump(data["verb_to_preparations"], pickle_file)


def parse_flow_graphs():
    with open(os.path.join(old_data_path, "recipe_tree_data.json"), "r") as json_file:
        data = json.load(json_file)

    with open(os.path.join(new_data_path, "flow_graphs.pkl"), "wb") as pickle_file:
        pickle.dump(data, pickle_file)


def parse_embeddings():
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
        os.path.join(new_data_path, "ingridients_embeddings.pkl"), "wb"
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
        os.path.join(new_data_path, "foodon_entities_embeddings.pkl"), "wb"
    ) as pickle_file:
        pickle.dump(data, pickle_file)


def parse_cooc():
    with open(os.path.join(old_data_path, "ing_occ_data.pkl"), "rb") as file:
        data = pickle.load(file)

    ingridients = list(data["ing_to_index"].keys())
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
        (len(ingridients), len(ingridients))
        == co_occurrence_matrix.shape
        == cosine_sim_matrix.shape
    )

    with open(os.path.join(new_data_path, "ingridients_cooc.pkl"), "wb") as pickle_file:
        pickle.dump(data, pickle_file)


def main():
    parse_matches()
    parse_flow_graphs()
    parse_embeddings()
    parse_cooc()


if __name__ == "__main__":
    folder_path = "data"

    # Delete the folder and its contents if it exists
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)

    # Create the folder
    os.makedirs(folder_path)

    main()
