class Config:
    OLD_DATA_PATH: str = "eat_pim/data/result/"
    NEW_DATA_PATH: str = "src/data/"

    INGREDIENTS_FILE: str = "ingredients.json"
    COOC_FILE: str = "ingredients_cooc.pkl"
    FULL_SHORT_FILE: str = "full_short_ingredients_matches.pkl"
    FOODON_MATCHES_FILE: str = "foodon_matches.pkl"
    WIKIDATA_MATCHES_FILE: str = "wikidata_matches.pkl"
    FLOW_GRAPS_FILE: str = "flow_graphs.pkl"
    INGREDIENTS_EMBEDDINGS_FILE: str = "ingredients_embeddings.pkl"
    FOODON_ENTITIES_EMBEDDINGS_FILE: str = "foodon_entities_embeddings.pkl"
    ACTIONS_EMBEDDING_FILE: str = "actions_embeddings.pkl"

    MIN_INGREDIENTS: int = 5


config = Config()
