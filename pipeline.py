from normalizator import normalizer
from schema_alignment import schema_alignment
from canopy_clustering import canopy_cluster
from record_matching import match_records
from entity_clustering import build_clusters
from data_fusion import fuse_cluster
import utils

from pathlib import Path
import pandas as pd
import os


SCHEMA_ALIGN = True
BLOCKING = False
RECORD_MATCHING = False
CLUSTERING = False
DATA_FUSION = False


INPUT_DIR = Path("normalized_csv")


inputs = [
    INPUT_DIR / "movies3_cleaned_imdb_cleaned.csv",
    INPUT_DIR / "movies3_cleaned_rotten_tomatoes_cleaned.csv",
    INPUT_DIR / "movies5_cleaned_imdb_cleaned.csv",
    INPUT_DIR / "movies5_cleaned_roger_ebert_cleaned.csv"
]


# files directories
files = {

    "Step I" : [
        "schema_alignment_csv/merged_movies.csv",
        "schema_alignment_csv/global_schema.csv",
        "Schema Alignment"
    ],

    "Step II" : [
        "record_linkage_csv/canopy_blocks.csv",
        "record_linkage_csv/matches.csv",
        "record_linkage_csv/singletons_records.csv",
        "record_linkage_csv/entity_clusters.csv",
        "Record Linkage"
    ],

    "Step III" : [
        "data_fusion_csv/fused_entities.csv",
        "Data Fusion"
    ]

}





# =====================================================
# STEP I
# SCHEMA ALIGNMENT
# =====================================================

if SCHEMA_ALIGN:

    # preprocessing
    normalizer("dataset_cleaned/movies3_cleaned/imdb_cleaned.csv", "a")
    normalizer("dataset_cleaned/movies3_cleaned/rotten_tomatoes_cleaned.csv", "b")
    normalizer("dataset_cleaned/movies5_cleaned/roger_ebert_cleaned.csv", "c")
    normalizer("dataset_cleaned/movies5_cleaned/imdb_cleaned.csv", "d")


    dataset_names = ["imdb_v3", "roger_ebert", "imdb_v5", "rotten_tomatoes"]
    dfs = [utils.load_movies_csv(f) for f in inputs]

    attributes = schema_alignment(dfs, dataset_names, files["Step I"][1])

    for df in dfs:
        df.columns = attributes

    merged_df = pd.concat(dfs, ignore_index=True)

    merged_df.to_csv(
        files["Step I"][0],
        index=False
    )

    print("Totale record:", len(merged_df))






# =====================================================
# STEP II
# RECORD LINKAGE
# =====================================================
# BLOCKING
# =====================================================

if BLOCKING:

    utils.path_check(files["Step I"])


    df = utils.load_movies_csv(files["Step I"][0])

    canopies = canopy_cluster(
        df,
        "record_linkage_csv/canopy_blocks.csv"
    )


# =====================================================
# RECORD MATCHING
# =====================================================

if RECORD_MATCHING:

    utils.subpath_check(files, [0], 3)

    merged_df = utils.load_movies_csv(files["Step I"][0])
    canopy_df = utils.load_movies_csv(files["Step II"][0])


    matches = match_records(
        merged_df,
        canopy_df,
        files["Step II"][1],
        files["Step II"][2],
        threshold=0.72
    )


# =====================================================
# CLUSTERING
# =====================================================

if CLUSTERING:

    utils.subpath_check(files, [0], 3)
    

    merged_df = utils.load_movies_csv(files["Step I"][0])
    matched_df = utils.load_movies_csv(files["Step II"][1])

    entities = build_clusters(matched_df, merged_df, files["Step II"][3])






# =====================================================
# STEP III
# DATA FUSION
# =====================================================
if DATA_FUSION:

    entities_cluster = utils.load_movies_csv(files["Step II"][3])

    all_fused = [fuse_cluster(group) for _, group in entities_cluster.groupby("entity_id")]

    # Unisci tutti i singoli film in un unico DataFrame
    final_df = pd.concat(all_fused, ignore_index=True)
    
    # Salva il file definitivo (una riga per ogni film reale)
    if not files["Step III"][0]:
        os.makedirs(files["Step III"][0], exist_ok=True)
    final_df.to_csv(files["Step III"][0], index=False)
    
    print("Data Fusion successfully completed!")