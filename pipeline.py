from normalizator import normalizer
from canopy_clustering import canopy_cluster
from record_matching import generate_candidate_pairs, match_records
from entity_clustering import build_clusters, get_unmatched_records, save_clusters
import utils

from pathlib import Path
import pandas as pd
import os


SCHEMA_ALIGN = False
RECORD_LINKAGE = False
RECORD_MATCHING = False
CLUSTERING = True


INPUT_DIR = Path("normalized_csv")


inputs = [
    INPUT_DIR / "movies3_cleaned_imdb_cleaned.csv",
    INPUT_DIR / "movies3_cleaned_rotten_tomatoes_cleaned.csv",
    INPUT_DIR / "movies5_cleaned_imdb_cleaned.csv",
    INPUT_DIR / "movies5_cleaned_roger_ebert_final.csv"
]


# files directories
files = {

    "Step I" : [
        "schema_alignment_csv/merged_movies.csv", "Schema Alignment"
    ],

    "Step II" : [
        "record_linkage_csv/canopy_blocks.csv",
        "",
        "Record Linkage"
    ]

}


# =====================================================
# STEP I
# SCHEMA ALIGNMENT
# =====================================================

if SCHEMA_ALIGN:

    normalizer("dataset_cleaned/movies3_cleaned/imdb_cleaned.csv", "a")
    normalizer("dataset_cleaned/movies3_cleaned/rotten_tomatoes_cleaned.csv", "b")
    normalizer("dataset_cleaned/movies5_cleaned/roger_ebert_final.csv", "c")
    normalizer("dataset_cleaned/movies5_cleaned/imdb_cleaned.csv", "d")

    dfs = [utils.load_movies_csv(f) for f in inputs]

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

if RECORD_LINKAGE:

    utils.path_check(files["Step I"])


    df = utils.load_movies_csv(files["Step I"][0])

    canopies = canopy_cluster(
        df,
        "record_linkage_csv/canopy_blocks.csv"
    )


# =====================================================
# STEP III
# RECORD MATCHING
# =====================================================

if RECORD_MATCHING:


    utils.subpath_check(files, [0], 3)

    df = utils.load_movies_csv("schema_alignment_csv/merged_movies.csv")
    canopy_df = utils.load_movies_csv(canopy_file)

    canopies = {}

    for cluster_id, group in canopy_df.groupby("Cluster_ID"):
        canopies[cluster_id] = list(group["ID"])

    candidate_pairs = generate_candidate_pairs(canopies)

    print("Candidate pairs:", len(candidate_pairs))

    matches = match_records(
        df,
        candidate_pairs,
        threshold=0.8
    )

    matches.to_csv(
        "record_linkage_csv/matches.csv",
        index=False
    )

    unmatched = get_unmatched_records(
        matches,
        df
    )

    unmatched.to_csv(
        "record_linkage_csv/unmatched_records.csv",
        index=False
    )

    print("Match trovati:", len(matches))
    print("Record senza match:", len(unmatched))


# =====================================================
# STEP IV
# CLUSTERING
# =====================================================

if CLUSTERING:

    matches_file = "record_linkage_csv/matches.csv"

    if not os.path.exists(matches_file):
        print("Error: matches.csv not found, execute record matching first")
        exit()

    df = utils.load_movies_csv("schema_alignment_csv/merged_movies.csv")
    matches = pd.read_csv(matches_file)

    entities = build_clusters(matches)

    print("EntitÃ  trovate:", len(entities))

    save_clusters(
        entities,
        "record_linkage_csv/entity_clusters.csv"
    )

    singletons = get_unmatched_records(
        matches,
        df
    )

    singletons.to_csv(
        "record_linkage_csv/singleton_records.csv",
        index=False
    )

    print("Clusters salvati")
    print("Record singoli:", len(singletons))