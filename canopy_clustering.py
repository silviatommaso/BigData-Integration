import csv
import time
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ======================
# TEXT PREPROCESSING
# ======================

def clean_text(value):
    """Cleans text by standardizing punctuation and spaces."""
    if pd.isna(value) or str(value).strip().lower() in ["null", "<na>", "nan", ""]:
        return ""

    text_clean = str(value).strip().lower()

    for char in [",", ".", "-", "(", ")", "[", "]", "/", "\\", ":", "_"]:
        text_clean = text_clean.replace(char, " ")

    return " ".join(text_clean.split())



def build_tfidf_matrices(df):
    """Generates separate TF-IDF matrices for Title and Director."""

    titles = df["clean_title"].tolist()
    directors = df["clean_director"].tolist()

    # Character n-grams are used to better handle spelling variations and dataset sparisity
    vectorizer_title = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 3),
        min_df=2
    )

    vectorizer_dir = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 3),
        min_df=2
    )

    X_title = vectorizer_title.fit_transform(titles)
    X_dir = vectorizer_dir.fit_transform(directors)

    return X_title, X_dir


# ======================
# CANOPY CLUSTER
# ======================

def canopy_cluster(merged_df, cluster_path):

    start_time = time.time()
    df = merged_df.copy()

    # Preprocess key attributes
    df["clean_title"] = df["Title"].apply(clean_text)
    df["clean_director"] = df["Director"].apply(clean_text)


    # Build separate TF-IDF matrices
    X_title, X_dir = build_tfidf_matrices(df)


    n = len(df)

    pool_mask = np.ones(n, dtype=bool)
    canopies = {}

    # ======================
    # THRESHOLD CONFIGURATION
    # ======================

    # Title similarity thresholds
    T1_title = 0.40
    T2_title = 0.70

    # Director similarity thresholds
    # Stricter values reduce false matches caused by common names
    T1_dir = 0.40
    T2_dir = 0.65

    print("=" * 60)
    print("CANOPY MULTI-CHANNEL (TITLE + DIRECTOR)")
    print(f"Dataset size: {n}")
    print("=" * 60)

    with open(cluster_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Cluster_ID"] + merged_df.columns.tolist())



    cluster_id = 0

    # Cache whether each record has a director value
    has_director = (df["clean_director"] != "").to_numpy()

    while np.any(pool_mask):
        
        cluster_id += 1

        # Select the first available record as the cluster center
        centro_id = np.where(pool_mask)[0][0]
        pool_mask[centro_id] = False

        # Compute title and director similarities separately
        sims_title = cosine_similarity(X_title[centro_id],X_title).ravel()
        sims_dir = cosine_similarity(X_dir[centro_id],X_dir).ravel()

        # If both records have a director, both title and director
        # similarities must satisfy the threshold.
        # If one director value is missing, only title similarity is used.
        centro_has_dir = has_director[centro_id]

        both_have_director = centro_has_dir & has_director

        # Loose threshold:
        # determines which records enter the canopy

        in_cluster_mask = np.where(both_have_director, (sims_title >= T1_title) & (sims_dir >= T1_dir), (sims_title >= T1_title))
        in_cluster_mask[centro_id] = True

        blocco_indices = np.where(in_cluster_mask)[0]


        # Tight threshold:
        # determines which records are removed from the candidate pool

        to_remove_mask = np.where(both_have_director, (sims_title >= T2_title) & (sims_dir >= T2_dir), (sims_title >= T2_title))
        to_remove_mask = to_remove_mask & pool_mask

        remove_count = np.sum(to_remove_mask)

        pool_mask[to_remove_mask] = False


        canopies[centro_id] = blocco_indices.tolist()

        # Save current cluster to file
        with open(cluster_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            for fid in blocco_indices:
                row = df.iloc[fid]
                writer.writerow([cluster_id] + [row.get(col) for col in df.columns])


        print(
            f"Cluster #{cluster_id} | "
            f"size={len(blocco_indices)} | "
            f"removed={remove_count} | "
            f"pool={np.sum(pool_mask)}"
        )


    end_time = time.time()


    print("=" * 60)
    print(f"Completed: {cluster_id} clusters")
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
    print("=" * 60)

    return canopies