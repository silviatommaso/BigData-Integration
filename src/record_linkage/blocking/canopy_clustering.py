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



def build_tfidf_matrices(df, columns, tfidf_params=None):
    """Generates separate TF-IDF matrices"""

    matrices = {}

    default_params = {
        "analyzer": "char_wb",
        "ngram_range": (2, 3),
        "min_df": 2
    }

    for col in columns:

        params = default_params.copy()

        if tfidf_params and col in tfidf_params:
            params.update(tfidf_params[col])

        vectorizer = TfidfVectorizer(**params)

        try:
            matrices[col] = vectorizer.fit_transform(
                df[f"clean_{col}"].tolist()
            )

        except ValueError:
            matrices[col] = np.zeros((len(df), 1))

    return matrices


# ======================
# CANOPY CLUSTER
# ======================

def canopy_cluster(merged_df, cluster_path, canopy_params):

    start_time = time.time()
    df = merged_df.copy()
    
    columns = canopy_params["columns"]
    thresholds = canopy_params["thresholds"]
    tfidf_params = canopy_params.get("tfidf", {})

    for col in columns:
        clean_col = f"clean_{col}"

        if col in df.columns:
            df[clean_col] = df[col].apply(clean_text)
        else:
            df[clean_col] = ""

    X = build_tfidf_matrices(df, columns, tfidf_params)

    n = len(df)

    pool_mask = np.ones(n, dtype=bool)
    canopies = {}

    print("=" * 60)
    print("CANOPY MULTI-CHANNEL")
    print(f"Columns: {columns}")
    print(f"Dataset size: {n}")
    print("=" * 60)

    with open(cluster_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Cluster_ID"] + merged_df.columns.tolist())

    has_values = {}

    for col in columns:
        has_values[col] = (
            df[f"clean_{col}"] != ""
        ).to_numpy()

    cluster_id = 0

    while np.any(pool_mask):

        cluster_id += 1

        centro_id = np.where(pool_mask)[0][0]
        pool_mask[centro_id] = False

        similarities = {}

        for col in columns:
            similarities[col] = cosine_similarity(
                X[col][centro_id],
                X[col]
            ).ravel()

        # T1
        in_cluster_mask = np.ones(n, dtype=bool)

        for col in columns:

            T1 = thresholds[col][0]

            both_have_value = (
                has_values[col][centro_id]
                &
                has_values[col]
            )

            condition = np.where(
                both_have_value,
                similarities[col] >= T1,
                True
            )

            in_cluster_mask &= condition

        in_cluster_mask[centro_id] = True

        blocco_indices = np.where(in_cluster_mask)[0]


        # T2
        to_remove_mask = np.ones(n, dtype=bool)

        for col in columns:

            T2 = thresholds[col][1]

            both_have_value = (
                has_values[col][centro_id]
                &
                has_values[col]
            )

            condition = np.where(
                both_have_value,
                similarities[col] >= T2,
                True
            )

            to_remove_mask &= condition

        to_remove_mask &= pool_mask

        remove_count = np.sum(to_remove_mask)

        pool_mask[to_remove_mask] = False


        canopies[centro_id] = blocco_indices.tolist()


        with open(cluster_path, "a", newline="", encoding="utf-8") as f:

            writer = csv.writer(f)

            for fid in blocco_indices:

                row = df.iloc[fid]

                output_row = []

                for col in merged_df.columns:

                    clean_col = f"{col}_clean"

                    if clean_col in df.columns:
                        output_row.append(row[clean_col])
                    else:
                        output_row.append(row[col])

                writer.writerow(
                    [cluster_id] + output_row
                )


        print(
            f"Cluster #{cluster_id} | "
            f"size={len(blocco_indices)} | "
            f"removed={remove_count} | "
            f"pool={np.sum(pool_mask)}"
        )


    end_time = time.time()

    print("=" * 60)
    print(f"Completed: {cluster_id} clusters")
    print(f"Total execution time: {end_time - start_time:.2f}s")
    print("=" * 60)

    return canopies