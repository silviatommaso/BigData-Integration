import csv
import time
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ======================
# TEXT PREPROCESSING
# ======================

def build_blocking_text(row):
    parts = []

    for col in ["Title", "Director", "Year"]:
        value = row.get(col)

        if pd.notna(value) and str(value).strip().lower() != "null":
            text_clean = str(value).strip().lower()

            for char in [",", ".", "-", "(", ")", "[", "]", "/", "\\", ":", "_"]:
                text_clean = text_clean.replace(char, " ")

            parts.append(" ".join(text_clean.split()))

    return " ".join(parts)


def build_tfidf_matrix(df):
    texts = df["blocking_text"].fillna("").tolist()

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(1, 2),
        min_df=10
    )

    return vectorizer.fit_transform(texts)


# ======================
# DIAGNOSTICA DATASET
# ======================

def dataset_diagnostics(X):

    print("\n" + "=" * 60)
    print("📊 DIAGNOSTICA TF-IDF SPACE")
    print("=" * 60)

    # 1. densità media (quanto sono sparsi i vettori)
    mean_density = np.mean(X.mean(axis=1))
    print(f"🔹 Mean vector density: {mean_density:.6f}")

    # 2. similarità media su sample
    sample = X[:200] if X.shape[0] > 200 else X
    sim_matrix = cosine_similarity(sample, sample)
    mean_sim = sim_matrix[np.triu_indices_from(sim_matrix, k=1)].mean()

    print(f"🔹 Mean pairwise cosine similarity (sample): {mean_sim:.4f}")

    # 3. interpretazione automatica
    if mean_sim < 0.1:
        print("⚠️ Dataset MOLTO sparso → canopy avrà cluster piccoli")
    elif mean_sim < 0.3:
        print("⚠️ Dataset moderatamente sparso")
    else:
        print("✅ Dataset abbastanza denso")

    print("=" * 60 + "\n")


# ======================
# CANOPY CLUSTER
# ======================

def canopy_cluster(df, cluster_path):

    start_time = time.time()

    df = df.copy()

    if "Year" in df.columns:
        df["Year"] = df["Year"].astype(str)

    # ======================
    # FEATURE BUILDING
    # ======================
    df["blocking_text"] = df.apply(build_blocking_text, axis=1)

    # ======================
    # TF-IDF MATRIX
    # ======================
    X = build_tfidf_matrix(df)

    # 🔥 DIAGNOSTICA AGGIUNTA QUI
    dataset_diagnostics(X)

    n = len(df)
    pool = set(range(n))
    canopies = {}

    # ======================
    # THRESHOLDS
    # ======================
    T1 = 0.60
    T2 = 0.45 

    print("=" * 60)
    print("🚀 CANOPY CLUSTERING CON TF-IDF")
    print(f"Dataset: {n}")
    print(f"Soglie: T1={T1} | T2={T2}")
    print("=" * 60)

    with open(cluster_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Cluster_ID", "ID", "Title", "Year", "Director", "Cast", "Genre", "Duration"])

    cluster_id = 0

    # ======================
    # MAIN LOOP
    # ======================
    while pool:

        cluster_id += 1

        centro_id = next(iter(pool))
        pool.remove(centro_id)

        centro_vec = X[centro_id]

        sims = cosine_similarity(centro_vec, X).ravel()

        blocco = []
        remove_set = []

        for i in pool:

            sim = sims[i]

            if sim >= T2:
                blocco.append(i)

            if sim >= T1:
                remove_set.append(i)

        pool.difference_update(remove_set)

        blocco.append(centro_id)
        canopies[centro_id] = blocco

        # ======================
        # OUTPUT RICH
        # ======================
        with open(cluster_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            for fid in blocco:
                row = df.iloc[fid]

                writer.writerow([
                    cluster_id,
                    row.get('ID'),
                    row.get('Title'),
                    row.get('Year'),
                    row.get('Director'),
                    row.get('Cast'),
                    row.get('Genre'),
                    row.get('Duration')
                ])

        print(
            f"➔ Cluster #{cluster_id} | "
            f"size={len(blocco)} | "
            f"removed={len(remove_set)} | "
            f"pool={len(pool)}"
        )

    end_time = time.time()

    print("=" * 60)
    print(f"🏁 COMPLETATO: {cluster_id} cluster")
    print(f"⏱️ Tempo totale: {end_time - start_time:.2f} secondi")
    print("=" * 60)

    return canopies