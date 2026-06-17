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
        ngram_range=(2, 3),
        min_df=4
    )
    return vectorizer.fit_transform(texts)


# ======================
# DIAGNOSTICA DATASET
# ======================

def dataset_diagnostics(X):
    print("\n" + "=" * 60)
    print("📊 DIAGNOSTICA TF-IDF SPACE")
    print("=" * 60)

    mean_density = np.mean(X.mean(axis=1))
    print(f"🔹 Mean vector density: {mean_density:.6f}")

    sample = X[:200] if X.shape[0] > 200 else X
    sim_matrix = cosine_similarity(sample, sample)
    mean_sim = sim_matrix[np.triu_indices_from(sim_matrix, k=1)].mean()

    print(f"🔹 Mean pairwise cosine similarity (sample): {mean_sim:.4f}")

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

    # Feature building e TF-IDF
    df["blocking_text"] = df.apply(build_blocking_text, axis=1)
    X = build_tfidf_matrix(df)

    # Diagnostica dello spazio vettoriale
    dataset_diagnostics(X)

    n = len(df)
    # Usiamo un array booleano per il pool: True significa che l'elemento è ancora disponibile
    pool_mask = np.ones(n, dtype=bool)
    canopies = {}

    # ============================================================
    # SOGLIE ALLINEATE ALLA TEORIA: T1 (Loose) < T2 (Tight)
    # ============================================================
    T1 = 0.30  # Soglia LARGA: per entrare nel cluster (Canopy)
    T2 = 0.40  # Soglia STRETTA: per essere rimossi dal pool
    # ============================================================

    print("=" * 60)
    print("🚀 CANOPY CLUSTERING CON TF-IDF (OTTIMIZZATO)")
    print(f"Dataset: {n}")
    print(f"Soglie: T1 (Loose)={T1} | T2 (Tight)={T2}")
    print("=" * 60)

    with open(cluster_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Cluster_ID", "ID", "Title", "Year", "Director", "Cast", "Genre", "Duration"])

    cluster_id = 0

    # Ciclo principale
    while np.any(pool_mask):
        cluster_id += 1

        # Scegli il primo indice disponibile nel pool come centroide
        centro_id = np.where(pool_mask)[0][0]
        
        # Il centroide viene rimosso a prescindere dal pool
        pool_mask[centro_id] = False

        # Calcola la similarità tra il centroide e TUTTI i record in un colpo solo
        centro_vec = X[centro_id]
        sims = cosine_similarity(centro_vec, X).ravel()

        # Condizione per ENTRARE nel cluster (Soglia larga T1)
        # Il centroide entra di diritto nel suo cluster
        in_cluster_mask = (sims >= T1)
        in_cluster_mask[centro_id] = True
        blocco_indices = np.where(in_cluster_mask)[0]

        # Condizione per ESSERE RIMOSSI dal pool (Soglia stretta T2)
        # Rimuoviamo dal pool solo gli elementi rimasti che superano T2
        to_remove_mask = (sims >= T2) & pool_mask
        remove_count = np.sum(to_remove_mask)
        
        # Aggiorna il pool escludendo i record troppo vicini
        pool_mask[to_remove_mask] = False

        canopies[centro_id] = blocco_indices.tolist()

        # Scrittura su file dei record del cluster corrente
        with open(cluster_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for fid in blocco_indices:
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
            f"size={len(blocco_indices)} | "
            f"removed={remove_count} | "
            f"pool={np.sum(pool_mask)}"
        )

    end_time = time.time()
    print("=" * 60)
    print(f"🏁 COMPLETATO: {cluster_id} cluster")
    print(f"⏱️ Tempo totale: {end_time - start_time:.2f} secondi")
    print("=" * 60)

    return canopies