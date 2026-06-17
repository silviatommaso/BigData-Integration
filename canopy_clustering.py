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
    """Pulisce il testo standardizzando la punteggiatura e gli spazi."""
    if pd.isna(value) or str(value).strip().lower() in ["null", "<na>", "nan", ""]:
        return ""
    text_clean = str(value).strip().lower()
    for char in [",", ".", "-", "(", ")", "[", "]", "/", "\\", ":", "_"]:
        text_clean = text_clean.replace(char, " ")
    return " ".join(text_clean.split())


def build_tfidf_matrices(df):
    """Genera matrici TF-IDF separate per Titolo e Regista."""
    titles = df["clean_title"].tolist()
    directors = df["clean_director"].tolist()
    
    # Utilizziamo impostazioni iper-ottimizzate per gli n-grammi di caratteri
    vectorizer_title = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), min_df=2)
    vectorizer_dir = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), min_df=2)
    
    X_title = vectorizer_title.fit_transform(titles)
    X_dir = vectorizer_dir.fit_transform(directors)
    
    return X_title, X_dir


# ======================
# CANOPY CLUSTER
# ======================

def canopy_cluster(df, cluster_path):
    start_time = time.time()
    df = df.copy()

    # Pre-pulizia dei campi chiave
    df["clean_title"] = df["Title"].apply(clean_text)
    df["clean_director"] = df["Director"].apply(clean_text)
    
    # Costruzione delle matrici separate
    X_title, X_dir = build_tfidf_matrices(df)

    n = len(df)
    pool_mask = np.ones(n, dtype=bool)
    canopies = {}

    # ============================================================
    # SOGLIE STRUTTURATE SUI DUE CANALI SEPARATI
    # ============================================================
    # Soglie per il Titolo
    T1_title = 0.40  
    T2_title = 0.70  
    
    # Soglie per il Regista (più severe per evitare falsi match di omonimia)
    T1_dir = 0.40    
    T2_dir = 0.65    
    # ============================================================

    print("=" * 60)
    print("🚀 CANOPY MULTI-CANALE (TITLE + DIRECTOR) OTTIMIZZATO")
    print(f"Dataset size: {n}")
    print("=" * 60)

    with open(cluster_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Cluster_ID", "ID", "Title", "Year", "Director", "Cast", "Genre", "Duration"])

    cluster_id = 0

    # Cache dei flag "director presente" per velocizzare il ciclo
    has_director = (df["clean_director"] != "").to_numpy()

    while np.any(pool_mask):
        cluster_id += 1

        # Scegli il primo indice disponibile nel pool come centroide
        centro_id = np.where(pool_mask)[0][0]
        pool_mask[centro_id] = False

        # 1. Calcola le similarità separate per Titolo e Regista
        sims_title = cosine_similarity(X_title[centro_id], X_title).ravel()
        sims_dir = cosine_similarity(X_dir[centro_id], X_dir).ravel()

        # 2. Logica di fusione condizionale (Gestione del Director NULL)
        # Se il centroide NON ha il regista, o il film target NON ha il regista, ci fidiamo solo del titolo
        centro_has_dir = has_director[centro_id]
        both_have_director = centro_has_dir & has_director
        
        # Maschera Finale Loose (Per entrare nel cluster)
        in_cluster_mask = np.where(
            both_have_director,
            (sims_title >= T1_title) & (sims_dir >= T1_dir),  # Se entrambi hanno il regista, devono passare entrambi i controlli
            (sims_title >= T1_title)                          # Se uno dei due è NULL, basta il titolo
        )
        in_cluster_mask[centro_id] = True                     # Il centroide entra di diritto
        blocco_indices = np.where(in_cluster_mask)[0]

        # Maschera Finale Tight (Per essere rimossi dal pool)
        to_remove_mask = np.where(
            both_have_director,
            (sims_title >= T2_title) & (sims_dir >= T2_dir),
            (sims_title >= T2_title)
        )
        to_remove_mask = to_remove_mask & pool_mask
        remove_count = np.sum(to_remove_mask)
        
        # Aggiorna il pool
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