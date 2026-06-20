import pandas as pd
import numpy as np

def column_profile_extraction(column):
    """
    Column column_profile -> stats of a column that helps us distinguishing it from others
    """
    # Null values removed
    cleaned_data = column.dropna()
    if cleaned_data.empty:
        return None
        
    # Type conversion (anytype -> string)
    rows = cleaned_data.astype(str).str.strip()
    tot_rows = len(rows)

    # row length
    row_length = rows.str.len()
    # single cell length (per row)
    word_count = rows.str.split().str.len()

    #----------------------------------------------------------------------------------------------------
    # NUMERICAL and CATEGORICAL DENSITY CHECK
    #----------------------------------------------------------------------------------------------------
    # - Numeric values check allows the distinction between numeric and text columns
    # - Values cardinality allows the distinction between categorical (genres) and non values (Title, IDs)
    #-----------------------------------------------------------------------------------------------------

    # numeric values check
    numeric_percentage = rows.str.count(r'\d').sum() / row_length.sum() if row_length.sum() > 0 else 0
    
    # values cardinality
    values_cardinality = len(rows.unique()) / tot_rows

    column_profile = {
        "avg_length": float(row_length.mean()),
        "avg_words": float(word_count.mean()),
        "numeric_percentage": float(numeric_percentage),
        "cardinality": float(values_cardinality),
        "is_purely_numeric": False,
        "avg": 0.0,
        "min": 0.0,
        "max": 0.0
    }

    #----------------------------------------------------------------------------------------------------
    # NUMERICAL VALUES DISTINCTION PARAMETERS
    #----------------------------------------------------------------------------------------------------
    # If a value is numerical, calculate its statistics (min, max, mean), to distinguish it from others
    # numerical columns
    #-----------------------------------------------------------------------------------------------------

    converted_num = pd.to_numeric(column.copy(), errors='coerce').dropna()
    # if at least 70% of data are numerical...
    if len(converted_num) / tot_rows > 0.7:  
        column_profile["is_purely_numeric"] = True
        column_profile["avg"] = float(converted_num.mean())
        column_profile["min"] = float(converted_num.min())
        column_profile["max"] = float(converted_num.max())

    return column_profile


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def profile_comparison(dfs, dfs_name):
    """
    Prende una lista di DataFrame e una lista con i loro nomi.
    Calcola i profili di TUTTE le colonne di TUTTI i dataset e le confronta.
    """
    profiles = {}
    
    # column_profile extraction (per column per dataframe)
    for df, df_name in zip(dfs, dfs_name):
        for col in df.columns:

            # unique key per dataframe
            unique_key = f"{df_name}.{col}"
            
            column_profile = column_profile_extraction(df[col])
            
            if column_profile is not None:
                profiles[unique_key] = column_profile

    # columns profile comparison
    keys = list(profiles.keys())
    global_results = {}
    
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):

            key_a = keys[i]
            key_b = keys[j]
            
            file_A = key_a.split('.')[0]
            file_B = key_b.split('.')[0]
            if file_A == file_B:
                continue
                
            prof_A = profiles[key_a]
            prof_B = profiles[key_b]
            
            #--------------------
            # BASE SCORE
            #--------------------
            max_len = max(prof_A["avg_length"], prof_B["avg_length"])
            sim_length = 1.0 - (abs(prof_A["avg_length"] - prof_B["avg_length"]) / max_len) if max_len > 0 else 1.0
            
            max_words = max(prof_A["avg_words"], prof_B["avg_words"])
            sim_words = 1.0 - (abs(prof_A["avg_words"] - prof_B["avg_words"]) / max_words) if max_words > 0 else 1.0
            
            sim_density = 1.0 - abs(prof_A["numeric_percentage"] - prof_B["numeric_percentage"])
            
            sim_values_cardinality = 1.0 - abs(prof_A["cardinality"] - prof_B["cardinality"])

            # Base score (Weights: 30% length, 30% words, 20% numbers, 20% cardinality)
            score_structure = (sim_length * 0.3) + (sim_words * 0.3) + (sim_density * 0.2) + (sim_values_cardinality * 0.2)

            #--------------------
            # FINAL SCORE
            #--------------------
            if prof_A["is_purely_numeric"] and prof_B["is_purely_numeric"]:

                # range overlap
                min_shared = max(prof_A["min"], prof_B["min"])
                max_shared = min(prof_A["max"], prof_B["max"])
                
                if min_shared <= max_shared:
                    union_range = max(prof_A["max"], prof_B["max"]) - min(prof_A["min"], prof_B["min"])
                    sim_range = (max_shared - min_shared) / union_range if union_range > 0 else 1.0
                else:
                    sim_range = 0.0
                    
                # average overlap
                max_avg = max(abs(prof_A["avg"]), abs(prof_B["avg"]))
                sim_avg = 1.0 - (abs(prof_A["avg"] - prof_B["avg"]) / max_avg) if max_avg > 0 else 1.0
                
                score_numeric = (sim_range * 0.5) + (sim_avg * 0.5)
                
                # textual + numerical --> trust num more
                final_score = round((score_structure * 0.4) + (score_numeric * 0.6), 4)
            else:
                # both textual
                final_score = round(score_structure, 4)
                
            # Salva il risultato finale nel dizionario globale
            global_results[(key_a, key_b)] = final_score

    return global_results


if __name__ == "__main__":
    # 1. Carica i dataset
    df1 = pd.read_csv("dataset_cleaned/movies3_cleaned/imdb_cleaned.csv")
    df2 = pd.read_csv("dataset_cleaned/movies5_cleaned/roger_ebert_cleaned.csv")

    # 2. Prepara le liste per la funzione (puoi aggiungere altri df qui dentro in futuro)
    lista_dfs = [df1, df2]
    nomi_dfs = ["imdb", "roger_ebert"]

    print(f"Avvio estrazione e confronto profili per {len(lista_dfs)} dataset...")
    print("-" * 70)
    
    # 3. Calcola i risultati globali
    risultati = profile_comparison(lista_dfs, nomi_dfs)
    
    # 4. Stampa i risultati ordinati dal match più probabile a scendere
    for coppie, score in sorted(risultati.items(), key=lambda item: item[1], reverse=True):
        # Filtriamo stampando solo i match con un minimo di senso (es. > 0.4) per evitare troppo log
        if score > 0.4:
            print(f"Match: {coppie[0]}  <--->  {coppie[1]}  |  Score: {score}")