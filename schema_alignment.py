import pandas as pd
import numpy as np
import difflib

def column_profile_extraction(column):
    """
    Extracts structural and mathematical footprints from ANY column,
    regardless of its native Pandas data type.
    """
    # Clean missing data
    cleaned_data = column.dropna()
    if cleaned_data.empty:
        return None
        
    # Universal string conversion for structural profiling
    rows = cleaned_data.astype(str).str.strip()
    tot_rows = len(rows)

    row_length = rows.str.len()
    word_count = rows.str.split().str.len()

    # Density and Categorical checks
    numeric_percentage = rows.str.count(r'\d').sum() / row_length.sum() if row_length.sum() > 0 else 0
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

    # Statistical profiling (attempt numerical conversion)
    converted_num = pd.to_numeric(column.copy(), errors='coerce').dropna()
    if len(converted_num) / tot_rows > 0.7:  # If more than 70% of rows are numeric
        column_profile["is_purely_numeric"] = True
        column_profile["avg"] = float(converted_num.mean())
        column_profile["min"] = float(converted_num.min())
        column_profile["max"] = float(converted_num.max())

    return column_profile


def calculate_name_similarity(name_A, name_B):
    """
    Computes syntactic similarity between two attribute names.
    Normalizes strings (lowercase, alphanumeric only) to avoid formatting mismatches.
    """
    clean_A = "".join(c for c in name_A.lower() if c.isalnum())
    clean_B = "".join(c for c in name_B.lower() if c.isalnum())
    return difflib.SequenceMatcher(None, clean_A, clean_B).ratio()


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def profile_comparison(dfs, dfs_name):
    """
    Takes a list of DataFrames and a list of their names.
    Compares ALL columns against each other, fusing schema metadata (names) and content (data profiles).
    """
    profiles = {}
    
    # Profile extraction
    for df, df_name in zip(dfs, dfs_name):
        for col in df.columns:
            unique_key = f"{df_name}.{col}"
            column_profile = column_profile_extraction(df[col])
            if column_profile is not None:
                profiles[unique_key] = column_profile

    # Cross-dataset comparison
    keys = list(profiles.keys())
    global_results = {}
    
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            key_a = keys[i]
            key_b = keys[j]
            
            # Split the universal key to extract dataset name and actual column name
            file_A, col_name_A = key_a.split('.', 1)
            file_B, col_name_B = key_b.split('.', 1)
            
            # Prevent self-matching columns from the same file
            if file_A == file_B:
                continue
                
            prof_A = profiles[key_a]
            prof_B = profiles[key_b]
            
            # Attribute Name Comparison (Syntactic)
            score_names = calculate_name_similarity(col_name_A, col_name_B)

            # Content Analysis (Structural)
            max_len = max(prof_A["avg_length"], prof_B["avg_length"])
            sim_length = 1.0 - (abs(prof_A["avg_length"] - prof_B["avg_length"]) / max_len) if max_len > 0 else 1.0
            
            max_words = max(prof_A["avg_words"], prof_B["avg_words"])
            sim_words = 1.0 - (abs(prof_A["avg_words"] - prof_B["avg_words"]) / max_words) if max_words > 0 else 1.0
            
            sim_density = 1.0 - abs(prof_A["numeric_percentage"] - prof_B["numeric_percentage"])
            sim_values_cardinality = 1.0 - abs(prof_A["cardinality"] - prof_B["cardinality"])

            score_structure = (sim_length * 0.5) + (sim_words * 0.25) + (sim_density * 0.2) + (sim_values_cardinality * 0.15)

            # Numerical Refinement (Statistical)
            if prof_A["is_purely_numeric"] and prof_B["is_purely_numeric"]:
                # Range overlap
                min_shared = max(prof_A["min"], prof_B["min"])
                max_shared = min(prof_A["max"], prof_B["max"])
                
                if min_shared <= max_shared:
                    union_range = max(prof_A["max"], prof_B["max"]) - min(prof_A["min"], prof_B["min"])
                    sim_range = (max_shared - min_shared) / union_range if union_range > 0 else 1.0
                else:
                    sim_range = 0.0
                    
                # Average closeness
                max_avg = max(abs(prof_A["avg"]), abs(prof_B["avg"]))
                sim_avg = 1.0 - (abs(prof_A["avg"] - prof_B["avg"]) / max_avg) if max_avg > 0 else 1.0
                
                score_numeric = (sim_range * 0.5) + (sim_avg * 0.5)
                score_data = (score_structure * 0.6) + (score_numeric * 0.4)
            else:
                score_data = score_structure
                
            # Hybrid Fusion (60% Data Content / 40% Metadata Names)
            final_score = round((score_data * 0.6) + (score_names * 0.4), 4)
            global_results[(key_a, key_b)] = final_score

    return global_results


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def extract_global_schema_clusters(global_results, dataset_names, threshold=0.9):
    """
    Groups pairwise matches into global tables
    """
    # Sort matches by confidence score descending to process the strongest links first
    sorted_matches = sorted(global_results.items(), key=lambda item: item[1], reverse=True)
    
    clusters = []
    
    for (node_a, node_b), score in sorted_matches:
        if score < threshold:
            break
            
        ds_a = node_a.split('.', 1)[0]
        ds_b = node_b.split('.', 1)[0]
        
        found_cluster_a = None
        found_cluster_b = None
        
        # Locate if nodes already belong to existing rows
        for cluster in clusters:
            if node_a in cluster:
                found_cluster_a = cluster
            if node_b in cluster:
                found_cluster_b = cluster
                
        # Case 1: Both nodes are already in different clusters -> Try to merge them
        if found_cluster_a and found_cluster_b:
            if found_cluster_a != found_cluster_b:
                # Check for dataset collisions before merging the two rows
                datasets_in_a = {item.split('.', 1)[0] for item in found_cluster_a}
                datasets_in_b = {item.split('.', 1)[0] for item in found_cluster_b}
                
                # Merge ONLY if they don't share any dataset origin
                if not datasets_in_a.intersection(datasets_in_b):
                    found_cluster_a.update(found_cluster_b)
                    clusters.remove(found_cluster_b)
                    
        # Case 2: Only node_a is tracked -> Try to add node_b
        elif found_cluster_a:
            datasets_in_cluster = {item.split('.', 1)[0] for item in found_cluster_a}
            if ds_b not in datasets_in_cluster:
                found_cluster_a.add(node_b)
                
        # Case 3: Only node_b is tracked -> Try to add node_a
        elif found_cluster_b:
            datasets_in_cluster = {item.split('.', 1)[0] for item in found_cluster_b}
            if ds_a not in datasets_in_cluster:
                found_cluster_b.add(node_a)
                
        # Case 4: Neither node is tracked -> Create a new row
        else:
            clusters.append({node_a, node_b})
            
    # Pivot rows into the final structured DataFrame
    rows = []
    for cluster in clusters:
        row_dict = {name: None for name in dataset_names}
        for item in cluster:
            ds_name, col_name = item.split('.', 1)
            row_dict[ds_name] = col_name
        rows.append(row_dict)
        
    return pd.DataFrame(rows)


#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def schema_alignment(data, data_names, output):

    # Compute raw matching scores matrix
    raw_results = profile_comparison(data, data_names)

    # Generate global schema alignment matrix
    global_schema = extract_global_schema_clusters(
        raw_results, data_names, threshold=0.75
    )

    # Save schema
    global_schema.to_csv(output, index=False)


    # find rows with null values in schema 
    rows_with_nulls = global_schema[global_schema.isnull().any(axis=1)]

    # fin attributes involved in problematic rows
    bad_attributes = rows_with_nulls.apply(lambda row: row.dropna().tolist(), axis=1).sum()
    bad_attributes = set(bad_attributes)

    # all attributes from schema
    all_attributes = global_schema.stack().dropna().unique().tolist()

    # keep only good attributes 
    columns_to_keep = [c for c in all_attributes if c not in bad_attributes]

    return columns_to_keep