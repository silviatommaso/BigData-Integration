from normalizator import normalizer
from pathlib import Path
import pandas as pd


SCHEMA_ALIGN = True


INPUT_DIR = Path("normalized_csv")

files = [
    INPUT_DIR / "movies3_cleaned_imdb_cleaned.csv",
    INPUT_DIR / "movies3_cleaned_rotten_tomatoes_cleaned.csv",
    INPUT_DIR / "movies5_cleaned_imdb_cleaned.csv",
    INPUT_DIR / "movies5_cleaned_roger_ebert_final.csv",
]

DTYPES = {
    "Year": "Int64",
    "Duration": "Int64"
}


############################
# STEP I -> SCHEMA ALIGNMENT
############################

#---------------------------------------------------------
# ID | Title | Year | Director | Cast | Genre | Duration |
#---------------------------------------------------------

if SCHEMA_ALIGN:

    #-------imdb movie3--------
    normalizer("dataset_cleaned/movies3_cleaned/imdb_cleaned.csv", "a")
    #-----rotten_tomatoes------
    normalizer("dataset_cleaned/movies3_cleaned/rotten_tomatoes_cleaned.csv", "b")
    #-------roger_ebert--------
    normalizer("dataset_cleaned/movies5_cleaned/roger_ebert_final.csv", "c")
    #-------imdb movie5--------
    normalizer("dataset_cleaned/movies5_cleaned/imdb_cleaned.csv", "d")


    # csv datasets unification
    dfs = [pd.read_csv(f) for f in files]

    dfs["Year"] = dfs["Year"].astype("Int64")
    dfs["Duration"] = dfs["Duration"].astype("Int64")

    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df.to_csv(INPUT_DIR / ".." / "merged_movies.csv", index=False)
    print(f"Totale righe: {len(merged_df)}")



###########################
# STEP II -> RECORD LINKAGE
###########################

#------------------------------
# Blocking by Canopy clustering
#------------------------------