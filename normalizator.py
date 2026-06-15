import pandas as pd
from pathlib import Path


# =========================
# CONFIG
# =========================

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "normalized_csv"
OUTPUT_DIR.mkdir(exist_ok=True)

files = [
    BASE_DIR / "dataset_cleaned/movies3_cleaned/imdb_cleaned.csv",
    BASE_DIR / "dataset_cleaned/movies3_cleaned/rotten_tomatoes_cleaned.csv",
    BASE_DIR / "dataset_cleaned/movies5_cleaned/imdb_cleaned.csv",
    BASE_DIR / "dataset_cleaned/movies5_cleaned/roger_ebert_final.csv",
]

COLUMN_MAPPING = {
    "id": "ID",
    "movie_name": "Title",
    "title": "Title",
    "name": "Title",
    "year": "Year",
    "release_year": "Year",
    "director": "Director",
    "directors": "Director",
    "actors": "Cast",
    "cast": "Cast",
    "genre": "Genre",
    "duration": "Duration",
}

FINAL_COLUMNS = [
    "ID",
    "Title",
    "Year",
    "Director",
    "Cast",
    "Genre",
    "Duration"
]


# =========================
# FUNCTIONS
# =========================

def indicization(df):
    df = df.copy()
    df["ID"] = range(1, len(df) + 1)
    return df


def normalize_lowercase(df):
    df = df.copy()
    df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)
    return df


def clean_year(x):
    if pd.isna(x):
        return None

    x = str(x)

    num = pd.to_numeric(x, errors="coerce")
    if not pd.isna(num):
        return int(num)

    dt = pd.to_datetime(x, errors="coerce")
    if not pd.isna(dt):
        return int(dt.year)

    return None


# =========================
# PIPELINE
# =========================

def normalizer(file_path):

    file_path = Path(file_path)
    
    df = pd.read_csv(file_path)

    # rename columns
    df = df.rename(columns=COLUMN_MAPPING)

    # year cleanup
    if "Year" in df.columns:
        df["Year"] = df["Year"].apply(clean_year).astype("Int64")

    # keep only final schema
    df = df[[c for c in FINAL_COLUMNS if c in df.columns]]

    # preprocessing pipeline
    df = normalize_lowercase(df)
    df = indicization(df)

    # output path
    output_file = OUTPUT_DIR / f"{file_path.parent.name}_{file_path.name}"

    df.to_csv(output_file, index=False)

    print(f"Salvato: {output_file}")