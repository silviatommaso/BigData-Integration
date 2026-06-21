import pandas as pd
from pathlib import Path
import re

# =========================
# CONFIG
# =========================
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "normalized_csv"
OUTPUT_DIR.mkdir(exist_ok=True)


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
    "duration": "Duration"
}

FINAL_COLUMNS = {
    "ID": "string",
    "Title": "string",
    "Year": "Int64",
    "Director": "string",
    "Cast": "string",
    "Genre": "string",
    "Duration": "Int64"
}

def clean_id(record_id):
    return re.sub(r'^[A-Za-z]+-?', '', str(record_id))

# =========================
# FUNCTIONS
# =========================



def clean_year(x):
    """Normalize year values to integers, extracting from dates if needed."""
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

def clean_duration(x):
    """Normalize duration values to total minutes."""
    if pd.isna(x):
        return None
    x = str(x).lower().strip()
    # 1. already numeric, e.g. 88.0
    num = pd.to_numeric(x, errors="coerce")
    if not pd.isna(num):
        return int(num)
    hours = 0
    minutes = 0
    # 2. hours
    h = re.search(r"(\d+)\s*(hr|h|hour|hours)", x)
    if h:
        hours = int(h.group(1))
    # 3. minutes
    m = re.search(r"(\d+)\s*(min|mins|m)", x)
    if m:
        minutes = int(m.group(1))
    # 4. fallback: just a number inside the string
    digits = re.findall(r"\d+", x)
    if digits and hours == 0 and minutes == 0:
        return int(digits[0])
    return hours * 60 + minutes

def fix_mojibake(x):
    """Fix double-encoded UTF-8 strings (mojibake) caused by encoding/decoding issues."""
    if not isinstance(x, str):
        return x
    try:
        # undo lowercase applied to the mojibake'd "Ã" character
        restored = x.replace("ã", "Ã")
        return restored.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return x

def fix_separators(x):
    """Replace '^' multi-value separator with a comma."""
    if isinstance(x, str):
        return x.replace("^", ", ")
    return x

def lowercase_text(x):
    """Lowercase text values for consistent matching across datasets."""
    if isinstance(x, str):
        return x.lower()
    return x


##################################################################################################################################################################################################################


def standardize_and_index(file_path, letter, col="id"):

    # read file
    file_path = Path(file_path)
    df = pd.read_csv(file_path,encoding="utf-8")

    # attributes names normalizer
    df.columns = [col.strip().lower() for col in df.columns]

    # reindexing
    df[col] = (
        df[col]
        .apply(clean_id)
        .apply(lambda x: f"{letter}{x}")
    )

    output_file = OUTPUT_DIR / f"{file_path.parent.name}_{file_path.name}"
    df.to_csv(output_file, index=False)

    return df


#-----------------------------------------------------------------------------------------------------------------------------------------------------------


def normalizer(df):    

    # fix encoding issues
    df = df.map(fix_mojibake)

    # attributes lowercase
    df.columns = [col.strip().lower() for col in df.columns]

    # rename columns
    df = df.rename(columns=COLUMN_MAPPING)

    # year cleanup
    if "year" in df.columns:
        df["year"] = df["year"].apply(clean_year).astype("Int64")

    # duration cleanup
    if "duration" in df.columns:
        df["duration"] = df["duration"].apply(clean_duration).astype("Int64")

    # keep only final schema
    df = df[[c for c in FINAL_COLUMNS.keys() if c in df.columns]]

    # fix separators and normalize casing for text columns
    for col in ["Title", "Director", "Cast", "Genre"]:
        if col in df.columns:
            df[col] = df[col].apply(fix_separators)
            df[col] = df[col].apply(lowercase_text)


    for col in df.columns:
        df[col] = df[col].apply(fix_separators)
        df[col] = df[col].apply(lowercase_text)


    for col, dtype in FINAL_COLUMNS.items():
        if col in df.columns:

            if dtype in ["int", "Int64"]:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

            elif dtype in ["float", "Float64"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            elif dtype == "datetime":
                df[col] = pd.to_datetime(df[col], errors="coerce")

            else:
                df[col] = df[col].astype(str)


    return df