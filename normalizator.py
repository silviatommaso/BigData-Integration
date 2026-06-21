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

FINAL_COLUMNS = ["ID", "Title", "Year", "Director", "Cast", "Genre", "Duration"]


# =========================
# CLEANING FUNCTIONS
# =========================

def clean_id(record_id):
    return re.sub(r'^[A-Za-z]+-?', '', str(record_id))


def clean_year(x):
    if pd.isna(x):
        return None

    x = str(x)

    # numeric
    num = pd.to_numeric(x, errors="coerce")
    if not pd.isna(num):
        return int(num)

    # datetime fallback
    dt = pd.to_datetime(x, errors="coerce")
    if not pd.isna(dt):
        return int(dt.year)

    return None


def clean_duration(x):
    if pd.isna(x):
        return None

    x = str(x).lower().strip()

    num = pd.to_numeric(x, errors="coerce")
    if not pd.isna(num):
        return int(num)

    hours = 0
    minutes = 0

    h = re.search(r"(\d+)\s*(hr|h|hour|hours)", x)
    if h:
        hours = int(h.group(1))

    m = re.search(r"(\d+)\s*(min|mins|m)", x)
    if m:
        minutes = int(m.group(1))

    digits = re.findall(r"\d+", x)
    if digits and hours == 0 and minutes == 0:
        return int(digits[0])

    return hours * 60 + minutes


def fix_mojibake(x):
    if not isinstance(x, str):
        return x
    try:
        restored = x.replace("ã", "Ã")
        return restored.encode("latin1").decode("utf-8")
    except:
        return x


def fix_separators(x):
    if isinstance(x, str):
        return x.replace("^", ", ")
    return x


def lowercase_text(x):
    if isinstance(x, str):
        return x.lower()
    return x


# =========================
# NORMALIZER
# =========================

def normalizer(df, letter):

    # 1. fix encoding
    df = df.map(fix_mojibake)

    # 2. normalize column names
    df.columns = [col.strip().lower() for col in df.columns]

    # 3. rename schema
    df = df.rename(columns=COLUMN_MAPPING)

    # 4. reindex ID
    if "ID" in df.columns:
        df["ID"] = df["ID"].apply(clean_id).apply(lambda x: f"{letter}{x}")

    # 5. clean year safely
    if "Year" in df.columns:
        df["Year"] = df["Year"].apply(clean_year)

    # 6. clean duration safely
    if "Duration" in df.columns:
        df["Duration"] = df["Duration"].apply(clean_duration)

    # 7. keep only final schema columns
    df = df[[c for c in FINAL_COLUMNS if c in df.columns]]

    # 8. text normalization
    text_cols = ["Title", "Director", "Cast", "Genre"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].apply(fix_separators).apply(lowercase_text)

    # 9. SAFE TYPE ENFORCEMENT (NO ASTYPE CRASH)
    for col in df.columns:
        if col in ["Year", "Duration"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        else:
            df[col] = df[col].astype("string")

    return df