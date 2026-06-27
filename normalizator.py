import pandas as pd
from pathlib import Path
import re



# =========================
# CLEANING FUNCTIONS
# =========================

# id cleaning and reindexing
def reindex_id(record_id):
    return re.sub(r'^[A-Za-z]+-?', '', str(record_id))


# clean year
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


# clean duration
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


################################################################################################################################################################################################


def final_columns(df, final_cols):
    df = df.rename(columns = final_cols)


# =========================
# NORMALIZER
# =========================

def normalizer(dfs, indexes):

    for i in range(len(dfs)):

        # fix encoding
        dfs[i] = dfs[i].map(fix_mojibake)


        # reindexing
        dfs[i].columns = [col.strip().lower() for col in dfs[i].columns]
        if "id" in dfs[i].columns:
            dfs[i]["id"] = dfs[i]["id"].apply(reindex_id).apply(lambda x: f"{indexes[i]}{x}")


        # clean year
        if "year" in dfs[i].columns:
            dfs[i]["year"] = dfs[i]["year"].apply(clean_year)
        # clean duration
        if "duration" in dfs[i].columns:
            dfs[i]["duration"] = dfs[i]["duration"].apply(clean_duration)
        
        # non numeric columns normalization
        non_numeric_cols = dfs[i].select_dtypes(exclude=["int64", "float64"]).columns.tolist()
        for col in non_numeric_cols:
            if col in dfs[i].columns:
                dfs[i][col] = dfs[i][col].apply(fix_separators).apply(lowercase_text)
                dfs[i][col] = dfs[i][col].astype("string")


    return dfs