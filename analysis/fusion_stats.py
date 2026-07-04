import pandas as pd
from pathlib import Path

MODE = "llm"
MODEL = "llama-3.3-70b-versatile"


model_dir = MODEL.replace("/", "_")

BASE_DIR = Path(__file__).resolve().parents[1]

if MODE == "classic":
    file_path = BASE_DIR / "results" / "data_fusion" / MODE / "fused_entities.csv"
else:
    file_path = BASE_DIR / "results" / "data_fusion" / MODE / model_dir / "fused_entities.csv"


df = pd.read_csv(file_path)

source_cols = ["A_Ids", "B_Ids", "C_Ids", "D_Ids"]

patterns = []

for _, row in df.iterrows():

    pattern = []

    for col in source_cols:

        if pd.notna(row[col]) and str(row[col]).strip() != "":
            pattern.append(col[0])  # A, B, C, D
        else:
            pattern.append("_")

    patterns.append(",".join(pattern))

result = (
    pd.Series(patterns)
    .value_counts()
    .sort_values(ascending=False)
)

print(result)