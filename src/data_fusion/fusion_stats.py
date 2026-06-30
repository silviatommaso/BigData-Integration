import pandas as pd

df = pd.read_csv(r"data_fusion_csv\fused_entities.csv")

source_cols = ["A_IDs", "B_IDs", "C_IDs", "D_IDs"]

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