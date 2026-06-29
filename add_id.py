import pandas as pd
import hashlib


INPUT = "record_linkage/llm/llm_requests.csv"
OUTPUT = "record_linkage/llm/llm_requests_with_id.csv"


def generate_request_id(id1, id2, score):

    key = f"{min(id1,id2)}_{max(id1,id2)}_{round(float(score),6)}"

    return hashlib.sha256(
        key.encode()
    ).hexdigest()[:16]


df = pd.read_csv(INPUT)


df["request_id"] = df.apply(
    lambda row: generate_request_id(
        row["id1"],
        row["id2"],
        row["classic_score"]
    ),
    axis=1
)


# sposta request_id come prima colonna
df = df[
    ["request_id"] + 
    [col for col in df.columns if col != "request_id"]
]


df.to_csv(
    OUTPUT,
    index=False
)


print("Request id aggiunti:", len(df))
print("Salvato:", OUTPUT)