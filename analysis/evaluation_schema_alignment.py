import pandas as pd
import os
from itertools import combinations


def load_schema_rows(path):
    """Load a global_schema.csv into a list of {schema_name: attribute} dicts,
    one dict per row (cluster), skipping empty cells."""
    df = pd.read_csv(path, dtype=str).fillna("")

    rows = []
    for _, raw_row in df.iterrows():
        row = {}
        for col in df.columns:
            val = str(raw_row[col]).strip().lower()
            if val:
                row[col.strip()] = val
        if row:
            rows.append(row)

    return rows


def rows_to_pairs(rows):
    """Expand each cluster row into the set of pairwise attribute links it
    implies (every pair of non-empty cells in a row is a claimed match)."""
    pairs = set()

    for row in rows:
        items = sorted(row.items())
        if len(items) < 2:
            continue
        for a, b in combinations(items, 2):
            pairs.add(frozenset([a, b]))

    return pairs


def format_pair(pair):
    (s1, a1), (s2, a2) = sorted(pair)
    return f"{s1}.{a1}  <->  {s2}.{a2}"


def evaluate(system_file, gt_file, name):
    pred_rows = load_schema_rows(system_file)
    gt_rows = load_schema_rows(gt_file)

    predicted_pairs = rows_to_pairs(pred_rows)
    truth_pairs = rows_to_pairs(gt_rows)

    TP = predicted_pairs & truth_pairs
    FP = predicted_pairs - truth_pairs
    FN = truth_pairs - predicted_pairs

    precision = len(TP) / len(predicted_pairs) if predicted_pairs else 0
    recall = len(TP) / len(truth_pairs) if truth_pairs else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    print()
    print(name)
    print("----------------")

    print("GT info")
    print("Ground truth clusters:", len(gt_rows))
    print("Ground truth pairwise links:", len(truth_pairs))

    print()
    print("global_schema.csv info")
    print("Predicted clusters:", len(pred_rows))
    print("Predicted pairwise links:", len(predicted_pairs))

    print()
    print("Results")
    print("TP:", len(TP))
    print("FP:", len(FP))
    print("FN:", len(FN))
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)

    if FP:
        print()
        print("FALSE POSITIVE LINKS")
        print("---------------------")
        for pair in sorted(FP, key=format_pair):
            print(format_pair(pair))

    if FN:
        print()
        print("FALSE NEGATIVE LINKS")
        print("---------------------")
        for pair in sorted(FN, key=format_pair):
            print(format_pair(pair))
    print()
    return {"precision": precision, "recall": recall, "f1": f1}


base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

classic = os.path.join(
    base_dir,
    "results",
    "schema_alignment",
    "classic",
    "global_schema.csv"
)

llm = os.path.join(
    base_dir,
    "results",
    "schema_alignment",
    "llm",
    "global_schema.csv"
)

gt = os.path.join(
    base_dir,
    "analysis",
    "ground_truth",
    "schema_alignment_gt.csv"
)

print("-------------------")
print("CLASSIC RESULTS")

evaluate(
    classic,
    gt,
    "Schema Alignment"
)

print("-------------------")
print("LLM RESULTS")

evaluate(
    llm,
    gt,
    "Schema Alignment"
)