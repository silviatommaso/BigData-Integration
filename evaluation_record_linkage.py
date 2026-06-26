import pandas as pd
import os

def load_valid_ids(csv_file):
    df = pd.read_csv(csv_file)
    return set(df["ID"].astype(str).str.extract(r"(\d+)", expand=False).dropna())

def normalize_pair(id1, id2, left_prefix, right_prefix):
    id1 = str(id1)
    id2 = str(id2)

    if id1.startswith(left_prefix) and id2.startswith(right_prefix):
        return (
            id1.replace(left_prefix, ""),
            id2.replace(right_prefix, "")
        )

    if id1.startswith(right_prefix) and id2.startswith(left_prefix):
        return (
            id2.replace(left_prefix, ""),
            id1.replace(right_prefix, "")
        )

    return None

def evaluate(matches_file, ground_truth_file, left_prefix, right_prefix, left_dataset_file, right_dataset_file, name):
    matches = pd.read_csv(matches_file)
    truth = pd.read_csv(ground_truth_file, comment="#")

    truth.columns = truth.columns.str.strip().str.lower()

    valid_left_ids = load_valid_ids(left_dataset_file)
    valid_right_ids = load_valid_ids(right_dataset_file)

    normalized_matches = []

    for _, row in matches.iterrows():
        pair = normalize_pair(
            row["id1"],
            row["id2"],
            left_prefix,
            right_prefix
        )

        if pair:
            normalized_matches.append(pair)

    duplicate_count = len(normalized_matches) - len(set(normalized_matches))
    predicted_pairs = set(normalized_matches)

    truth["ltable.id"] = truth["ltable.id"].astype(str).str.extract(r"(\d+)")
    truth["rtable.id"] = truth["rtable.id"].astype(str).str.extract(r"(\d+)")

    truth_pairs = {}

    discarded_gt = 0
    discarded_gold_1 = 0
    discarded_gold_0 = 0

    for _, row in truth.iterrows():

        if pd.isna(row["gold"]):
            continue

        left_id = str(row["ltable.id"])
        right_id = str(row["rtable.id"])

        if left_id not in valid_left_ids or right_id not in valid_right_ids:
            discarded_gt += 1

            if row["gold"] == 1:
                discarded_gold_1 += 1
            else:
                discarded_gold_0 += 1

            continue

        truth_pairs[(left_id, right_id)] = int(row["gold"])

    TP = 0
    FP = 0
    FN = 0

    for pair in predicted_pairs:
        if pair in truth_pairs:
            if truth_pairs[pair] == 1:
                TP += 1
            else:
                FP += 1

    for pair, gold in truth_pairs.items():
        if gold == 1 and pair not in predicted_pairs:
            FN += 1

    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    matched_gold_1 = sum(1 for p in predicted_pairs if p in truth_pairs and truth_pairs[p] == 1)
    matched_gold_0 = sum(1 for p in predicted_pairs if p in truth_pairs and truth_pairs[p] == 0)

    print()
    print(name)
    print("----------------")

    print("GT info")
    print("Ground truth pairs before validation:", len(truth))
    print("Ground truth pairs after validation:", len(truth_pairs))
    print("Discarded GT pairs:", discarded_gt)
    print("Discarded gold=1:", discarded_gold_1)
    print("Discarded gold=0:", discarded_gold_0)

    print()
    print("matches.csv info")
    print(f"Matches after filtering ({left_prefix}-{right_prefix} candidates):", len(predicted_pairs))
    print("Duplicate pairs found:", duplicate_count)
    print("Predicted matches between", os.path.basename(left_dataset_file), "and", os.path.basename(right_dataset_file) + ":", len(predicted_pairs))

    print()
    print("Results")
    print("True matches in ground truth:", sum(1 for x in truth_pairs.values() if x == 1))
    print("Predicted pairs inside labeled set:", matched_gold_1 + matched_gold_0)
    print("TP:", TP)
    print("FP:", FP)
    print("FN:", FN)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)

matches_file = r"record_linkage\classic\matches.csv"
llm_matches_file = r"record_linkage\llm\matches.csv"

evaluate(
    matches_file,
    "ground_truth/movies_3_labeled_data.csv",
    "a",
    "b",
    r"normalized_csv\movies3_cleaned_imdb_cleaned.csv",
    r"normalized_csv\movies3_cleaned_rotten_tomatoes_cleaned.csv",
    "Movies 3"
)

evaluate(
    matches_file,
    "ground_truth/movies_5_labeled_data.csv",
    "d",
    "c",
    r"normalized_csv\movies5_cleaned_roger_ebert_cleaned.csv",
    r"normalized_csv\movies5_cleaned_imdb_cleaned.csv",
    "Movies 5"
)

print("-------------------")
print("LLM RESULTS")

evaluate(
    llm_matches_file,
    "ground_truth/movies_3_labeled_data.csv",
    "a",
    "b",
    r"normalized_csv\movies3_cleaned_imdb_cleaned.csv",
    r"normalized_csv\movies3_cleaned_rotten_tomatoes_cleaned.csv",
    "Movies 3"
)

evaluate(
    llm_matches_file,
    "ground_truth/movies_5_labeled_data.csv",
    "d",
    "c",
    r"normalized_csv\movies5_cleaned_roger_ebert_cleaned.csv",
    r"normalized_csv\movies5_cleaned_imdb_cleaned.csv",
    "Movies 5"
)