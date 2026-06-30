import pandas as pd
import os

def load_valid_ids(merged_file, prefix):
    
    df = pd.read_csv(merged_file)
    ids = df["Id"].astype(str)

    return set(ids[ids.str.startswith(prefix)].str.extract(r"(\d+)", expand=False).dropna())

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

def evaluate(matches_file, ground_truth_file, left_prefix, right_prefix, mode, name):
    matches = pd.read_csv(matches_file)
    truth = pd.read_csv(ground_truth_file, comment="#")

    truth.columns = truth.columns.str.strip().str.lower()
    
    merged_file = f"schema_alignment\{mode}\merged_movies.csv"

    valid_left_ids = load_valid_ids(merged_file, left_prefix)

    valid_right_ids = load_valid_ids(merged_file, right_prefix)

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
    false_positive_pairs = []

    for pair in predicted_pairs:
        if pair in truth_pairs:
            if truth_pairs[pair] == 1:
                TP += 1
            else:
                FP += 1
                false_positive_pairs.append(pair)

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
    print(
        "Predicted matches between",
        left_prefix,
        "and",
        right_prefix + ":",
        len(predicted_pairs)
    )

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

    if false_positive_pairs:

        print()
        print("FALSE POSITIVE RECORDS")
        print("---------------------")

        merged_df = pd.read_csv(merged_file)
        merged_df = merged_df.convert_dtypes()


        for left_id, right_id in false_positive_pairs:

            left_full = left_prefix + left_id
            right_full = right_prefix + right_id


            print()
            print("PAIR:", left_full, "-", right_full)


            records = merged_df[
                merged_df["Id"]
                .astype(str)
                .isin(
                    [
                        left_full,
                        right_full
                    ]
                )
            ]


            print(records.to_string(index=False))


matches_file = r"record_linkage\classic\matches.csv"
llm_matches_file = r"record_linkage\llm\matches.csv"

evaluate(
    matches_file,
    "ground_truth/movies_3_labeled_data.csv",
    "a",
    "b",
    "classic",
    "Movies 3"
)

evaluate(
    matches_file,
    "ground_truth/movies_5_labeled_data.csv",
    "d",
    "c",
    "classic",
    "Movies 5"
)

print("-------------------")
print("LLM RESULTS")

evaluate(
    llm_matches_file,
    "ground_truth/movies_3_labeled_data.csv",
    "a",
    "b",
    "llm",
    "Movies 3"
)

evaluate(
    llm_matches_file,
    "ground_truth/movies_5_labeled_data.csv",
    "d",
    "c",
    "llm",
    "Movies 5"
)