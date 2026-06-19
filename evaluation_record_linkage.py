import pandas as pd

def evaluate(
    matches_file,
    ground_truth_file,
    left_prefix,
    right_prefix,
    name,
    debug=False
):

    matches = pd.read_csv(matches_file)

    truth = pd.read_csv(
        ground_truth_file,
        comment="#"
    )

    truth.columns = (
        truth.columns
        .str.strip()
        .str.lower()
    )

    # =========================
    # FILTER DATASET PAIR
    # =========================

    matches = matches[
        matches["id1"].str.startswith(left_prefix)
        &
        matches["id2"].str.startswith(right_prefix)
    ]


    if debug:
        print("\nDEBUG",name)
        print("Matches after filtering:",len(matches))

    # =========================
    # REMOVE PREFIX
    # =========================

    matches["id1"] = (
        matches["id1"]
        .astype(str)
        .str.replace(left_prefix,"",regex=False)
    )

    matches["id2"] = (
        matches["id2"]
        .astype(str)
        .str.replace(right_prefix,"",regex=False)
    )


    predicted_pairs=set(
        tuple(x)
        for x in matches[
            [
                "id1",
                "id2"
            ]
        ].values
    )

    # =========================
    # NORMALIZE GROUND TRUTH
    # =========================

    truth["ltable.id"] = (
        truth["ltable.id"]
        .astype(str)
        .str.extract(r"(\d+)")
    )

    truth["rtable.id"] = (
        truth["rtable.id"]
        .astype(str)
        .str.extract(r"(\d+)")
    )

    truth_pairs={}

    for _,row in truth.iterrows():

        if pd.isna(row["gold"]):
            continue


        pair=(
            str(row["ltable.id"]),
            str(row["rtable.id"])
        )

        truth_pairs[pair]=int(row["gold"])

    # =========================
    # DEBUG MATCHES INSIDE GT
    # =========================

    matched_gold_1=0
    matched_gold_0=0

    for pair in predicted_pairs:

        if pair in truth_pairs:

            if truth_pairs[pair]==1:
                matched_gold_1+=1

            else:
                matched_gold_0+=1

    if debug:

        print("Predicted matches with gold=1:",matched_gold_1)
        print("Predicted matches with gold=0:",matched_gold_0)

        print("\nFirst predicted pairs:")
        print(list(predicted_pairs)[:10])

        print("\nFirst ground truth pairs:")
        print(list(truth_pairs.items())[:10])

    # =========================
    # METRICS
    # =========================

    TP=0
    FP=0
    FN=0

    for pair in predicted_pairs:

        if pair in truth_pairs:

            if truth_pairs[pair]==1:
                TP+=1

            else:
                FP+=1

    for pair,gold in truth_pairs.items():

        if gold==1 and pair not in predicted_pairs:
            FN+=1

    precision=(
        TP/(TP+FP)
        if TP+FP>0
        else 0
    )

    recall=(
        TP/(TP+FN)
        if TP+FN>0
        else 0
    )

    f1=(
        2*precision*recall/(precision+recall)
        if precision+recall>0
        else 0
    )

    print()
    print(name)
    print("----------------")

    print("Predicted matches (entire dataset):",len(predicted_pairs))
    print(
        "True matches in ground truth:",
        sum(1 for x in truth_pairs.values() if x==1)
    )
    print("Predicted pairs inside labeled set:",matched_gold_1+matched_gold_0)
    print("TP:",TP)
    print("FP:",FP)
    print("FN:",FN)

    print("Precision:",precision)
    print("Recall:",recall)
    print("F1:",f1)

matches_file="record_linkage_csv/matches.csv"

evaluate(
    matches_file,
    "ground_truth/movies_3_labeled_data.csv",
    "a",
    "b",
    "Movies 3",
    debug=False
)


evaluate(
    matches_file,
    "ground_truth/movies_5_labeled_data.csv",
    "c",
    "d",
    "Movies 5",
    debug=False
)