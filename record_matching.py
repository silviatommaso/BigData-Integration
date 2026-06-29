import pandas as pd
from itertools import combinations
from rapidfuzz.fuzz import ratio



#----------------------------------------------------------------------------------------------------------------------------------------------------------------

def generate_candidate_pairs(canopies):

    pairs = set()
    for records in canopies.values():

        for a, b in combinations(records, 2):

            pairs.add(
                tuple(sorted((a, b)))
            )
    return list(pairs)

#----------------------------------------------------------------------------------------------------------------------------------------------------------------

def text_similarity(a,b):
    if pd.isna(a) or pd.isna(b):
        return 0
    return round(ratio(str(a).lower(),str(b).lower())/100, 3)


def year_similarity(a,b):
    if pd.isna(a) or pd.isna(b):
        return 0

    diff = abs(int(float(a)) - int(float(b)))

    if diff==0:
        return 1

    if diff==1:
        return 0.25

    return 0


def jaccard_similarity(a,b):
    if pd.isna(a) or pd.isna(b):
        return 0

    set_a=set(x.strip().lower() for x in str(a).split(","))
    set_b=set(x.strip().lower() for x in str(b).split(","))

    if not (set_a | set_b):
        return 0

    return round(len(set_a & set_b)/len(set_a | set_b), 3)

def hybrid_similarity(a, b):

    if pd.isna(a) or pd.isna(b):
        return 0


    list_a = [
        x.strip().lower()
        for x in str(a).split(",")
        if x.strip()
    ]

    list_b = [
        x.strip().lower()
        for x in str(b).split(",")
        if x.strip()
    ]


    # caso 1: entrambi singoli
    if len(list_a) == 1 and len(list_b) == 1:

        return text_similarity(
            list_a[0],
            list_b[0]
        )


    # caso 2: entrambi multipli
    if len(list_a) > 1 and len(list_b) > 1:

        return jaccard_similarity(a,b)


    # caso 3: uno singolo e uno multiplo
    # confronto il singolo contro tutti gli elementi dell'altra lista

    if len(list_a) == 1:

        single = list_a[0]
        multiple = list_b

    else:

        single = list_b[0]
        multiple = list_a


    scores = [
        text_similarity(single, x)
        for x in multiple
    ]


    return max(scores)

SIMILARITY_FUNCTIONS = {
    "text": text_similarity,
    "year": year_similarity,
    "jaccard": jaccard_similarity,
    "hybrid": hybrid_similarity
}

def record_similarity(r1, r2, attributes):

    score = 0
    similarities = {}

    for column, config in attributes.items():

        similarity_function = SIMILARITY_FUNCTIONS[config["similarity"]]
        
        if column not in r1 or column not in r2:
            sim = 0
        else:
            sim = similarity_function(
                r1[column],
                r2[column]
            )

        similarities[column] = sim
        score += config["weight"] * sim

    return round(score,3), similarities

########################################################################################################################################################################################################################


def match_records(canopy_df, matched_path, attributes, canopy_id_position = 1, threshold=0.75, save = True):

    id_column = canopy_df.columns[canopy_id_position]
    assert canopy_df[id_column].notna().all(), "canopy_df contains rows with missing ID"

    # generation of a dictionary {cluster_id : record_id} from canopy_cluster's blocks
    canopies = {}
    for cluster_id, group in canopy_df.groupby("Cluster_ID"):
        canopies[cluster_id] = list(group[id_column])


    candidate_pairs = generate_candidate_pairs(canopies)
    print("Candidate pairs:", len(candidate_pairs))


    dupes = canopy_df[canopy_df[id_column].duplicated()]
    print("Duplicate IDs:", len(dupes))

    records = (
        canopy_df
        .drop_duplicates(subset=id_column)
        .set_index(id_column)
        .to_dict("index")
    )

    # record matching generation
    matches=[]

    for a,b in candidate_pairs:

        r1=records[a]
        r2=records[b]

        score, similarities = record_similarity(r1,r2, attributes)

        if score>=threshold:

            match = ({    
                "id1":a,
                "id2":b,
                "score":score,
            })
            for column, sim in similarities.items():
                match[f"{column.lower()}_similarity"] = sim

            matches.append(match)

    similarity_columns = [
        f"{column.lower()}_similarity"
        for column in attributes
    ]
    # matches with score
    if matches:
        matches = pd.DataFrame(matches)
    else:
        matches = pd.DataFrame(columns=[
            "id1",
            "id2",
            "score",
            *similarity_columns
        ])

    print("Total matches found:", len(matches))

    before = len(matches)

    matches = matches.drop_duplicates(
        subset=["id1", "id2"]
    ).sort_values(
        by="score",
        ascending=False
    ).reset_index(drop=True)

    print("Duplicate matches removed:", before - len(matches))


    if save:
        matches.to_csv(matched_path, index=False)
    else:
        print("Run without saving")
    
    return matches