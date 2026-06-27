import pandas as pd
from itertools import combinations
from rapidfuzz.fuzz import ratio

def get_unmatched_records(matches,df):
    if matches.empty:
        return df
    matched_ids=set(matches["id1"]) | set(matches["id2"])

    return df[
        ~df["ID"].isin(matched_ids)
    ]

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

def string_similarity(a,b):
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


def cast_similarity(a,b):
    if pd.isna(a) or pd.isna(b):
        return 0

    cast_a=set(x.strip().lower() for x in str(a).split(","))
    cast_b=set(x.strip().lower() for x in str(b).split(","))

    if len(cast_a|cast_b)==0:
        return 0

    return round(len(cast_a&cast_b)/len(cast_a|cast_b), 3)


def record_similarity(r1,r2):

    title=string_similarity(r1["Title"],r2["Title"])
    director=string_similarity(r1["Director"],r2["Director"])
    year=year_similarity(r1["Year"],r2["Year"])
    cast=cast_similarity(r1["Cast"],r2["Cast"])

    score=(
        0.50*title+
        0.25*year+
        0.15*director+
        0.10*cast
    )

    return score,title,director,year,cast

########################################################################################################################################################################################################################


def match_records(canopy_df, matched_path, singletons_path, threshold=0.8, save = True):
    assert canopy_df["ID"].notna().all(), "canopy_df contains rows with missing ID"

    # generation of a dictionary {cluster_id : record_id} from canopy_cluster's blocks
    canopies = {}
    for cluster_id, group in canopy_df.groupby("Cluster_ID"):
        canopies[cluster_id] = list(group["ID"])


    candidate_pairs = generate_candidate_pairs(canopies)
    print("Candidate pairs:", len(candidate_pairs))


    dupes = canopy_df[canopy_df["ID"].duplicated()]
    print("Duplicate IDs:", len(dupes))

    records = (
        canopy_df
        .drop_duplicates(subset="ID")
        .set_index("ID")
        .to_dict("index")
    )

    # record matching generation
    matches=[]

    for a,b in candidate_pairs:

        r1=records[a]
        r2=records[b]

        score,title,director,year,cast=record_similarity(r1,r2)

        if score>=threshold:

            matches.append({    
                "id1":a,
                "id2":b,
                "score":score,
                "title_similarity":title,
                "director_similarity":director,
                "year_similarity":year,
                "cast_similarity":cast
            })


    # matches with score
    if matches:
        matches = pd.DataFrame(matches)
    else:
        matches = pd.DataFrame(columns=[
            "id1", "id2", "score",
            "title_similarity",
            "director_similarity",
            "year_similarity",
            "cast_similarity"
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

    matches["score"] = matches["score"].round(3)
    # singletons 
    singletons = get_unmatched_records(matches, canopy_df)

    if save:
        matches.to_csv(matched_path, index=False)
        pd.DataFrame(singletons).to_csv(singletons_path, index=False)
    else:
        print("Run without saving")
    print("Records with no match:", len(singletons))

    return matches