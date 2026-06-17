import pandas as pd
from itertools import combinations
from rapidfuzz.fuzz import ratio

def generate_candidate_pairs(canopies):
    pairs=[]
    for records in canopies.values():
        for a,b in combinations(records,2):
            pairs.append((a,b))
    return pairs

def string_similarity(a,b):
    if pd.isna(a) or pd.isna(b):
        return 0
    return ratio(str(a).lower(),str(b).lower())/100

def year_similarity(a,b):
    if pd.isna(a) or pd.isna(b):
        return 0

    diff=abs(int(a)-int(b))

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

    return len(cast_a&cast_b)/len(cast_a|cast_b)

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

def match_records(df,pairs,threshold=0.8):

    matches=[]

    records=df.set_index("ID").to_dict("index")

    for a,b in pairs:

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

    return pd.DataFrame(matches)