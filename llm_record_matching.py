from groq import Groq
import json
import pandas as pd
from dotenv import load_dotenv
import os
import time
import hashlib

from record_matching import match_records


def generate_request_id(id1,id2,score):

    key=f"{min(id1,id2)}_{max(id1,id2)}_{round(float(score),6)}"

    return hashlib.sha256(
        key.encode()
    ).hexdigest()[:16]


def get_client():

    load_dotenv()

    api_key=os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY is not set")

    return Groq(api_key=api_key)



def llm_match_record(record1,record2,client,model="llama-3.3-70b-versatile"):

    prompt=f"""
You are an entity resolution system.

Decide if the two movie records refer to the same movie.

Give a confidence score between 0 and 1.

Record A:
Title: {record1['Title']}
Year: {record1['Year']}
Director: {record1['Director']}
Cast: {record1['Cast']}

Record B:
Title: {record2['Title']}
Year: {record2['Year']}
Director: {record2['Director']}
Cast: {record2['Cast']}

Return ONLY JSON:

{{
"match": true/false,
"confidence": float,
"explanation": "short reason"
}}
"""

    try:

        response=client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role":"user",
                    "content":prompt
                }
            ],
            temperature=0
        )

        content=response.choices[0].message.content.strip()

        content=content.replace("```json","").replace("```","").strip()

        return json.loads(content)


    except Exception as e:

        if "429" in str(e):
            raise RuntimeError("RATE_LIMIT")

        print("LLM error:",e)

        return None



def append_csv(row,file):

    pd.DataFrame([row]).to_csv(
        file,
        mode="a",
        header=False,
        index=False
    )



def llm_record_matching(
    merged_df,
    canopy_df,
    matches_output,
    llm_requests_output,
    llm_threshold=0.65,
    auto_threshold=0.75
):

    print("LLM assisted record matching started")


    candidate_matches=match_records(
        merged_df,
        canopy_df,
        None,
        None,
        threshold=llm_threshold,
        save=False
    )


    print("Total candidates:",len(candidate_matches))


    classic_matches=candidate_matches[
        candidate_matches["score"]>=auto_threshold
    ]


    llm_candidates=candidate_matches[
        (candidate_matches["score"]>=llm_threshold)
        &
        (candidate_matches["score"]<auto_threshold)
    ]
    
    llm_candidates_no_duplicates = llm_candidates.drop_duplicates(
        subset=[
            "id1",
            "id2",
            "score"
        ]
    ).reset_index(drop=True)


    print("\n--- MATCH SPLIT ---")

    print(
        "Total candidate matches:",
        len(candidate_matches)
    )

    print(
        "Automatic matches (no LLM):",
        len(classic_matches)
    )

    print(
        "LLM candidates before dedup:",
        len(llm_candidates)
    )

    print(
        "Duplicate LLM candidates removed:",
        len(llm_candidates) - len(llm_candidates_no_duplicates)
    )

    print(
        "LLM requests to evaluate:",
        len(llm_candidates_no_duplicates)
    )


    if not os.path.exists(matches_output):

        pd.DataFrame(columns=[
            "id1",
            "id2",
            "score",
            "title_similarity",
            "director_similarity",
            "year_similarity",
            "cast_similarity",
            "method",
            "confidence"
        ]).to_csv(
            matches_output,
            index=False
        )


    if not os.path.exists(llm_requests_output):

        pd.DataFrame(columns=[
            "request_id",
            "id1",
            "id2",
            "classic_score",
            "match",
            "confidence",
            "explanation"
        ]).to_csv(
            llm_requests_output,
            index=False
        )


    # ======================
    # CARICO CACHE
    # ======================

    cache=pd.read_csv(llm_requests_output)
    print(
        "Total cache rows:",
        len(cache)
    )

    print(
        "Unique request ids:",
        cache["request_id"].nunique()
    )

    print(
        "Duplicate request ids:",
        len(cache)-cache["request_id"].nunique()
    )

    if "request_id" not in cache.columns:

        cache["request_id"]=cache.apply(
            lambda r: generate_request_id(
                r["id1"],
                r["id2"],
                r["classic_score"]
            ),
            axis=1
        )


        cache.to_csv(
            llm_requests_output,
            index=False
        )



    cache_map={}


    for _,r in cache.iterrows():

        cache_map[r["request_id"]]=r



    # ======================
    # DIVIDO CACHE / NUOVE
    # ======================

    cached_results={}

    pending_llm=[]


    for _,row in llm_candidates.iterrows():

        request_id=generate_request_id(
            row["id1"],
            row["id2"],
            row["score"]
        )


        if request_id in cache_map:

            cached_results[request_id]=cache_map[request_id]

        else:

            pending_llm.append(
                (row,request_id)
            )



    print(
        "Cached requests:",
        len(cached_results)
    )


    print(
        "New LLM calls needed:",
        len(pending_llm)
    )



    # ======================
    # SALVO CLASSICI
    # ======================

    existing=pd.read_csv(matches_output)


    existing_pairs=set(
        zip(
            existing["id1"],
            existing["id2"]
        )
    )


    classic_added=0


    for _,row in classic_matches.iterrows():

        pair=(row["id1"],row["id2"])


        if pair in existing_pairs:
            continue


        append_csv({

            "id1":row["id1"],
            "id2":row["id2"],
            "score":row["score"],
            "title_similarity":row["title_similarity"],
            "director_similarity":row["director_similarity"],
            "year_similarity":row["year_similarity"],
            "cast_similarity":row["cast_similarity"],
            "method":"algorithm",
            "confidence":""

        },matches_output)


        classic_added+=1



    print(
        "Classic matches added:",
        classic_added
    )



    # ======================
    # PROCESSO CACHE
    # ======================

    new_matches=0


    for request_id,cached in cached_results.items():


        if bool(cached["match"]) and float(cached["confidence"])>=0.75:

            append_csv({

                "id1":cached["id1"],
                "id2":cached["id2"],
                "score":cached["classic_score"],
                "title_similarity":"",
                "director_similarity":"",
                "year_similarity":"",
                "cast_similarity":"",
                "method":"LLM",
                "confidence":cached["confidence"]

            },matches_output)


            new_matches+=1



    # ======================
    # NUOVE CHIAMATE LLM
    # ======================

    client=get_client()

    calls=0



    for row,request_id in pending_llm:


        record1=merged_df[
            merged_df["ID"]==row["id1"]
        ].iloc[0]


        record2=merged_df[
            merged_df["ID"]==row["id2"]
        ].iloc[0]



        try:

            result=llm_match_record(
                record1,
                record2,
                client
            )


        except RuntimeError:

            print("Rate limit reached")
            break



        if result is None:
            continue



        calls+=1



        append_csv({

            "request_id":request_id,
            "id1":row["id1"],
            "id2":row["id2"],
            "classic_score":row["score"],
            "match":int(result["match"]),
            "confidence":result["confidence"],
            "explanation":result["explanation"]

        },llm_requests_output)



        if result["match"] and result["confidence"]>=0.75:


            append_csv({

                "id1":row["id1"],
                "id2":row["id2"],
                "score":row["score"],
                "title_similarity":row["title_similarity"],
                "director_similarity":row["director_similarity"],
                "year_similarity":row["year_similarity"],
                "cast_similarity":row["cast_similarity"],
                "method":"LLM",
                "confidence":result["confidence"]

            },matches_output)


            new_matches+=1



        print(
            f"\rLLM progress {calls}/{len(pending_llm)}",
            end=""
        )


        time.sleep(1.5)



    print()


    print(
        "Cache hits:",
        len(cached_results)
    )


    print(
        "New LLM calls:",
        calls
    )


    print(
        "New LLM matches:",
        new_matches
    )


    return pd.read_csv(matches_output)