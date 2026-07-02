from groq import Groq
import json
import pandas as pd
from dotenv import load_dotenv
import os
import time
import hashlib

from .record_matching import match_records

WAITING_TIME = 0.5 # time to wait between each LLM request

def generate_request_id(id1,id2,score):

    key=f"{min(id1,id2)}_{max(id1,id2)}_{round(float(score),6)}"

    return hashlib.sha256(
        key.encode()
    ).hexdigest()[:16]

def call_llm(prompt, client, model="llama-3.3-70b-versatile", temperature = 0):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role":"user",
                "content":prompt
            }
        ],
        temperature=temperature,
        response_format={"type": "json_object"}
    )

    return response.choices[0].message.content

def build_match_row(id1, id2, score, similarities, attributes, method, confidence):

    row = {
        "id1": id1,
        "id2": id2,
        "score": score
    }

    for column in attributes:

        key = f"{column.lower()}_similarity"

        row[key] = similarities[key]

    row["method"] = method
    row["confidence"] = confidence

    return row


def classify_error(msg: str):

    m = str(msg).lower()

    if (
        "per day" in m
        or "daily" in m
        or "tpd" in m
    ):
        return "daily_limit"

    if (
        "rate limit" in m
        or "tokens per minute" in m
        or "tpm" in m
    ):
        return "retry"

    return "unknown"


def get_client():

    load_dotenv()
    api_key=os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY is not set")

    return Groq(api_key=api_key)

import re

def parse_llm_json(text):

    if text is None:
        return None

    text = text.strip()

    text = text.replace("```json","")
    text = text.replace("```","")

    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL
    ).strip()

    try:
        return json.loads(text)

    except:

        match = re.search(
            r"\{.*\}",
            text,
            flags=re.DOTALL
        )

        if match:

            try:
                return json.loads(match.group())

            except:
                return None

        return None


def llm_match_record(record1, record2, attributes, client, model="llama-3.3-70b-versatile", temperature = 0):

    record_a = ""
    record_b = ""

    for column in attributes:

        record_a += f"{column}: {record1[column]}\n"
        record_b += f"{column}: {record2[column]}\n"

    prompt = f"""
        You are an entity resolution system.

        Decide if the two records refer to the same entity.

        Give a confidence score between 0 and 1.

        Rules:
        - High confidence only with strong evidence.
        - Different attributes should reduce confidence.
        - Explain briefly.

        Compare the following attributes:

        Record A:
        {record_a}

        Record B:
        {record_b}

        Return ONLY JSON:

        {{
        "match": true/false,
        "confidence": float,
        "explanation": "short reason"
        }}
    """

    try:

        output = call_llm(
            prompt,
            client,
            model,
            temperature
        )

    except Exception as e:

        action = classify_error(str(e))

        return {
            "status": action,
            "result": None,
            "error": str(e)
        }

    parsed = parse_llm_json(output)

    if parsed is None:

        print("\nRAW OUTPUT:")
        print(output)
        print("-" * 80)

        return {
            "status":"json_error",
            "result":None,
            "error":"Invalid JSON"
        }

    return {

        "status":"ok",
        "result":parsed,
        "error":None

    }



def append_csv(row,file):

    
    pd.DataFrame([row]).to_csv(
        file,
        mode="a",
        header=False,
        index=False
    )



def llm_record_matching(
    canopy_df,
    matches_output,
    llm_requests_output,
    attributes,
    canopy_id_position=1,
    llm_threshold=0.65,
    auto_threshold=0.75,
    model="llama-3.3-70b-versatile",
    temperature=0
):

    print("LLM assisted record matching started")
    id_column = canopy_df.columns[canopy_id_position]


    candidate_matches=match_records(
        canopy_df,
        None,
        attributes,
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

    print("Total candidate matches:", len(candidate_matches))

    print("Automatic matches (no LLM):", len(classic_matches))

    print("LLM candidates before dedup:", len(llm_candidates))

    print("Duplicate LLM candidates removed:", len(llm_candidates) - len(llm_candidates_no_duplicates))

    print("LLM requests to evaluate:", len(llm_candidates_no_duplicates))

    similarity_columns = [
        f"{column}_similarity"
        for column in attributes
    ]

    if not os.path.exists(matches_output):

        pd.DataFrame(columns=[
            "id1",
            "id2",
            "score",
            *similarity_columns,
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
    print("Total cache rows:",len(cache))

    print("Unique request ids:", cache["request_id"].nunique())

    print("Duplicate request ids:", len(cache)-cache["request_id"].nunique())

    cache_map={}


    for _,r in cache.iterrows():

        cache_map[r["request_id"]]=r


    # ======================
    # DIVIDO CACHE / NUOVE
    # ======================

    cached_results={}

    pending_llm=[]

    candidate_map={}


    for _,row in llm_candidates_no_duplicates.iterrows():

        request_id=generate_request_id(
            row["id1"],
            row["id2"],
            row["score"]
        )


        candidate_map[request_id]=row


        if request_id in cache_map:

            cached_results[request_id]=cache_map[request_id]

        else:

            pending_llm.append(
                (row,request_id)
            )

    print("Cached requests:", len(cached_results))

    print("New LLM calls needed:",len(pending_llm))

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


        append_csv(
            build_match_row(
                row["id1"],
                row["id2"],
                row["score"],
                row,
                attributes,
                "algorithm",
                ""
            ),
            matches_output
        )

        existing_pairs.add(pair)
        classic_added+=1

    print("Classic matches added:", classic_added)

    # ======================
    # CACHE PROCESS
    # ======================

    cache_matches = 0
    new_llm_matches = 0

    for request_id,cached in cached_results.items():


        if int(cached["match"]) == 1 and float(cached["confidence"])>=0.75:
            
            pair = (
                cached["id1"],
                cached["id2"]
            )

            if pair in existing_pairs:
                continue
            
            row = candidate_map[request_id]

            append_csv(
                build_match_row(
                    row["id1"],
                    row["id2"],
                    row["score"],
                    row,
                    attributes,
                    "LLM",
                    cached["confidence"]
                ),
                matches_output
            )

            existing_pairs.add(pair)
            cache_matches+=1

    # ======================
    # NEW LLM CALLS
    # ======================

    client = get_client()

    calls = 0
    attempts = 0
    MAX_RETRIES = 3

    for row, request_id in pending_llm:

        r1 = canopy_df[canopy_df[id_column] == row["id1"]]
        r2 = canopy_df[canopy_df[id_column] == row["id2"]]

        if r1.empty or r2.empty:
            print(f"Warning: ID {row['id1']} or {row['id2']} not found, skipping")
            continue

        record1 = r1.iloc[0]
        record2 = r2.iloc[0]

        attempts += 1
        result = None

        for retry in range(MAX_RETRIES):

            response = llm_match_record(
                record1,
                record2,
                attributes,
                client,
                model,
                temperature
            )

            status = response["status"]

            if status == "ok":
                result = response["result"]
                break

            elif status == "retry":
                wait = min(60, 2 ** retry)
                print(f"\nRate limit. Retry in {wait}s")
                time.sleep(wait)
                continue

            elif status == "daily_limit":
                print("\nDaily limit reached.")
                return pd.read_csv(matches_output)

            elif status == "json_error":
                print(f"\nInvalid JSON (retry {retry+1}/{MAX_RETRIES})")

                if retry < MAX_RETRIES - 1:
                    continue

                break

            else:
                print("\nLLM Error:", response["error"])
                break

        if result is None:

            print(f"\nWarning: failed after {MAX_RETRIES} retries, skipping {row['id1']}-{row['id2']}")
            print(f"\rLLM progress {attempts}/{len(pending_llm)}", end="")
            continue

        calls += 1

        append_csv({
            "request_id": request_id,
            "id1": row["id1"],
            "id2": row["id2"],
            "classic_score": row["score"],
            "match": int(result["match"]),
            "confidence": result["confidence"],
            "explanation": result["explanation"]
        }, llm_requests_output)

        if result["match"] and result["confidence"] >= 0.75:

            pair = (row["id1"], row["id2"])

            if pair not in existing_pairs:

                append_csv(
                    build_match_row(
                        row["id1"],
                        row["id2"],
                        row["score"],
                        row,
                        attributes,
                        "LLM",
                        result["confidence"]
                    ),
                    matches_output
                )

                existing_pairs.add(pair)
                new_llm_matches += 1

        print(f"\rLLM progress {attempts}/{len(pending_llm)}", end="")

        time.sleep(WAITING_TIME)


    final_matches = pd.read_csv(matches_output)


    total_llm_matches = len(
        final_matches[
            final_matches["method"] == "LLM"
        ]
    )

    total_algorithm_matches = len(
        final_matches[
            final_matches["method"] == "algorithm"
        ]
    )

    final_matches = final_matches.sort_values(
        by="score",
        ascending=False
    ).reset_index(drop=True)

    final_matches.to_csv(
        matches_output,
        index=False
    )

    print("\nCache hits:", len(cached_results))
    print("New LLM calls:", calls)
    print("Cache matches restored:", cache_matches)
    print("New LLM matches:", new_llm_matches)

    print(
        "Final matches:",
        len(final_matches),
        "| Algorithm:",
        total_algorithm_matches,
        "| LLM:",
        total_llm_matches
    )

    return final_matches