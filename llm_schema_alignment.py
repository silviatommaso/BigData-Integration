from groq import Groq
from dotenv import load_dotenv
import os
import time
import pandas as pd
from pathlib import Path
import json

import utils


# -----------------------
# CONFIGURATION
# -----------------------

load_dotenv()

client = Groq(
    api_key=os.getenv("API_KEY")
)

PAUSE_BETWEEN_MODELS = 10

LLMS = [
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "openai/gpt-oss-120b",
    "groq/compound-mini"
]



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



# -----------------------
# PROMPT CONSTRUCTION
# -----------------------

def build_messages(samples, dataset_names):

    datasets_text = []

    for name, sample in zip(dataset_names, samples):

        sample_csv = sample.to_csv(index=False)

        datasets_text.append(
            f"""
            {name}:
            {sample_csv}
            """
        )

    datasets_text = "\n".join(datasets_text)


    output_header = ",".join(dataset_names)
    print(output_header)
    example_row_1 = ",".join([f"attribute_{i+1}" for i in range(len(dataset_names))])
    print(example_row_1)
    example_row_2 = ",".join(["" if i % 2 else f"attribute_{i+10}" for i in range(len(dataset_names))])
    print(example_row_2)

    return [

        {
            "role": "system",
            "content": f"""
            Some movie platforms are merging into a single company. To support this merger, they need to unify their movie databases. Each platform has provided one or more datasets.

            In this data integration pipeline, you are a data engineer responsible for schema alignment: identifying which attributes (columns) in each dataset correspond to each other across the datasets.

            Below are dataset names and sample excerpts from each dataset, extracted from CSV files. The first row represents the schema (column names).

            Rules:
            - An attribute does not necessarily have a match in every dataset.
            - If an attribute has no corresponding attribute in another dataset, leave the corresponding cell empty.
            - Each row of the output represents a single semantic attribute shared across datasets.
            - Columns of the output CSV correspond exactly to the datasets in the order they are provided.
            - An attribute may appear at most once in each output column.
            - Return only the CSV content and nothing else.
            - Do not provide explanations, markdown, or code fences.

            Output format example:

            {output_header}
            {example_row_1}
            {example_row_2}
            """
        },

        {
            "role": "user",
            "content": f"""
                        Datasets:

                        {datasets_text}
                        """
        }
    ]

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# -----------------------
# LLM CALL
# -----------------------

def call_llm(messages, model_name):

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0
    )

    usage = getattr(response, "usage", None)

    tokens = {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens
    } if usage else None

    content = response.choices[0].message.content.strip()

    return content, tokens


# -----------------------
# RUN ALL MODELS
# -----------------------

def result_definer(messages):

    results = []

    outputs = {}
    latencies = {}
    token_usage = {}

    max_retries = 3

    for model in LLMS:

        for attempt in range(max_retries):

            try:
                start = time.time()

                outputs[model], token_usage[model] = call_llm(messages, model)

                latencies[model] = time.time() - start

                time.sleep(PAUSE_BETWEEN_MODELS)

                break

            except Exception as e:

                wait = 2 ** attempt
                print(f"Error with {model}: {e} | retry in {wait}s")

                time.sleep(wait)

                if attempt == max_retries - 1:
                    outputs[model] = None
                    token_usage[model] = None
                    latencies[model] = None

    for model in LLMS:

        results.append({
            "model": model,
            "prediction": outputs.get(model),
            "latency": latencies.get(model),
            "tokens": token_usage.get(model)
        })

    return results

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# -----------------------
# SAMPLE SELECTION
# -----------------------

def get_sample(df, n=5, min_differences=5, seed=None):

    df_clean = df.dropna()

    if len(df_clean) < n:
        df_clean = df

    candidates = df_clean.sample(frac=1, random_state=seed)


    selected = []

    for _, row in candidates.iterrows():

        valid = True

        for existing in selected:

            num_differences = (row != existing).sum()

            if num_differences < min_differences:
                valid = False
                break

        if valid:
            selected.append(row)

        if len(selected) == n:
            break

    if len(selected) < n:

        remaining = candidates.loc[
            ~candidates.index.isin(
                [r.name for r in selected]
            )
        ]

        needed = n - len(selected)

        selected.extend([row for _, row in remaining.head(needed).iterrows()])

    return pd.DataFrame(selected).reset_index(drop=True)


################################################################################################################################################################################################################################################################################

# -----------------------
# MAIN FUNCTION
# -----------------------

def prompt_aligning(dfs, dataset_names, output):

    samples = []

    for df in dfs:
        samples.append(
            get_sample(df)
        )

    messages = build_messages(
        samples,
        dataset_names
    )

    results = result_definer(messages)

    with open(output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    
    return results







