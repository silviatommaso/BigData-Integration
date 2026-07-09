# -BigData-Integration-

This repository presents a comparative study of a traditional Big Data Integration pipeline and an LLM-assisted one.

---

## Overview

To achieve this objective, the project implements a unified Big Data Integration pipeline capable of executing both the traditional workflow and its LLM-assisted counterpart. This unified design enables a direct comparison between the two approaches while keeping the overall integration process consistent.

The pipeline follows the three core phases of a standard Big Data Integration workflow:

1. **Schema Alignment** – Map source attributes to a unified mediated schema by identifying semantic correspondences across heterogeneous datasets.
2. **Record Linkage (Entity Resolution)** – Identify records from different data sources that refer to the same real-world entity.
3. **Data Fusion (Truth Discovery)** – Reconcile conflicting attribute values and produce a single, integrated representation for each identified entity.


The experiments are conducted on three heterogeneous movie datasets derived from the Magellan project—IMDb, Rotten Tomatoes, and Roger Ebert—and are evaluated using standard execution-based metrics to assess the effectiveness of both the traditional and the LLM-assisted pipelines.

---

## Project Structure

```
BigData-Integration-AO/
├── pipeline.py                         # Main orchestrator — runs the full pipeline end-to-end
├── config.py                           
├── data/
│   └── dataset_cleaned/                # Cleaned input CSVs (per dataset) + attribute_descriptions.json
│       ├── movies3_cleaned/
│       └── movies5_cleaned/
├── src/
│   ├── preprocessing/                  # Dataset cleaning & the Roger Ebert validation scraper
│   ├── schema_alignment/               # Classic (column-profiling) and LLM schema alignment
│   ├── record_linkage/
│   │   ├── blocking/                   # Canopy clustering (TF-IDF + cosine similarity blocking)
│   │   ├── record_matching/            # Classic weighted-similarity matcher and LLM matcher
│   │   └── clustering/                 # Builds entity clusters from confirmed matches
│   ├── data_fusion/                    # Cluster-level attribute fusion + fusion statistics
│   └── utils/                          # CSV loading, normalization, shared helpers
├── analysis/
│   ├── analysis.py                     # Exploratory data analysis of the raw datasets
│   ├── evaluation_record_linkage.py    # Precision/recall/F1 vs. ground truth
│   ├── evaluation_schema_alignment.py  # Schema alignment evaluation vs. ground truth
│   ├── fusion_stats.py                 # Computes statistics on fused entity source combinations 
│   ├── record_similarity_pair.py       # Compute similarity between two records 
│   └── ground_truth/                   # Labeled candidate sets and gold-standard schema mapping
└── results/                            # All pipeline outputs (see "Outputs" below)
    ├── schema_alignment/{classic,llm}/
    ├── record_linkage/{classic,llm}/
    └── data_fusion/{classic,llm}/
```

---

### Data Sources

Cleaned CSVs live under `data/dataset_cleaned/`:

```
movies3_cleaned/imdb_cleaned.csv
movies3_cleaned/rotten_tomatoes_cleaned.csv
movies5_cleaned/imdb_cleaned.csv
movies5_cleaned/roger_ebert_cleaned.csv
```

`attribute_descriptions.json` documents each dataset's original attributes (used as context for the LLM schema-alignment prompt).

---

## Installation

1. Create a free API key from Groq (https://groq.com/).
   **Note:** The free tier is subject to rate limits and token quotas. Since the pipeline may issue a large number of LLM requests, you may exhaust the available quota before all experiments complete. If this happens, you can either wait for the quota to reset or create additional API keys using separate Groq accounts to continue the execution.

2. Clone the repo
   ```
   git clone https://github.com/silviatommaso/BigData-Integration.git
   ```
3. Create a .env file based on the .env.example and add your API keys
   ```
   API_KEY = "your_api_key_here"
   ```
---

## Usage

By default, the project runs the classical integration pipeline.

### Pipeline mode

To test alternative configurations, edit the `pipeline.py` file and set the `PIPELINE_MODE` parameter. The available options are:

- `classic`: runs the traditional pipeline only  
- `llm`: runs the LLM-assisted pipeline only  
- `both`: executes both pipelines for comparison  


### Pipeline stages configuration

It is also possible to control which stages of the pipeline are executed by enabling or disabling individual components:

```
STEPS = {
    "schema_alignment": False,

    "record_linkage": {
        "blocking": False,
        "matching": False,
        "clustering": False
    },

    "data_fusion": True
}
```

**Note:** Each stage depends on the output of the previous one. Therefore, for the first execution, it is recommended to enable all steps.

---


### LLM-assisted schema alignment

For the LLM-assisted pipeline, two schema alignment modes are available:

- **Without attribute metadata**: uses only the generic prompt for schema alignment.
- **With attribute metadata**: uses an additional manually curated JSON file (included in the repository) that describes the semantics and role of each attribute in the datasets.

This behavior is controlled by the `ATTRIBUTE_DESCRIPTION` parameter in `pipeline.py`.

- By default, this parameter points to the metadata JSON file.
- If set to `None`, the LLM-assisted alignment runs without attribute-level descriptions.


### Rate Limit Handling

If token-per-minute rate limits are reached, the pause intervals between API requests can be increased to avoid throttling errors.

This is particularly relevant when running the LLM-assisted pipeline, as multiple sequential API calls are required during schema alignment and record matching.

The delay parameters can be adjusted as follows:

- In `src/record_matching/llm_record_matching.py`, modify:
```python
WAITING_TIME
```

- In src/schema_alignment/llm_schema_alignment.py, modify:
```python
PAUSE_BETWEEN_MODELS
```

---

## Outputs

All results are written under `results/`, mirroring the `classic` / `llm` pipeline split (LLM outputs are further namespaced by model, e.g. `llm/openai_gpt-oss-120b/`):

```
results/
├── schema_alignment/{classic,llm}/global_schema.csv, merged_movies.csv
├── record_linkage/
│   ├── canopy_blocks.csv                     # shared blocking output
│   └── {classic,llm}/matches.csv, entity_clusters.csv, singletons.csv, llm_requests.csv
└── data_fusion/{classic,llm}/fused_entities.csv   # final integrated dataset
```

`fused_entities.csv` is the final deliverable: one row per real-world movie, combining attributes from all contributing sources.

---
## Using Different Datasets

The pipeline is generic and _technically_ it _could_ be adapted to any set of heterogeneous data sources that describe the same type of real-world entity. To plug in new datasets, edit the configuration at the top of `pipeline.py`:

1. **Add cleaned CSVs**: place your cleaned input files under a new folder in `data/dataset_cleaned/`(e.g. `data/dataset_cleaned/my_dataset_cleaned/`).

2. **Update the `inputs` list and `merged_file` name**: point `inputs` to your new CSV file paths instead of (or in addition to) the existing `movies3_cleaned` / `movies5_cleaned` files, and optionally rename `merged_file` to something meaningful for your dataset (e.g. `"merged_my_dataset.csv"`).

3. **Update the `SOURCES` dictionary**: define a name and reliability weight for each new source (used during data fusion), e.g.:
   ```python
      SOURCES = {
         "a": {"name": "my_dataset", "weight": 1.0},
         ...
      }
   ```

4. **Update `config.py`**: adapt the configuration parameters to the new dataset.

   - **Update `matching_attributes`** to define the attributes used for record matching, their corresponding weights, and the similarity functions applied to each attribute.

   - **Adjust record matching thresholds** if needed according to the characteristics of the new dataset:

      ```python
      # Record matching thresholds
      MATCHING_THRESHOLD = 0.75

      # LLM record matching thresholds
      LLM_VERIFICATION_THRESHOLD = 0.65
      LLM_AUTO_MATCH_THRESHOLD = 0.75
      ```
   - **Adapt `canopy_params`** to specify the blocking configuration for the new dataset, including blocking attributes, similarity thresholds, and TF-IDF parameters.

   - **Update `fusion_attributes`** to define the fusion strategy adopted for each attribute during the data fusion phase (e.g., conflict resolution rules or attribute-specific aggregation strategies).

5. **Update `attribute_descriptions.json`**: add an entry describing each attribute of your new dataset (name, semantics, expected type/role). This file is used as context for the LLM-assisted schema alignment prompt, so accurate descriptions improve alignment quality. If you don't want to update it, simply set `ATTRIBUTE_DESCRIPTIONS = None` where it's defined inside `pipeline.py` (note that accuracy will worsen without attribute-level descriptions).

6. **(Optional) Ground truth for evaluation**: if you want to run `evaluation_record_linkage.py` or `evaluation_schema_alignment.py` on the new data, add the corresponding labeled candidate sets / gold-standard mapping under `analysis/ground_truth/`. Without this, the pipeline will still run end-to-end, but the evaluation scripts won't have anything to compare against.

7. **Re-run with all steps enabled**: since each stage depends on the previous one's output, the first run on a new dataset should have all steps in `STEPS` set to `True` (schema alignment, blocking, matching, clustering, fusion).

No changes to the core pipeline logic (schema alignment, record linkage, data fusion) are required.

---

## Acknowledgements

**Raw sources**: the original (pre-cleaning) IMDB / Rotten Tomatoes / Roger Ebert movie datasets used as input to this project come from the **[Magellan Data Repository](https://sites.google.com/site/anhaidgroup/useful-stuff/the-magellan-data-repository)** (AnHai Doan's group, UW-Madison)

We thank the creators of these resources for their contributions to research and benchmarking in this field.


## Contact

Silvia Tommaso - https://www.linkedin.com/in/silvia-tommaso-9476a2398/ - silvia.tommaso18112002@gmail.com
Francesco Pittacolo - https://www.linkedin.com/in/francesco-pittacolo-6b9906352/ - francesco.pittacolo@gmail.com
