from normalizator import normalizer
from schema_alignment import schema_alignment, cols_to_keep
from llm_schema_alignment import prompt_aligning
from canopy_clustering import canopy_cluster
from record_matching import match_records
from llm_record_matching import llm_record_matching
from entity_clustering import build_clusters
from data_fusion import fuse_cluster
import utils
import csv
from pathlib import Path
import pandas as pd



# =====================================================
# CONFIGURATION
# =====================================================

PIPELINE_MODE = "llm"


if PIPELINE_MODE == "both":
    PIPELINES = ["classic", "llm"]
else:
    PIPELINES = [PIPELINE_MODE]


STEPS = {
    "schema_alignment": False,

    "record_linkage": {
        "blocking": False,
        "matching": True,
        "clustering": False
    },

    "data_fusion": False
}


INPUT_DIR = Path("dataset_cleaned")
NORMALIZED_FILE = Path("normalized_csv")

inputs = [
    INPUT_DIR / "movies3_cleaned" / "imdb_cleaned.csv",
    INPUT_DIR / "movies3_cleaned" / "rotten_tomatoes_cleaned.csv",
    INPUT_DIR / "movies5_cleaned" / "imdb_cleaned.csv",
    INPUT_DIR / "movies5_cleaned" / "roger_ebert_cleaned.csv"
]

normalized = [
    NORMALIZED_FILE / "movies3_cleaned_imdb_cleaned.csv",
    NORMALIZED_FILE / "movies3_cleaned_rotten_tomatoes_cleaned.csv",
    NORMALIZED_FILE / "movies5_cleaned_imdb_cleaned.csv",
    NORMALIZED_FILE / "movies5_cleaned_roger_ebert_cleaned.csv"
]
# =====================================================
# COMMON FILES
# =====================================================

COMMON = {

    "canopy":
        "record_linkage/canopy_blocks.csv"

}


# =====================================================
# PIPELINE OUTPUTS
# =====================================================

outputs = {

    "classic": {

        "global_schema":
            "schema_alignment/classic/global_schema.csv",

        "merged":
        "schema_alignment/classic/merged_movies.csv",

        "matches":
            "record_linkage/classic/matches.csv",

        "singletons":
            "record_linkage/classic/singletons.csv",

        "clusters":
            "record_linkage/classic/entity_clusters.csv",

        "fusion":
            "data_fusion/classic/fused_entities.csv"
    },


    "llm": {

        "alignment_stats":
            "schema_alignment/llm/schema_alignment_results_stat.json",

        "global_schema":
            "schema_alignment/llm/global_schema.csv",

        "merged":
        "schema_alignment/llm/merged_movies.csv",

        "matches":
            "record_linkage/llm/matches.csv",

        "requests":
            "record_linkage/llm/llm_requests.csv",

        "clusters":
            "record_linkage/llm/entity_clusters.csv",

        "fusion":
            "data_fusion/llm/fused_entities.csv"
    }

}

INDEXES = ["a", "b", "c", "d"]
DATASETS_NAMES = ["imdb_3", "rotten_tomatoes", "imdb_5", "roger_ebert"]



# =====================================================
# PIPELINES (CLASSIC + LLM)
# =====================================================

for pipeline in PIPELINES:


    print()
    print("==============================")
    print("Running:", pipeline)
    print("==============================")


    out = outputs[pipeline]

    for file in out.values():

        Path(file).parent.mkdir(parents=True, exist_ok=True)

    # =====================================================
    # STEP I
    # SCHEMA ALIGNMENT
    # =====================================================

    if STEPS["schema_alignment"]:

        dfs = [utils.load_movies_csv(f) for f in inputs]

        if pipeline == "llm":

            global_schemas = prompt_aligning(dfs, DATASETS_NAMES, out["alignment_stats"])

            gpt_item = next(item for item in global_schemas if item["model"] == "openai/gpt-oss-120b")
            prediction = gpt_item["prediction"]

            # split rows
            rows = prediction.split("\n")
            csv_rows = [row.split(",") for row in rows]

            with open(out["global_schema"], "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerows(csv_rows)

            
            columns_to_keep = cols_to_keep(utils.load_movies_csv(out["global_schema"]))

        else:

            columns_to_keep = schema_alignment(dfs, DATASETS_NAMES, out["global_schema"])



        # normalization
        for i in range(len(dfs)):
            dfs[i] = dfs[i].drop(
                columns=[c for c in dfs[i].columns if c not in columns_to_keep]
            )

            dfs[i] = normalizer(dfs[i], INDEXES[i])


        merged_df = pd.concat(dfs, ignore_index=True)
        merged_df.to_csv(out["merged"], index=False)


    else:

        merged_df = utils.load_movies_csv(out["merged"])
        print("Loaded merged dataset:", len(merged_df))


    # =====================================================
    # STEP II
    # BLOCKING (COMMON)
    # =====================================================

    if STEPS["record_linkage"]["blocking"]:


        Path(COMMON["canopy"]).parent.mkdir(parents=True, exist_ok=True)

        canopy_cluster(merged_df, COMMON["canopy"])


    canopy_df = utils.load_movies_csv(COMMON["canopy"])

    # =================================================
    # MATCHING
    # =================================================

    if STEPS["record_linkage"]["matching"]:


        if pipeline == "classic":


            matches = match_records(
                canopy_df,
                out["matches"],
                out["singletons"],
                threshold=0.75
            )


        else:


            matches = llm_record_matching(
                canopy_df,
                out["matches"],
                out["requests"],
                llm_threshold=0.65,
                auto_threshold=0.75,
            )
    else:

        matches = utils.load_movies_csv(
            out["matches"]
        )


    # =================================================
    # CLUSTERING
    # =================================================

    if STEPS["record_linkage"]["clustering"]:

        print("START CLUSTERING")


        clusters = build_clusters(
            matches,
            merged_df,
            out["clusters"]
        )

    # =================================================
    # DATA FUSION
    # =================================================

    if STEPS["data_fusion"]:


        print("Starting data fusion")


        entities = utils.load_movies_csv(out["clusters"])


        fused = [

            fuse_cluster(group)

            for _, group

            in entities.groupby(
                "entity_id"
            )

        ]


        final_df = pd.concat(
            fused,
            ignore_index=True
        )


        final_df.to_csv(
            out["fusion"],
            index=False
        )


    print(
        pipeline,
        "completed"
    )