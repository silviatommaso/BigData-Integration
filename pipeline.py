from normalizator import normalizer
from schema_alignment import schema_alignment, final_schema
from llm_schema_alignment import prompt_aligning
from canopy_clustering import canopy_cluster
from record_matching import match_records
from llm_record_matching import llm_record_matching
from entity_clustering import build_clusters
from data_fusion import fuse_cluster
import utils
from pathlib import Path
import pandas as pd




# =====================================================
# CONFIGURATION
# =====================================================

PIPELINE_MODE = "classic"


if PIPELINE_MODE == "both":
    PIPELINES = ["classic", "llm"]
else:
    PIPELINES = [PIPELINE_MODE]


STEPS = {
    "schema_alignment": True,

    "record_linkage": {
        "blocking": False,
        "matching": False,
        "clustering": False
    },

    "data_fusion": False
}



INPUT_DIR = Path("dataset_cleaned")

inputs = [
    INPUT_DIR / "movies3_cleaned" / "imdb_cleaned.csv",
    INPUT_DIR / "movies3_cleaned" / "rotten_tomatoes_cleaned.csv",
    INPUT_DIR / "movies5_cleaned" / "imdb_cleaned.csv",
    INPUT_DIR / "movies5_cleaned" / "roger_ebert_cleaned.csv"
]




# =====================================================
# COMMON FILES
# =====================================================

COMMON = {

    "canopy":
        "record_linkage/canopy_blocks.csv"

}


# =====================================================
# PIPELINE files
# =====================================================

files = {

    "classic": {

        "global_schema" : "schema_alignment/classic/global_schema.csv",
        "merged" : "schema_alignment/classic/merged_movies.csv",
        "matches" : "record_linkage/classic/matches.csv",
        "singletons" : "record_linkage/classic/singletons.csv",
        "clusters" : "record_linkage/classic/entity_clusters.csv",
        "fusion" : "data_fusion/classic/fused_entities.csv"
        
        },

    "llm": {

        "alignment_stats" : "schema_alignment/llm/schema_alignment_results_stat.json",
        "attribute_descriptions" : "schema_alignment/llm/attribute_descriptions.json",
        "global_schema" : "schema_alignment/llm/global_schema.csv",
        "merged" : "schema_alignment/llm/merged_movies.csv",
        "matches" : "record_linkage/llm/matches.csv",
        "requests" : "record_linkage/llm/llm_requests.csv",
        "clusters" : "record_linkage/llm/entity_clusters.csv",
        "fusion" : "data_fusion/llm/fused_entities.csv"

    }

}

INDEXES = ["a", "b", "c", "d"]
DATASETS_NAMES = ["imdb_3", "rotten_tomatoes", "imdb_5", "roger_ebert"]

















################################################################################################################################################################################################################################################################################

# =====================================================
# PIPELINE (CLASSIC + LLM)
# =====================================================

for pipeline in PIPELINES:


    print()
    print("==============================")
    print("Running:", pipeline)
    print("==============================")


    pipeline_files = files[pipeline]

    for file in pipeline_files.values():

        Path(file).parent.mkdir(parents=True, exist_ok=True)







################################################################################################################################################################################################################################################################################
# STEP I
# SCHEMA ALIGNMENT
################################################################################################################################################################################################################################################################################

    if STEPS["schema_alignment"]:

        dfs = [utils.load_movies_csv(f) for f in inputs]
        dfs = normalizer(dfs, INDEXES)


        if pipeline == "llm":

            # LLM global schema extraction
            prompt_aligning(dfs, DATASETS_NAMES, pipeline_files["alignment_stats"], pipeline_files["global_schema"], pipeline_files["attribute_descriptions"])
            # list of columns to keep from each dataset
            dfs = final_schema(dfs, utils.load_movies_csv(pipeline_files["global_schema"]))

        else:

            # list of columns to keep from each dataset
            dfs = schema_alignment(dfs, DATASETS_NAMES, pipeline_files["global_schema"])


        merged_df = pd.concat(dfs, ignore_index=True)
        merged_df.to_csv(pipeline_files["merged"], index=False)

    else:

        merged_df = utils.load_movies_csv(pipeline_files["merged"])
    
    print("Loaded merged dataset:", len(merged_df))


################################################################################################################################################################################################################################################################################




################################################################################################################################################################################################################################################################################
# STEP II
# RECORD LINKAGE
################################################################################################################################################################################################################################################################################

    # =====================================================
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
            matches = match_records(merged_df, canopy_df, pipeline_files["matches"], pipeline_files["singletons"], threshold=0.75)
        else:
            matches = llm_record_matching(merged_df, canopy_df, pipeline_files["matches"], pipeline_files["requests"], llm_threshold=0.65, auto_threshold=0.75,)
    
    else:

        matches = utils.load_movies_csv(pipeline_files["matches"])


################################################################################################################################################################################################################################################################################



################################################################################################################################################################################################################################################################################
# CLUSTERING
################################################################################################################################################################################################################################################################################

    if STEPS["record_linkage"]["clustering"]:

        print("START CLUSTERING")


        clusters = build_clusters(matches, merged_df, pipeline_files["clusters"])

################################################################################################################################################################################################################################################################################



################################################################################################################################################################################################################################################################################
# DATA FUSION
################################################################################################################################################################################################################################################################################

    if STEPS["data_fusion"]:


        print("Starting data fusion")


        entities = utils.load_movies_csv(pipeline_files["clusters"])


        fused = [fuse_cluster(group) for _, group in entities.groupby("entity_id")]


        final_df = pd.concat(fused, ignore_index=True)
        # save to csv
        final_df.to_csv(pipeline_files["fusion"], index=False)


    print(pipeline, "completed")

################################################################################################################################################################################################################################################################################