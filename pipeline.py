from src.utils.normalizator import normalizer
from src.schema_alignment.schema_alignment import schema_alignment, final_schema
from src.schema_alignment.llm_schema_alignment import prompt_aligning
from src.record_linkage.blocking.canopy_clustering import canopy_cluster
from src.record_linkage.record_matching.record_matching import match_records
from src.record_linkage.record_matching.llm_record_matching import llm_record_matching
from src.record_linkage.clustering.entity_clustering import build_clusters
from src.data_fusion.data_fusion import fuse_cluster
import src.utils.utils as utils
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

# =====================================================
# CONFIGURATION
# =====================================================

# Execution mode: "classic", "llm" or "both"
PIPELINE_MODE = "llm"


# Input datasets
INPUT_DIR = BASE_DIR / "data" / "dataset_cleaned"

inputs = [
    INPUT_DIR / "movies3_cleaned" / "imdb_cleaned.csv",
    INPUT_DIR / "movies3_cleaned" / "rotten_tomatoes_cleaned.csv",
    INPUT_DIR / "movies5_cleaned" / "imdb_cleaned.csv",
    INPUT_DIR / "movies5_cleaned" / "roger_ebert_cleaned.csv"
]


# Enable/disable individual pipeline stages
STEPS = {
    "schema_alignment": True,

    "record_linkage": {
        "blocking": True,
        "matching": True,
        "clustering": True
    },

    "data_fusion": True
}

# Temperature for LLM schema alignment
SCHEMA_TEMPERATURE = 0

# Parameters for canopy clustering (blocking phase)
# If a TF-IDF parameter is not defined, default values are used:
# analyzer="char_wb", ngram_range=(2,3), min_df=2.
canopy_params = {

    "columns": [
        "Title",
        "Director"
    ],

    "thresholds": {
        "Title": (0.40, 0.70),
        "Director": (0.40, 0.65)
    },

    "tfidf": {
        "Title": {
            "analyzer": "char_wb",
            "ngram_range": (2,3),
            "min_df": 2
        },
        "Director": {
            "analyzer": "char_wb",
            "ngram_range": (2,3),
            "min_df": 2
        }
    }
}


# Attribute weights and similarity functions used for record matching
matching_attributes = {
    "Title": {
        "weight": 0.50,
        "similarity": "text",
    },
    "Director": {
        "weight": 0.15,
        "similarity": "hybrid",
    },
    "Year": {
        "weight": 0.25,
        "similarity": "year",
    },
    "Cast": {
        "weight": 0.10,
        "similarity": "jaccard",
    }
}

# LLM configuration for record matching
LLM_MODEL = "openai/gpt-oss-120b"
MATCHING_TEMPERATURE = 0

# Source names and reliability weights used during data fusion
SOURCES = {
    "a": {
        "name": "imdb_3",
        "weight": 1.0
    },
    "b": {
        "name": "rotten_tomatoes",
        "weight": 0.2
    },
    "c": {
        "name": "imdb_5",
        "weight": 0.4
    },
    "d": {
        "name": "roger_ebert",
        "weight": 1.0
    }
}

# Fusion strategy adopted for each attribute
fusion_attributes = {
    "Title": "atomic",
    "Director": "multi",
    "Year": "atomic",
    "Cast": "multi",
    "Genre": "multi",
    "Duration": "atomic"
}


################################################################################################################################################################################################################################################################################

# =====================================================
# PIPELINE
# =====================================================


if PIPELINE_MODE == "both":
    PIPELINES = ["classic", "llm"]
else:
    PIPELINES = [PIPELINE_MODE]


BASE_DIR = BASE_DIR / "results"
model_dir = LLM_MODEL.replace("/", "_")


for pipeline in PIPELINES:


    print()
    print("==============================")
    print("Running:", pipeline)
    print("==============================")

################################################################################################################################################################################################################################################################################
# STEP I
# SCHEMA ALIGNMENT
################################################################################################################################################################################################################################################################################

    schema_dir = BASE_DIR / "schema_alignment" / pipeline
    
    if pipeline == "llm":
        schema_dir_dir = schema_dir / model_dir

    merged_path = schema_dir / "merged_movies.csv"


    if STEPS["schema_alignment"]:

        schema_dir.mkdir(parents=True, exist_ok=True)

        datasets = [s["name"] for s in SOURCES.values()]
        indexes = list(SOURCES.keys())

        dfs = [utils.load_movies_csv(f) for f in inputs]
        dfs = normalizer(dfs, indexes)


        if pipeline == "llm":
            
            ATTRIBUTE_DESCRIPTIONS = INPUT_DIR / "attribute_descriptions.json"
            
            # LLM global schema extraction
            prompt_aligning(
                dfs,
                datasets,
                schema_dir / "schema_alignment_results_stat.json",
                schema_dir / "global_schema.csv",
                ATTRIBUTE_DESCRIPTIONS,
                SCHEMA_TEMPERATURE
            )
            # list of columns to keep from each dataset
            dfs = final_schema(dfs, utils.load_movies_csv(schema_dir / "global_schema.csv"))

        else:
            print([df.columns for df in dfs])
            # list of columns to keep from each dataset
            dfs = schema_alignment(dfs, datasets, schema_dir / "global_schema.csv")


        merged_df = pd.concat(dfs, ignore_index=True)
        merged_df = merged_df.convert_dtypes()
        merged_df.to_csv(merged_path, index=False)

    else:

        merged_df = utils.load_movies_csv(merged_path)
    
    print("Loaded merged dataset:", len(merged_df))


################################################################################################################################################################################################################################################################################




################################################################################################################################################################################################################################################################################
# STEP II
# RECORD LINKAGE
################################################################################################################################################################################################################################################################################
    
    linkage_dir = BASE_DIR / "record_linkage" / pipeline

    if pipeline == "llm":
        linkage_dir = linkage_dir / model_dir


    # =====================================================
    # BLOCKING (COMMON)
    # =====================================================

    canopy_path = BASE_DIR / "record_linkage" / "canopy_blocks.csv"
    
    if STEPS["record_linkage"]["blocking"]:
        canopy_path.parent.mkdir(parents=True, exist_ok=True)
        canopy_cluster(
            merged_df,
            canopy_path,
            canopy_params
        )


    utils.path_check(
        [canopy_path],
        "record_linkage blocking step"
    )

    canopy_df = utils.load_movies_csv(canopy_path)
    
    # =================================================
    # MATCHING
    # =================================================
    
    matches_path = linkage_dir / "matches.csv"

    if STEPS["record_linkage"]["matching"]:

        linkage_dir.mkdir(
            parents=True,
            exist_ok=True
        )
        if pipeline == "classic":

            matches = match_records(
                canopy_df,
                matches_path,
                matching_attributes,
                threshold=0.75
            )

        else:

            matches = llm_record_matching(
                canopy_df,
                matches_path,
                linkage_dir / "llm_requests.csv",
                matching_attributes,
                llm_threshold=0.65,
                auto_threshold=0.75,
                model=LLM_MODEL,
                temperature=MATCHING_TEMPERATURE
            )
    else:

        matches = utils.load_movies_csv(matches_path)


################################################################################################################################################################################################################################################################################



################################################################################################################################################################################################################################################################################
# CLUSTERING
################################################################################################################################################################################################################################################################################
    
    clusters_path = linkage_dir / "entity_clusters.csv"
    singletons_path = linkage_dir / "singletons.csv"

    if STEPS["record_linkage"]["clustering"]:

        print("START CLUSTERING")


        clusters = build_clusters(
            matches,
            merged_df,
            0,
            clusters_path,
            singletons_path
        )

################################################################################################################################################################################################################################################################################



################################################################################################################################################################################################################################################################################
# DATA FUSION
################################################################################################################################################################################################################################################################################

    if STEPS["data_fusion"]:

        fusion_dir = BASE_DIR / "data_fusion" / pipeline
        
        if pipeline == "llm":
            fusion_dir = fusion_dir / model_dir

        fusion_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        print("Starting data fusion")

        entities = utils.load_movies_csv(clusters_path)

        fused = [
            fuse_cluster(
                group,
                fusion_attributes,
                SOURCES
            )
            for _, group in entities.groupby("entity_id")
        ]


        final_df = pd.concat(fused, ignore_index=True)
        # save to csv
        final_df.to_csv(fusion_dir / "fused_entities.csv", index=False)


    print(pipeline, "completed")

################################################################################################################################################################################################################################################################################