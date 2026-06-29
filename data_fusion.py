import pandas as pd
import re
from collections import defaultdict
from difflib import SequenceMatcher




# ----------------------------
# Source ID extractor
# ----------------------------
def get_source(record_id):
    return str(record_id)[0].lower()


def clean_id(record_id):
    return re.sub(r'^[A-Za-z]+-?', '', str(record_id))

#------------------------------------------------------------------------------------------------------------------------------------------------------------


# ----------------------------
# Lists parser
# ----------------------------
def parse_list(x):
    if pd.isna(x):
        return set()
    return set([i.strip().lower() for i in str(x).split(",") if i.strip()])


# Director
def parse_director(x):
    return parse_list(x)

# Cast
def parse_cast(x):
    return parse_list(x)

# Genre
def parse_genre(x):
    genres = set()
    if pd.isna(x):
        return genres
    for g in str(x).split(","):
        g = g.strip().lower()
        if "&" in g:
            parts = [p.strip() for p in g.split("&")]
            genres.update(parts)
        else:
            genres.add(g)
    return genres


def list_to_string(x):
        if x is None:
            return ""
        if isinstance(x, (list, set, tuple)):
            return ", ".join(str(i) for i in x)
        return str(x)


#------------------------------------------------------------------------------------------------------------------------------------------------------------


# ============================
# score calculators
# ============================


# ----------------------------
# atomic fields
# ----------------------------
def weighted_mode(values, sources, weights):
    score = defaultdict(float)

    for v, s in zip(values, sources):
        if pd.isna(v):
            continue
        score[v] += weights.get(s, 0)

    if not score:
        return None

    # take values with maximum score
    return max(score.items(), key=lambda x: x[1])[0]


# ----------------------------
# multivalued fields
# ----------------------------
def confidence_fusion(list_values, sources, weights, threshold_ratio=0.3):
    score = defaultdict(float)

    for vals, s in zip(list_values, sources):
        w = weights.get(s, 0)
        for v in vals:
            score[v] += w

    if not score:
        return []

    # confidence filter
    max_score = max(score.values())
    dynamic_threshold = max_score * threshold_ratio
    candidates = sorted([v for v, sc in score.items() if sc >= dynamic_threshold], key=lambda x: score[x], reverse=True)

    # duplicate deletion
    fused_names = []
    
    for candidate in candidates:
        duplicate_found = False
        
        # Split and orders candidates (ex: "phil lord" -> ["lord", "phil"])
        words_cand = candidate.split()
        cand_ordered = " ".join(sorted(words_cand))
        
        for name in fused_names:
            words_name = name.split()
            ordered_name = " ".join(sorted(words_name))
            
            # check duplicate 
            if cand_ordered == ordered_name:
                duplicate_found = True
                break
            



            # dynamic threshold
            min_len = min(len(candidate), len(name))
            
            if min_len < 10:
                similarity_threshold = 0.80
            else:
                similarity_threshold = 0.73


            # string similarity
            sim_original = SequenceMatcher(None, candidate, name).ratio()
            sim_ordered = SequenceMatcher(None, cand_ordered, name).ratio()
            actual_sim = max(sim_original, sim_ordered)
            
            # check diminuitives
            inters = set(words_cand).intersection(set(words_name))
            words_sharing = len(inters) >= 1
            
            contained = cand_ordered in ordered_name or ordered_name in cand_ordered
            

            if actual_sim >= similarity_threshold or (words_sharing and contained):
                duplicate_found = True
                break
                
        if not duplicate_found:
            fused_names.append(candidate)

    return sorted(fused_names)

##################################################################################################################################################################


# ----------------------------
# main
# ----------------------------
def fuse_cluster(df, attributes, sources_config, id_position=1, output_path=None):

    id_column = df.columns[id_position]


    # weights estratti dalla config
    weights = {
        source: config["weight"]
        for source, config in sources_config.items()
    }


    # source extraction
    sources = df[id_column].apply(get_source)


    fused = {}


    # ----------------------------
    # provenance GENERICA
    # ----------------------------

    for source in sources_config:

        fused[f"{source.upper()}_Ids"] = ";".join(
            df.loc[
                sources == source,
                id_column
            ]
            .apply(clean_id)
            .astype(str)
        )


    # ----------------------------
    # ATTRIBUTES GENERICI
    # ----------------------------

    for column, field_type in attributes.items():


        if column not in df.columns:
            continue


        if field_type == "atomic":


            fused[column] = weighted_mode(
                df[column],
                sources,
                weights
            )


        elif field_type == "multi":


            values = df[column].apply(parse_list)


            fused[column] = confidence_fusion(
                values,
                sources,
                weights
            )


            fused[column] = list_to_string(
                fused[column]
            )

    return pd.DataFrame([fused])