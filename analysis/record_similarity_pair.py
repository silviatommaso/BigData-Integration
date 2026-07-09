import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from config import matching_attributes
from src.record_linkage.record_matching.record_matching import record_similarity


def record_similarity_pair(id1, id2, merged, id_column="Id"):

    r1 = merged.loc[merged[id_column] == id1].iloc[0].to_dict()
    r2 = merged.loc[merged[id_column] == id2].iloc[0].to_dict()

    score, similarities = record_similarity(
        r1,
        r2,
        matching_attributes
    )

    print("=" * 60)
    print("RECORD SIMILARITY FOR ONE PAIR")
    print("=" * 60)

    print(f"\nRecord pair: {id1} <-> {id2}")

    print("\nAttribute similarities:")
    for attribute, similarity in similarities.items():
        print(f"{attribute}: {similarity}")

    print("\nFinal weighted score:", score)

    print("\nWeight contribution:")
    for attribute, config in matching_attributes.items():

        sim = similarities[attribute]

        if sim is not None:
            contribution = config["weight"] * sim
            print(
                f"{attribute}: "
                f"{config['weight']} * {sim} = {round(contribution, 3)}"
            )

merged_path = ROOT / "results/schema_alignment/classic/merged_movies.csv"
merged = pd.read_csv(merged_path)



# Example usage: compare the similarity between two records identified by their IDs
record_similarity_pair(
    "a48",
    "b50",
    merged,
    id_column="Id"
)