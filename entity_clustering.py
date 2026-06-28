import networkx as nx
import pandas as pd


def save_clusters(clusters, merged_df, path):

    lookup = merged_df.set_index("Id")
    rows = []

    for entity in clusters:
        for record_id in entity["records"]:
            row = lookup.loc[record_id]
            rows.append({
                "entity_id": entity["entity_id"],
                "Id": record_id,
                "Title": row["Title"],
                "Year": row["Year"],
                "Director": row["Director"],
                "Cast": row["Cast"],
                "Genre": row["Genre"],
                "Duration": row["Duration"]
            })
            
    pd.DataFrame(rows).to_csv(path, index=False)
    print("Clusters saved:", len(clusters))


    pd.DataFrame(rows).to_csv(path, index=False)

    print("Entità  trovate:", len(clusters))
    print("Clusters salvati")


################################################################################################################################################################################


def build_clusters(matches, merged_df, clusters_path, singletons_path, save=True):
    G = nx.Graph()

    for _, row in matches.iterrows():
        G.add_edge(row["id1"], row["id2"], weight=row["score"])

    clusters = []
    for entity_id, component in enumerate(nx.connected_components(G)):
        clusters.append({
            "entity_id": entity_id,
            "records": list(component)
        })

    # singletons: records that never appeared in any match
    matched_ids = set(matches["id1"]) | set(matches["id2"])
    singletons = merged_df[~merged_df["Id"].isin(matched_ids)]

    print("Clusters found:", len(clusters))
    print("Singletons found:", len(singletons))

    if save:
        save_clusters(clusters, merged_df, clusters_path)
        singletons.to_csv(singletons_path, index=False)
    else:
        print("Run without saving")

    return clusters, singletons