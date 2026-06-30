import networkx as nx
import pandas as pd


def save_clusters(clusters, merged_df, id_column, path):

    lookup = merged_df.set_index(id_column)
    rows = []

    for entity in clusters:
        for record_id in entity["records"]:

            original_row = lookup.loc[record_id].to_dict()

            cluster_row = {
                "entity_id": entity["entity_id"],
                id_column: record_id
            }

            # aggiungo tutti gli altri attributi dopo
            cluster_row.update(original_row)

            rows.append(cluster_row)

    df = pd.DataFrame(rows)


    # to avoid float values for 
    for column in df.columns:

        if df[column].dtype == "float64":

            if df[column].dropna().apply(float.is_integer).all():

                df[column] = df[column].astype("Int64")
    pd.DataFrame(rows).to_csv(
        path,
        index=False
    )

    print("Clusters saved:", len(clusters))
    print("Entity found:", len(clusters))



################################################################################################


def build_clusters(matches, merged_df, merged_id_position, clusters_path, singletons_path, save=True):

    id_column = merged_df.columns[merged_id_position]

    G = nx.Graph()


    for _, row in matches.iterrows():

        G.add_edge(
            row["id1"],
            row["id2"],
            weight=row["score"]
        )


    clusters = []

    for entity_id, component in enumerate(nx.connected_components(G)):

        clusters.append({
            "entity_id": entity_id,
            "records": list(component)
        })


    matched_ids = set(matches["id1"]) | set(matches["id2"])

    singletons = merged_df[
        ~merged_df[id_column].isin(matched_ids)
    ]


    print("Clusters found:", len(clusters))
    print("Singletons found:", len(singletons))


    if save:

        save_clusters(
            clusters,
            merged_df,
            id_column,
            clusters_path
        )

        singletons.to_csv(
            singletons_path,
            index=False
        )


    return clusters, singletons