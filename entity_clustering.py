import networkx as nx
import pandas as pd


def build_clusters(matches):

    G=nx.Graph()

    for _,row in matches.iterrows():

        G.add_edge(
            row["id1"],
            row["id2"],
            weight=row["score"]
        )


    clusters=[]

    for entity_id,component in enumerate(nx.connected_components(G)):

        clusters.append({
            "entity_id":entity_id,
            "records":list(component)
        })


    return clusters


def get_unmatched_records(matches,df):

    matched_ids=set(matches["id1"]) | set(matches["id2"])

    return df[
        ~df["ID"].isin(matched_ids)
    ]


def save_clusters(clusters,path):

    rows=[]

    for entity in clusters:

        for record_id in entity["records"]:

            rows.append({
                "entity_id":entity["entity_id"],
                "ID":record_id
            })


    pd.DataFrame(rows).to_csv(
        path,
        index=False
    )