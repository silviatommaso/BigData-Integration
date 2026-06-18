import networkx as nx
import pandas as pd


def save_clusters(clusters,path):

    rows=[]

    for entity in clusters:

        for record_id in entity["records"]:

            rows.append({
                "entity_id":entity["entity_id"],
                "ID":record_id
            })


    pd.DataFrame(rows).to_csv(path, index=False)

    print("Entità  trovate:", len(clusters))
    print("Clusters salvati")


################################################################################################################################################################################


def build_clusters(matches, output_path):

    G=nx.Graph()

    # create a graph with matched ids as nodes and score as arc's weight
    for _,row in matches.iterrows():

        G.add_edge(
            row["id1"],
            row["id2"],
            weight=row["score"]
        )


    # per each connected component assign an identificative number (cluster id)
    clusters=[]

    for entity_id,component in enumerate(nx.connected_components(G)):

        clusters.append({
            "entity_id":entity_id,
            "records":list(component)
        })


    save_clusters(clusters, output_path)

    return clusters