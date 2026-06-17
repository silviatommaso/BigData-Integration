import pandas as pd

def load_movies_csv(path):

    df=pd.read_csv(path)

    for col in ["Year","Duration"]:
        if col in df.columns:
            df[col]=df[col].astype("Int64")

    return df