import pandas as pd


# function to reindex a CSV file by adding an ID column with incremental values starting from 1
def indicization(df, file_path, id_column_name):
    df[id_column_name] = range(1, len(df) + 1)
    df.to_csv(file_path, index=False)


# function to normalize string values in a CSV file by converting them to lowercase
def normalize_lowercase(df, file_path):
    df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)
    df.to_csv(file_path, index=False)




def normalizer(file_path, id_column_name):

    # valid per each file
    df = pd.read_csv(file_path)

    indicization(df, file_path, id_column_name)
    normalize_lowercase(df, file_path)