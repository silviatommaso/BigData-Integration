import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno

df = pd.read_csv("tmdb.csv", sep=';', low_memory=False)

# print("DATA TYPES:")
# mask = (df["cast"].isna()) & (df["type"] == "TV Show")
# df_masked = df.loc[~mask]
# print(df_masked)
# print(df_masked.loc[df_masked["director"].isna(), "listed_in"].unique())

# print("NULL VALUES:")
# print(df.info())
# print(df.isnull().sum())


# msno.matrix(df[["director", "cast", "country"]].sort_values(by="director"))
# plt.show()