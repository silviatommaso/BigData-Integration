import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno

df_imdb = pd.read_csv("movies3/csv_files/rotten_tomatoes.csv", low_memory=False)
df_rottentomatoes = pd.read_csv("rotten_tomatoes.csv", low_memory=False)
df_rogerebert = pd.read_csv("roger_ebert.csv", low_memory=False)

#########################
##### DATA ANALYSIS #####
#########################

# print("IMDB DATASET ANALYSIS\n")
# print("\n")

# print("\n")

# print("NULL VALUES:\n")
# print(df_imdb.info())
# print("\n")
# null_values_imdb = df_imdb.isnull().sum()
# print(null_values_imdb)

# print("\n")

# print("NULL STATISTICS:\n ")
# null_imdb_values = df_imdb[["Creator", "Cast", "Duration", "RatingValue", "Genre", "Description"]].isnull().sum()
# percentage_null_imdb_values = null_imdb_values/len(df_imdb[["Creator", "Cast", "Duration", "RatingValue", "Genre", "Description"]])
# print(percentage_null_imdb_values)

################################################################################################################################################

print("\n")
print("\n")
print("\n")

################################################################################################################################################

print("ROTTEN TOMATOES DATASET ANALYSIS\n")
print("\n")

print("\n")

print("NULL VALUES:\n")
print(df_rottentomatoes.info())
print("\n")
null_values_rottentomatoes = df_rottentomatoes.isnull().sum()
print(null_values_rottentomatoes)

print("\n")

print("NULL STATISTICS:\n ")
null_rottentomatoes_values = df_rottentomatoes[["Release Date", "Creator", "Actors", "Cast", "Duration", "RatingValue", "RatingCount", "ReviewCount", "Genre", "Filming Locations"]].isnull().sum()
percentage_null_rottentomatoes_values = null_rottentomatoes_values/len(df_rottentomatoes[["Release Date", "Creator", "Actors", "Cast", "Duration", "RatingValue", "RatingCount", "ReviewCount", "Genre", "Filming Locations"]])
print(percentage_null_rottentomatoes_values)

# msno.matrix(df_rottentomatoes[["Release Date", "Creator", "Country"]].sort_values(by="Release Date"))
# plt.show()

################################################################################################################################################

print("\n")
print("\n")
print("\n")

################################################################################################################################################

print("ROGER EBERT DATASET ANALYSIS\n")
print("\n")

print("\n")

print("NULL VALUES:\n")
print(df_rogerebert.info())
print("\n")
null_values_rogerebert = df_rogerebert.isnull().sum()
print(null_values_rogerebert)

print("\n")

print("NULL STATISTICS:\n ")
null_rogerebert_values = df_rogerebert[["year", "directors", "actors", "genre", "pg_rating", "duration"]].isnull().sum()
percentage_null_rogerebert_values = null_rogerebert_values/len(df_rogerebert[["year", "directors", "actors", "genre", "pg_rating", "duration"]])
print(percentage_null_rogerebert_values)