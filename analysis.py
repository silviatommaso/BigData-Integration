import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno

df_imdb_movie3 = pd.read_csv("dataset_cleaned/movies3_cleaned/imdb_cleaned.csv", low_memory=False)
df_imdb_movie5 = pd.read_csv("dataset_cleaned/movies5_cleaned/imdb_cleaned.csv", low_memory=False)
df_rottentomatoes = pd.read_csv("dataset_cleaned/movies3_cleaned/rotten_tomatoes_cleaned.csv", low_memory=False)
df_rogerebert = pd.read_csv("dataset_cleaned/movies5_cleaned/roger_ebert_final.csv", low_memory=False)

#########################
##### DATA ANALYSIS #####
#########################

print("IMDB MOVIES 3 DATASET ANALYSIS\n")
print("\n")

print("\n")

print("NULL VALUES:\n")
print(df_imdb_movie3.info())
print("\n")
null_values_imdb = df_imdb_movie3.isnull().sum()
print(null_values_imdb)

print("\n")

print("NULL STATISTICS:\n ")
null_imdb_values = df_imdb_movie3[["Year", "Creators", "Cast", "Genre", "Duration", "ContentRating", "Summary"]].isnull().sum()
percentage_null_imdb_values = null_imdb_values/len(df_imdb_movie3[["Year", "Creators", "Cast", "Genre", "Duration", "ContentRating", "Summary"]])
print(percentage_null_imdb_values)

# msno.matrix(df_imdb_movie3[["Year", "Creators", "Cast", "Genre", "Duration", "ContentRating", "Summary"]].sort_values(by="Year"))
# plt.show()

################################################################################################################################################

print("\n")
print("\n")
print("\n")

################################################################################################################################################

print("ROTTEN TOMATOES MOVIES 3 DATASET ANALYSIS\n")
print("\n")

print("\n")

print("NULL VALUES:\n")
print(df_rottentomatoes.info())
print("\n")
null_values_rottentomatoes = df_rottentomatoes.isnull().sum()
print(null_values_rottentomatoes)

print("\n")

print("NULL STATISTICS:\n ")
null_rottentomatoes_values = df_rottentomatoes[["Year", "Rating", "Director", "Creators", "Cast", "Genre", "Duration", "Summary"]].isnull().sum()
percentage_null_rottentomatoes_values = null_rottentomatoes_values/len(df_rottentomatoes[["Year", "Rating", "Director", "Creators", "Cast", "Genre", "Duration", "Summary"]])
print(percentage_null_rottentomatoes_values)

# msno.matrix(df_rottentomatoes[["Year", "Rating", "Director", "Creators", "Cast", "Genre", "Duration", "Summary"]].sort_values(by="Year"))
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

# msno.matrix(df_rogerebert[["year", "directors", "actors", "genre", "pg_rating", "duration"]].sort_values(by="directors"))
# plt.show()

################################################################################################################################################

print("\n")
print("\n")
print("\n")

################################################################################################################################################

print("IMDB MOVIES 5 DATASET ANALYSIS\n")
print("\n")

print("\n")

print("NULL VALUES:\n")
print(df_imdb_movie5.info())
print("\n")
null_values_imdb = df_imdb_movie5.isnull().sum()
print(null_values_imdb)

print("\n")

print("NULL STATISTICS:\n ")
null_imdb_values = df_imdb_movie5[["actors"]].isnull().sum()
percentage_null_imdb_values = null_imdb_values/len(df_imdb_movie5[["actors"]])
print(percentage_null_imdb_values)