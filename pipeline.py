from normalizator import normalizer



############################
# STEP I -> SCHEMA ALIGNMENT
############################

#---------------------------------------------------------
# ID | Title | Year | Director | Cast | Genre | Duration |
#---------------------------------------------------------

#-------imdb movie3--------
normalizer("dataset_cleaned/movies3_cleaned/imdb_cleaned.csv", "a")
#-----rotten_tomatoes------
normalizer("dataset_cleaned/movies3_cleaned/rotten_tomatoes_cleaned.csv", "b")
#-------roger_ebert--------
normalizer("dataset_cleaned/movies5_cleaned/roger_ebert_final.csv", "c")
#-------imdb movie5--------
normalizer("dataset_cleaned/movies5_cleaned/imdb_cleaned.csv", "d")


#