from roger_ebert_validationScraper import validation_scraper
from normalizator import normalizer

# STEP I: datasets cleaning and normalization


#----------------------
#-----imdb movie3------
#----------------------

normalizer("dataset_cleaned/movies3_cleaned/imdb_cleaned.csv")

#--------------------------
#-----rotten_tomatoes------
#--------------------------

normalizer("dataset_cleaned/movies3_cleaned/rotten_tomatoes_cleaned.csv")


#----------------------
#-----roger_ebert------
#----------------------

"In this step, we will clean and preprocess the Roger Ebert dataset. "
"We will handle missing values, standardize formats, and prepare the data for analysis. "
"This includes tasks such as:"

" - filling in missing director names (where possible)"
" - filling in missing cast members (where possible)"

validation_scraper()
normalizer("dataset_cleaned/movies5_cleaned/roger_ebert_final.csv")

#----------------------
#-----imdb movie5------
#----------------------

normalizer("dataset_cleaned/movies5_cleaned/imdb_cleaned.csv")