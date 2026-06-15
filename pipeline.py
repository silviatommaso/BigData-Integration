from roger_ebert_validationScraper import validation_scraper
from normalizator import normalizer

# STEP I: cleaning and preprocessing of datasets


#----------------------
#-----roger_ebert------
#----------------------

"In this step, we will clean and preprocess the Roger Ebert dataset. "
"We will handle missing values, standardize formats, and prepare the data for analysis. "
"This includes tasks such as:"

" - filling in missing director names (where possible)"
" - filling in missing cast members (where possible)"

validation_scraper()
normalizer("dataset_cleaned/movies5_cleaned/roger_ebert_final.csv", "id")