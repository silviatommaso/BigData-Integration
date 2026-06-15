from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import time
import pandas as pd
import os
import re


base_url = "https://www.rogerebert.com/reviews/"
output_file = "directors_cast_filled.csv"


df = pd.read_csv("dataset_cleaned/movies5_cleaned/roger_ebert_cleaned.csv")

missing = df[df["directors"].isnull() | df["actors"].isnull()].copy()
missing = missing.dropna(subset=["movie_name", "year"]).copy()


# ----------------------------
# SCRAPER KEYWORDS
# ----------------------------
KEYWORDS_DIRECTOR = [
    "written and directed by",
    "director",
    "directed by",
    "directed and written by"
]

KEYWORDS_CAST = ["cast"]




#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



# ----------------------------
# SLUG
# ----------------------------
def build_slug(title, year=None):
    title = str(title).lower().strip()

    # rimuove apostrofi
    title = title.replace("'", "")

    # rimuove i punti completamente (U.S. -> us)
    title = title.replace(".", "")

    # sostituisce qualsiasi altro carattere non alfanumerico con "-"
    title = re.sub(r"[^a-z0-9]+", "-", title)

    # elimina trattini multipli
    title = re.sub(r"-+", "-", title)

    # elimina trattini iniziali/finali
    title = title.strip("-")

    if year is None or pd.isna(year):
        return title

    return f"{title}-{int(year)}"


missing["slug_with_year"] = missing.apply(
    lambda r: build_slug(r["movie_name"], r["year"]),
    axis=1
)

missing["slug_title_only"] = missing["movie_name"].apply(
    lambda t: build_slug(t)
)



# Feature scraping function
def extract_features(page, keywords):
    soup = BeautifulSoup(page.content(), "html.parser")

    h4s = soup.select("h4.text-2xl.mb-1.font-heading-serif")

    for h4 in h4s:
        text = h4.get_text(strip=True).lower()

        if any(k in text for k in keywords):
            ul = h4.find_next("ul")
            if ul:
                return ", ".join(
                    a.get_text(strip=True) for a in ul.find_all("a")
                )

    return None



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



def validation_scraper():

    # ----------------------------
    # INIT OUTPUT FILE
    # ----------------------------
    if not os.path.exists(output_file):
        pd.DataFrame(columns=["movie_name", "year", "url", "directors_found", "cast_found"]).to_csv(
            output_file,
            index=False
        )
    else:
        return



    # start scraper
    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            user_data_dir="my_profile",
            headless=False
        )

        page = context.new_page()

        # cookie banner once
        try:
            page.goto("https://www.rogerebert.com", wait_until="domcontentloaded")
            time.sleep(2)
            page.locator("text=Accept").click(timeout=3000)
        except:
            pass

        for _, row in missing.iterrows():

            url1 = base_url + row["slug_with_year"]
            url2 = base_url + row["slug_title_only"]
            url3 = base_url + "great-movie-" + row["slug_with_year"]
            url4 = base_url + "great-movie-" + row["slug_title_only"]

            directors = None
            cast = None
            used_url = None

            for url in [url1, url2, url3, url4]:

                if not isinstance(url, str):
                    continue

                print("Scraping:", url)

                try:
                    page.goto(url, wait_until="domcontentloaded")
                    time.sleep(2)

                    directors = extract_features(page, KEYWORDS_DIRECTOR)
                    cast = extract_features(page, KEYWORDS_CAST)
                    if directors or cast:
                        used_url = url
                        break

                except Exception as e:
                    print("Errore:", url, e)

            # ----------------------------
            # WRITE ONLY IF FOUND
            # ----------------------------
            if directors is not None and directors.strip() != "" or cast is not None and cast.strip() != "":
                pd.DataFrame([{
                    "movie_name": row["movie_name"],
                    "year": row["year"],
                    "url": used_url,
                    "directors_found": directors,
                    "cast_found": cast
                }]).to_csv(
                    output_file,
                    mode="a",
                    header=False,
                    index=False
                )

                print("✔ FOUND:", row["movie_name"])

            else:
                print("✘ NOT FOUND:", row["movie_name"])

        context.close()

    print("DONE -> directors_cast_filled.csv")

    infer_and_fill_directors(
        original_csv="dataset_cleaned/movies5_cleaned/roger_ebert_cleaned.csv",
        enriched_csv="directors_cast_filled.csv",
        output_csv="dataset_cleaned/movies5_cleaned/roger_ebert_final.csv"
    )


########################################################################################################################################################################


# Infer missing directors values with the found ones in the output file
def infer_and_fill_directors(original_csv, enriched_csv, output_csv):

    # ----------------------------
    # LOAD
    # ----------------------------
    df = pd.read_csv(original_csv)
    enriched = pd.read_csv(enriched_csv)
    enriched["cast_found"] = enriched["cast_found"].replace("", pd.NA)
    enriched["cast_found"] = enriched["cast_found"].astype("string")

    # ----------------------------
    # KEYS NORMALIZATION
    # ----------------------------
    df["movie_name"] = df["movie_name"].str.strip().str.lower()
    enriched["movie_name"] = enriched["movie_name"].str.strip().str.lower()

    # year sicuro (evita float tipo 2015.0)
    df["year"] = df["year"].astype(str).str.replace(".0", "", regex=False)
    enriched["year"] = enriched["year"].astype(str).str.replace(".0", "", regex=False)

    # ----------------------------
    # MAP DIRECTORS - CAST
    # ----------------------------
    enriched_map = (
        enriched
        .set_index(["movie_name", "year"])[["directors_found", "cast_found"]]
        .to_dict("index")
    )

    # ------------------------------
    # FILL MISSING DIRECTORS - CAST
    # ------------------------------
    def fill_missing(row):

        key = (row["movie_name"], row["year"])

        if key not in enriched_map:
            return row

        found = enriched_map[key]

        cast_found = found.get("cast_found")
        director_found = found.get("directors_found")

        if pd.isnull(row["directors"]) and director_found:
            row["directors"] = director_found

        if pd.isnull(row["actors"]) and cast_found:
            row["actors"] = cast_found

        return row
    

    df = df.apply(fill_missing, axis=1)

    # ----------------------------
    # SAVE FINAL DATASET
    # ----------------------------
    df.to_csv(output_csv, index=False)

    print("DONE -> dataset filled saved at:", output_csv)

    # intermidiate file cleanup
    os.remove(enriched_csv)

    return df