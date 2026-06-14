from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import time
import pandas as pd
import os


base_url = "https://www.rogerebert.com/reviews/"
output_file = "directors_filled.csv"

# # ----------------------------
# # LOAD DATA
# # ----------------------------
df = pd.read_csv("dataset_cleaned/movies5_cleaned/roger_ebert_cleaned.csv")

missing = df[df["directors"].isnull()].copy()
missing = missing.dropna(subset=["movie_name", "year"]).copy()

# ----------------------------
# SLUG
# ----------------------------
def build_slug(title, year=None):
    title = str(title).lower().strip().replace(" ", "-")

    if year is None or pd.isna(year):
        return title

    return f"{title}-{int(year)}"

missing["slug_with_year"] = missing.apply(
    lambda r: build_slug(r["movie_name"], r["year"]),
    axis=1
)

missing["slug_title_only"] = missing["movie_name"].apply(
    lambda t: str(t).lower().strip().replace(" ", "-")
)

# ----------------------------
# SCRAPER
# ----------------------------
KEYWORDS = [
    "written and directed by",
    "director",
    "directed by",
    "directed and written by"
]

def extract_directors(page):
    soup = BeautifulSoup(page.content(), "html.parser")

    h4s = soup.select("h4.text-2xl.mb-1.font-heading-serif")

    for h4 in h4s:
        text = h4.get_text(strip=True).lower()

        if any(k in text for k in KEYWORDS):
            ul = h4.find_next("ul")
            if ul:
                return ", ".join(
                    a.get_text(strip=True) for a in ul.find_all("a")
                )

    return None

#######################################################################################################################################################################


#######################################################################################################################################################################


#######################################################################################################################################################################


def validation_scraper():

    # ----------------------------
    # INIT OUTPUT FILE
    # ----------------------------
    if not os.path.exists(output_file):
        pd.DataFrame(columns=["movie_name", "year", "url", "directors_found"]).to_csv(
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
            used_url = None

            for url in [url1, url2, url3, url4]:

                if not isinstance(url, str):
                    continue

                print("Scraping:", url)

                try:
                    page.goto(url, wait_until="domcontentloaded")
                    time.sleep(2)

                    directors = extract_directors(page)

                    if directors:
                        used_url = url
                        break

                except Exception as e:
                    print("Errore:", url, e)

            # ----------------------------
            # WRITE ONLY IF FOUND
            # ----------------------------
            if directors is not None and directors.strip() != "":
                pd.DataFrame([{
                    "movie_name": row["movie_name"],
                    "year": row["year"],
                    "url": used_url,
                    "directors_found": directors
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

    print("DONE -> directors_filled.csv")

    infer_and_fill_directors(
        original_csv="dataset_cleaned/movies5_cleaned/roger_ebert_cleaned.csv",
        enriched_csv="directors_filled.csv",
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

    # ----------------------------
    # NORMALIZZA CHIAVI (IMPORTANTISSIMO)
    # ----------------------------
    df["movie_name"] = df["movie_name"].str.strip().str.lower()
    enriched["movie_name"] = enriched["movie_name"].str.strip().str.lower()

    # year sicuro (evita float tipo 2015.0)
    df["year"] = df["year"].fillna(-1).astype(int)
    enriched["year"] = enriched["year"].fillna(-1).astype(int)

    # ----------------------------
    # CREA MAPPA DIRECTORS
    # ----------------------------
    enriched_map = enriched.set_index(
        ["movie_name", "year"]
    )["directors_found"].to_dict()

    # ----------------------------
    # FILL MISSING DIRECTORS
    # ----------------------------
    def fill_director(row):
        if pd.notna(row["directors"]):
            return row["directors"]

        key = (row["movie_name"], row["year"])
        return enriched_map.get(key, row["directors"])

    df["directors"] = df.apply(fill_director, axis=1)

    # ----------------------------
    # SAVE FINAL DATASET
    # ----------------------------
    df.to_csv(output_csv, index=False)

    print("DONE -> dataset filled saved at:", output_csv)

    #rimozione file intermedio
    os.remove(enriched_csv)

    return df