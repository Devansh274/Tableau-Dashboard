
import pandas as pd
import numpy as np
import re
import unicodedata
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "cleaned_data"

OUTPUT_DIR.mkdir(exist_ok=True)

BA_FILE = BASE_DIR / "ba_reviews.csv"
COUNTRIES_FILE = BASE_DIR / "Countries.csv"

report = []


def log(message):
    print(message)
    report.append(message)

def clean_text(value):

    if pd.isna(value):
        return np.nan

    value = unicodedata.normalize("NFKC", str(value))
    value = value.replace("\u200b", "")
    value = value.replace("\ufeff", "")
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def remove_duplicates(df, name):
    before = len(df)
    df = df.drop_duplicates().copy()
    removed = before - len(df)

    log(f"{name}: Removed {removed} duplicate rows")

    return df


def report_missing(df, name):
    log(f"\nMissing values in {name}:")

    missing = df.isna().sum()
    for column, count in missing.items():
        if count > 0:
            log(f"  {column}: {count}")

def clean_ba_reviews():

    log("\n= BA REVIEWS CLEANING =")

    df = pd.read_csv(BA_FILE)

    log(f"Original shape: {df.shape}")

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    text_columns = df.select_dtypes(
        include=["object", "string"]
    ).columns

    for col in text_columns:
        df[col] = df[col].apply(clean_text)

    date_columns = ["date", "date_flown"]

    for col in date_columns:
        df[col] = pd.to_datetime(
            df[col],
            format="%d/%m/%Y",
            errors="coerce"
        )

    df["recommended"] = (
        df["recommended"]
        .str.lower()
        .map({"yes": "Yes", "no": "No"})
    )

    df["trip_verified"] = (
        df["trip_verified"]
        .str.lower()
        .map({
            "verified": "Verified",
            "not verified": "Not Verified"
        })
    )

    rating_columns = [
        "rating",
        "seat_comfort",
        "cabin_staff_service",
        "food_beverages",
        "ground_service",
        "value_for_money",
        "entertainment"
    ]

    for col in rating_columns:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        )

    df.loc[
        ~df["rating"].between(1, 10),
        "rating"
    ] = np.nan

    service_columns = [
        "seat_comfort",
        "cabin_staff_service",
        "food_beverages",
        "ground_service",
        "value_for_money",
        "entertainment"
    ]

    for col in service_columns:

        df.loc[
            df[col] == -1,
            col
        ] = np.nan

        df.loc[
            ~df[col].between(1, 5),
            col
        ] = np.nan

    categorical_columns = [
        "place",
        "aircraft",
        "traveller_type",
        "seat_type"
    ]

    for col in categorical_columns:
        df[col] = df[col].replace("", np.nan)

    df = remove_duplicates(df, "BA Reviews")

    report_missing(df, "BA Reviews")

    for col in date_columns:
        df[col] = df[col].dt.strftime("%Y-%m-%d")

    output_file = OUTPUT_DIR / "ba_reviews_cleaned.csv"

    df.to_csv(output_file, index=False)

    log(f"Final shape: {df.shape}")
    log(f"Saved: {output_file}")

    return df


def clean_countries():

    log("\n= COUNTRIES CLEANING =")

    df = pd.read_csv(COUNTRIES_FILE)

    log(f"Original shape: {df.shape}")

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    for col in df.columns:
        df[col] = df[col].apply(clean_text)

    df["code"] = df["code"].str.upper()

    valid_code = df["code"].str.fullmatch(
        r"[A-Z]{3}", na=False
    )

    invalid_codes = (~valid_code).sum()

    log(f"Invalid/missing country codes: {invalid_codes}")


    df = remove_duplicates(df, "Countries")

    duplicate_countries = df["country"].duplicated().sum()

    log(f"Duplicate country names: {duplicate_countries}")

    report_missing(df, "Countries")

    output_file = OUTPUT_DIR / "countries_cleaned.csv"

    df.to_csv(output_file, index=False)

    log(f"Final shape: {df.shape}")
    log(f"Saved: {output_file}")

    return df


if __name__ == "__main__":

    ba_df = clean_ba_reviews()

    countries_df = clean_countries()

    report_file = OUTPUT_DIR / "cleaning_report.txt"

    with open(report_file, "w", encoding="utf-8") as file:
        file.write("\n".join(report))


    print(f"BA Reviews: {ba_df.shape}")
    print(f"Countries: {countries_df.shape}")
    print(f"Output folder: {OUTPUT_DIR}")
