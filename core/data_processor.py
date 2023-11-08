import json
from argparse import ArgumentParser
from pandas import read_csv
from sqlalchemy import create_engine


def create_db(brands: str, categories: str, offers: str, output: str) -> None:
    """
    Create a database from CSV files with brand, category, and offer data.

    Args:
    - brands (str): Path to the CSV file containing brand and category data.
    - categories (str): Path to the CSV file containing category mapping data.
    - offers (str): Path to the CSV file containing offer and retailer data.
    - output (str): Path to the SQLite database file to be created.

    This function loads data from the provided CSV files, aggregates and combines information,
    and stores it in an SQLite database. The resulting database table includes columns for
    'BRAND', 'RETAILER', 'CATEGORIES', 'SUPER_CATEGORIES', and 'TARGET', where 'TARGET' is a
    string that combines various information for each offer. The database is created at the
    specified output location.
    """

    # Load the datasets
    brand_category_df = read_csv(brands)
    offer_retailer_df = read_csv(offers)
    categories_df = read_csv(categories)

    # Group brand_category_df by 'BRAND_BELONGS_TO_CATEGORY' (brand) and aggregate categories
    brand_categories = (
        brand_category_df.groupby("BRAND")["BRAND_BELONGS_TO_CATEGORY"]
        .agg(list)
        .reset_index()
    )
    brand_categories.rename(
        columns={"BRAND_BELONGS_TO_CATEGORY": "CATEGORIES"}, inplace=True
    )

    # Map the aggregated categories to the 'BRAND' column in offer_retailer_df
    merged_df = offer_retailer_df.merge(brand_categories, on="BRAND", how="left")

    # Create a mapping of 'PRODUCT_CATEGORY' to 'IS_CHILD_CATEGORY_TO'
    category_mapping = categories_df.set_index("PRODUCT_CATEGORY")[
        "IS_CHILD_CATEGORY_TO"
    ].to_dict()

    # Function to get unique 'IS_CHILD_CATEGORY_TO' values for each brand
    def get_super_categories(categories):
        super_categories = set()
        if isinstance(categories, list):
            for category in categories:
                super_category = category_mapping.get(category)
                if super_category:
                    super_categories.add(super_category)
        return list(super_categories) if super_categories else None

    # Apply the function to the 'CATEGORIES' column
    merged_df["SUPER_CATEGORIES"] = merged_df["CATEGORIES"].apply(get_super_categories)

    # Preprocess the text data and replace NaN values with an empty string
    merged_df.RETAILER.fillna("", inplace=True)
    merged_df.BRAND.fillna("", inplace=True)
    merged_df.CATEGORIES.fillna("", inplace=True)
    merged_df.SUPER_CATEGORIES.fillna("", inplace=True)

    merged_df["TARGET"] = merged_df["TARGET"] = (
        merged_df["BRAND"]
        + " ; "
        + merged_df["RETAILER"]
        + " ; "
        + merged_df["CATEGORIES"].str.join(", ")
        + " ; "
        + merged_df["SUPER_CATEGORIES"].str.join(", ")
    ).str.lower()

    # JSONize columns
    merged_df.CATEGORIES = merged_df.CATEGORIES.apply(lambda x: [] if x == "" else x)
    merged_df.SUPER_CATEGORIES = merged_df.SUPER_CATEGORIES.apply(
        lambda x: [] if x == "" else x
    )
    merged_df.CATEGORIES = merged_df.CATEGORIES.apply(json.dumps)
    merged_df.SUPER_CATEGORIES = merged_df.SUPER_CATEGORIES.apply(json.dumps)

    # Create a SQLAlchemy engine to connect to the database
    engine = create_engine(f"sqlite:///{output}")

    # Convert the DataFrame to a SQL table and write it to the database
    merged_df.to_sql("offers", engine, index=True, if_exists="replace")


if __name__ == "__main__":
    # Argument parser
    parser = ArgumentParser(description="Process CSV data files")

    # Define the command-line arguments for data paths with default values
    parser.add_argument(
        "--brands",
        default="data/raw/brand_category.csv",
        type=str,
        help="Path to the brand_category.csv file",
    )
    parser.add_argument(
        "--categories",
        default="data/raw/categories.csv",
        type=str,
        help="Path to the categories.csv file",
    )
    parser.add_argument(
        "--offers",
        default="data/raw/offer_retailer.csv",
        type=str,
        help="Path to the offer_retailer.csv file",
    )
    parser.add_argument(
        "--output",
        default="data/processed/database.sqlite",
        type=str,
        help="Path to the output SQLite database file",
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    create_db(**vars(args))
