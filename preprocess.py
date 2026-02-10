# preprocess.py

import pandas as pd
import sqlite3

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def clean_data(df):
    # Drop duplicates
    df = df.drop_duplicates()

    # Handle missing values
    df = df.fillna(method="ffill")

    # Convert target column (example: loan_status)
    df["loan_status"] = df["loan_status"].apply(
        lambda x: 1 if x == "Default" else 0
    )

    return df

def save_to_sql(df, db_name="loan_data.db"):
    conn = sqlite3.connect(db_name)
    df.to_sql("loans", conn, if_exists="replace", index=False)
    conn.close()

if __name__ == "__main__":
    data = load_data("loan_data.csv")
    cleaned_data = clean_data(data)
    save_to_sql(cleaned_data)
    print("Data cleaning and SQL storage completed.")
