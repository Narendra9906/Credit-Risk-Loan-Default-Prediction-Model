# train_model.py

import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import joblib

def load_from_sql(db_name="loan_data.db"):
    conn = sqlite3.connect(db_name)
    df = pd.read_sql("SELECT * FROM loans", conn)
    conn.close()
    return df

def train_model(df):
    X = df.drop("loan_status", axis=1)
    y = df["loan_status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("Classification Report:")
    print(classification_report(y_test, predictions))
    print("ROC-AUC Score:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

    joblib.dump(model, "loan_model.pkl")

if __name__ == "__main__":
    data = load_from_sql()
    train_model(data)
