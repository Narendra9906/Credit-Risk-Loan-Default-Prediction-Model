# %%
#  Data Cleaning Script for CSV Datasets

import pandas as pd

def clean_csv_data(filepath, output_filepath=None):
    """
    Clean a CSV dataset with common preprocessing steps.

    Parameters:
    - filepath: str, path to the input CSV file
    - output_filepath: str, optional, path to save the cleaned CSV

    Returns:
    - pd.DataFrame: cleaned DataFrame
    """
    # Load the data
    df = pd.read_csv(filepath)

    print(f"Original dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # 1. Remove duplicates
    df = df.drop_duplicates()
    print(f"After removing duplicates: {df.shape}")

    # 2. Handle missing values
    # Check for missing values
    missing_info = df.isnull().sum()
    print(f"Missing values per column:\n{missing_info[missing_info > 0]}")

    # Option 1: Drop rows with any missing values (use with caution)
    # df = df.dropna()

    # Option 2: Fill missing values (customize based on your data)
    # For numerical columns, fill with mean or median
    numerical_cols = df.select_dtypes(include=['number']).columns
    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())  # or .mean()

    # For categorical columns, fill with mode or a placeholder
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')

    # 3. Standardize data types (example: convert dates)
    # If you have date columns, uncomment and adjust:
    # date_cols = ['date_column_name']  # replace with actual column names
    # for col in date_cols:
    #     df[col] = pd.to_datetime(df[col], errors='coerce')

    # 4. Remove outliers (example for numerical columns using IQR)
    # for col in numerical_cols:
    #     Q1 = df[col].quantile(0.25)
    #     Q3 = df[col].quantile(0.75)
    #     IQR = Q3 - Q1
    #     lower_bound = Q1 - 1.5 * IQR
    #     upper_bound = Q3 + 1.5 * IQR
    #     df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    # 5. Trim whitespace from string columns
    for col in categorical_cols:
        df[col] = df[col].astype(str).str.strip()

    # 6. Convert categorical variables to consistent case (optional)
    # df[categorical_cols] = df[categorical_cols].apply(lambda x: x.str.lower())

    print(f"Cleaned dataset shape: {df.shape}")

    # Save to new CSV if output_filepath is provided
    if output_filepath:
        df.to_csv(output_filepath, index=False)
        print(f"Cleaned data saved to {output_filepath}")

    return df

# Example usage
if __name__ == "__main__":
    # Replace 'your_dataset.csv' with your actual CSV file path
    cleaned_df = clean_csv_data(r'Dataset\accepted_2007_to_2018Q4.csv', 'cleaned_dataset.csv')
    print("Data cleaning completed.")

# %%
%pip install matplotlib seaborn

# %%
%pip install scikit-learn

# %%
#  EDA - Exploratory Data Analysis

import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(df):
    """
    Perform Exploratory Data Analysis on the dataset
    """

    print("\n================ EDA STARTED ================\n")

    # Basic information
    print("\nDataset Info\n")
    print(df.info())

    print("\nStatistical Summary\n")
    print(df.describe())

    print("\nUnique values per column\n")
    print(df.nunique())

    # -------------------------------
    # 1. Target Variable Distribution
    # -------------------------------
    if 'loan_status' in df.columns:

        plt.figure(figsize=(8,5))
        df['loan_status'].value_counts().plot(kind='bar')
        plt.title("Loan Status Distribution")
        plt.xlabel("Loan Status")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.show()


    # -------------------------------
    # 2. Numerical Feature Distribution
    # -------------------------------
    numerical_cols = df.select_dtypes(include=['number']).columns

    for col in numerical_cols[:10]:  # limiting to first 10 for readability
        plt.figure(figsize=(6,4))
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.show()


    # -------------------------------
    # 3. Correlation Matrix
    # -------------------------------
    plt.figure(figsize=(12,8))
    corr = df[numerical_cols].corr()

    sns.heatmap(corr, cmap="coolwarm", annot=False)
    plt.title("Correlation Matrix")
    plt.show()


    # -------------------------------
    # 4. Loan Amount vs Default
    # -------------------------------
    if 'loan_amnt' in df.columns and 'loan_status' in df.columns:

        plt.figure(figsize=(8,5))
        sns.boxplot(x='loan_status', y='loan_amnt', data=df)
        plt.title("Loan Amount vs Loan Status")
        plt.show()


    # -------------------------------
    # 5. Interest Rate Analysis
    # -------------------------------
    if 'int_rate' in df.columns and 'loan_status' in df.columns:

        plt.figure(figsize=(8,5))
        sns.boxplot(x='loan_status', y='int_rate', data=df)
        plt.title("Interest Rate vs Loan Status")
        plt.show()


    # -------------------------------
    # 6. Annual Income Distribution
    # -------------------------------
    if 'annual_inc' in df.columns:

        plt.figure(figsize=(6,4))
        sns.histplot(df['annual_inc'], bins=50)
        plt.title("Annual Income Distribution")
        plt.show()


    # -------------------------------
    # 7. Debt-to-Income Ratio
    # -------------------------------
    if 'dti' in df.columns:

        plt.figure(figsize=(6,4))
        sns.histplot(df['dti'], bins=50)
        plt.title("Debt-to-Income Ratio Distribution")
        plt.show()


    # -------------------------------
    # 8. Top Loan Purposes
    # -------------------------------
    if 'purpose' in df.columns:

        plt.figure(figsize=(10,5))
        df['purpose'].value_counts().head(10).plot(kind='bar')
        plt.title("Top Loan Purposes")
        plt.xticks(rotation=45)
        plt.show()


    # -------------------------------
    # 9. Grade vs Default
    # -------------------------------
    if 'grade' in df.columns and 'loan_status' in df.columns:

        plt.figure(figsize=(8,5))
        sns.countplot(data=df, x='grade', hue='loan_status')
        plt.title("Loan Grade vs Loan Status")
        plt.show()


    print("\n================ EDA COMPLETED ================\n")

# Call the function with the cleaned dataframe
perform_eda(cleaned_df)

# %%
#  Feature Engineering for Credit Risk Modeling

import numpy as np

def feature_engineering(df):
    """
    Create financial risk features for credit risk modeling
    """

    print("\n================ FEATURE ENGINEERING STARTED ================\n")

    # Track original columns to identify new features
    original_cols = set(df.columns)

    # -----------------------------
    # 1. Loan to Income Ratio
    # -----------------------------
    if 'loan_amnt' in df.columns and 'annual_inc' in df.columns:
        df['loan_to_income_ratio'] = df['loan_amnt'] / (df['annual_inc'] + 1)


    # -----------------------------
    # 2. Installment to Income Ratio
    # -----------------------------
    if 'installment' in df.columns and 'annual_inc' in df.columns:
        df['installment_income_ratio'] = df['installment'] / (df['annual_inc'] + 1)


    # -----------------------------
    # 3. Credit Score Average
    # -----------------------------
    if 'fico_range_low' in df.columns and 'fico_range_high' in df.columns:
        df['avg_fico_score'] = (df['fico_range_low'] + df['fico_range_high']) / 2


    # -----------------------------
    # 4. Credit Utilization Ratio
    # -----------------------------
    if 'revol_bal' in df.columns and 'loan_amnt' in df.columns:
        df['credit_utilization_ratio'] = df['revol_bal'] / (df['loan_amnt'] + 1)


    # -----------------------------
    # 5. Account Density
    # -----------------------------
    if 'total_acc' in df.columns:
        df['account_density'] = df['total_acc'] / (df['total_acc'].max() + 1)


    # -----------------------------
    # 6. Debt Burden
    # -----------------------------
    if 'dti' in df.columns:
        df['debt_burden'] = df['dti'] / 100


    # -----------------------------
    # 7. Loan Term Risk
    # -----------------------------
    if 'term' in df.columns:
        df['term_months'] = df['term'].str.extract(r'(\d+)').astype(float)


    # -----------------------------
    # 8. Interest Rate Numeric
    # -----------------------------
    if 'int_rate' in df.columns:
        df['int_rate_numeric'] = df['int_rate'].astype(str).str.replace('%','').astype(float)


    # -----------------------------
    # 9. Income Log Transformation
    # -----------------------------
    if 'annual_inc' in df.columns:
        df['log_income'] = df['annual_inc'].apply(lambda x: np.log1p(x))


    # -----------------------------
    # 10. Loan Amount Log
    # -----------------------------
    if 'loan_amnt' in df.columns:
        df['log_loan_amount'] = df['loan_amnt'].apply(lambda x: np.log1p(x))


    # -----------------------------
    # 11. Credit History Length
    # -----------------------------
    if 'earliest_cr_line' in df.columns:
        df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'], errors='coerce')
        df['credit_history_length'] = (pd.Timestamp.now() - df['earliest_cr_line']).dt.days / 365


    # -----------------------------
    # 12. Revolving Utilization Numeric
    # -----------------------------
    if 'revol_util' in df.columns:
        df['revol_util_numeric'] = df['revol_util'].astype(str).str.replace('%','').astype(float)


    # -----------------------------
    # 13. Loan per Account
    # -----------------------------
    if 'loan_amnt' in df.columns and 'total_acc' in df.columns:
        df['loan_per_account'] = df['loan_amnt'] / (df['total_acc'] + 1)


    # -----------------------------
    # 14. Installment Burden
    # -----------------------------
    if 'installment' in df.columns and 'loan_amnt' in df.columns:
        df['installment_burden'] = df['installment'] / (df['loan_amnt'] + 1)


    # -----------------------------
    # 15. Income per Account
    # -----------------------------
    if 'annual_inc' in df.columns and 'total_acc' in df.columns:
        df['income_per_account'] = df['annual_inc'] / (df['total_acc'] + 1)


    new_features = set(df.columns) - original_cols
    print("New Features Created:")
    print(new_features)

    print("\nDataset Shape After Feature Engineering:", df.shape)

    print("\n================ FEATURE ENGINEERING COMPLETED ================\n")

    return df

# %%
# Call the feature engineering function
engineered_df = feature_engineering(cleaned_df)

# %%
# Labeling the Data

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def create_default_label(df):
    """
    Create binary default label from loan_status column
    """

    print("\n================ DEFAULT LABEL CREATION ================\n")

    if 'loan_status' not in df.columns:
        print("loan_status column not found!")
        return df

    # Define default categories
    default_status = [
        'Charged Off',
        'Default',
        'Late (31-120 days)',
        'Late (16-30 days)'
    ]

    # Create binary label
    df['default'] = df['loan_status'].apply(
        lambda x: 1 if x in default_status else 0
    )

    # Display distribution
    print("Default Label Distribution:")
    print(df['default'].value_counts())

    print("\nDefault Rate:")
    print(df['default'].mean())

    print("\n================ DEFAULT LABEL CREATED ================\n")

    return df

# %%
# Data Preparation Pipeline (scikit-learn)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def data_preparation_pipeline(df):
    """
    Prepare data using scikit-learn pipeline:
    - Feature scaling
    - Encoding categorical variables
    - Feature selection
    """
    print("\n================ DATA PREPARATION PIPELINE ================\n")

    # ----------------------------
    # Feature Selection
    # ----------------------------
    selected_features = [
        'loan_amnt',
        'annual_inc',
        'dti',
        'installment',
        'avg_fico_score',
        'loan_to_income_ratio',
        'installment_income_ratio',
        'credit_utilization_ratio',
        'credit_history_length',
        'int_rate_numeric',
        'term_months',
        'grade',
        'home_ownership',
        'purpose'
    ]

    # Keep only available columns
    selected_features = [col for col in selected_features if col in df.columns]

    X = df[selected_features]
    y = df['default']

    print("Selected Features:")
    print(selected_features)

    # ----------------------------
    # Identify Column Types
    # ----------------------------
    numerical_features = X.select_dtypes(include=['number']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    print("\nNumerical Features:")
    print(numerical_features)

    print("\nCategorical Features:")
    print(categorical_features)

    # ----------------------------
    # Scaling for numerical features
    # ----------------------------
    numerical_pipeline = Pipeline(
        steps=[
            ('scaler', StandardScaler())
        ]
    )
    # ----------------------------
    # Encoding categorical features
    # ----------------------------
    categorical_pipeline = Pipeline(
        steps=[
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]
    )

    # ----------------------------
    # Column Transformer
    # ----------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)
        ]
    )

    # Fit and transform
    X_processed = preprocessor.fit_transform(X)
    print("\nProcessed Feature Matrix Shape:")
    print(X_processed.shape)

    print("\n================ PIPELINE COMPLETED ================\n")

    return X_processed, y, preprocessor

# %%
# Risk Score Calculation

def calculate_risk_score(df):
    """
    Calculate borrower risk score based on financial indicators
    """
    print("\n================ RISK SCORE CALCULATION ================\n")

    # ----------------------------
    # Select risk features
    # ----------------------------
    risk_features = [
        'loan_to_income_ratio',
        'installment_income_ratio',
        'credit_utilization_ratio',
        'dti',
        'int_rate_numeric',
        'avg_fico_score'
    ]

    risk_features = [col for col in risk_features if col in df.columns]

    print("Risk Features Used:")
    print(risk_features)

    # ----------------------------
    # Normalize features
    # ----------------------------
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()

    df_scaled = df[risk_features].copy()
    df_scaled = scaler.fit_transform(df_scaled)
    df_scaled = pd.DataFrame(df_scaled, columns=risk_features)

    # ----------------------------
    # Risk weights
    # ----------------------------
    weights = {
        'loan_to_income_ratio': 0.20,
        'installment_income_ratio': 0.15,
        'credit_utilization_ratio': 0.20,
        'dti': 0.15,
        'int_rate_numeric': 0.10,
        'avg_fico_score': -0.20   # higher credit score reduces risk
    }

    # ----------------------------
    # Calculate risk score
    # ----------------------------
    df['risk_score'] = 0

    for feature in risk_features:
        weight = weights.get(feature, 0)
        df['risk_score'] += df_scaled[feature] * weight

    # Normalize risk score to 0-100
    df['risk_score'] = (df['risk_score'] - df['risk_score'].min()) / (
        df['risk_score'].max() - df['risk_score'].min()
    ) * 100

    # ----------------------------
    # Risk Segmentation
    # ----------------------------
    def risk_category(score):

        if score < 30:
            return "Low Risk"

        elif score < 60:
            return "Medium Risk"

        else:
            return "High Risk"

    df['risk_segment'] = df['risk_score'].apply(risk_category)

    # ----------------------------
    # Summary
    # ----------------------------
    print("\nRisk Score Statistics:")
    print(df['risk_score'].describe())

    print("\nRisk Segment Distribution:")
    print(df['risk_segment'].value_counts())

    print("\n================ RISK SCORING COMPLETED ================\n")

    return df

# %%
def risk_segmentation_analysis(df):
    """
    Analyze borrower risk segments and identify patterns
    """

    print("\n================ RISK SEGMENTATION ANALYSIS ================\n")

    # --------------------------------
    # 1. Default Rate by Risk Segment
    # --------------------------------
    segment_default = df.groupby('risk_segment')['default'].agg(
        total_loans='count',
        defaults='sum'
    )

    segment_default['default_rate'] = (
        segment_default['defaults'] / segment_default['total_loans']
    ) * 100

    print("\nDefault Rate by Risk Segment:")
    print(segment_default)


    # --------------------------------
    # 2. Loan Amount by Risk Segment
    # --------------------------------
    loan_amount_analysis = df.groupby('risk_segment')['loan_amnt'].agg(
        avg_loan='mean',
        median_loan='median',
        total_loans='count'
    )

    print("\nLoan Amount Analysis by Risk Segment:")
    print(loan_amount_analysis)


    # --------------------------------
    # 3. Default Rate by Loan Grade
    # --------------------------------
    if 'grade' in df.columns:

        grade_risk = df.groupby('grade')['default'].mean() * 100

        print("\nDefault Rate by Loan Grade:")
        print(grade_risk.sort_values(ascending=False))


    # --------------------------------
    # 4. Default Rate by Loan Purpose
    # --------------------------------
    if 'purpose' in df.columns:

        purpose_risk = df.groupby('purpose')['default'].mean() * 100

        print("\nTop Risky Loan Purposes:")
        print(purpose_risk.sort_values(ascending=False).head(10))


    # --------------------------------
    # 5. Income Risk Segmentation
    # --------------------------------
    if 'annual_inc' in df.columns:

        df['income_segment'] = pd.qcut(
            df['annual_inc'],
            q=4,
            labels=['Low Income', 'Lower-Middle', 'Upper-Middle', 'High Income']
        )

        income_risk = df.groupby('income_segment')['default'].mean() * 100

        print("\nDefault Rate by Income Segment:")
        print(income_risk)


    # --------------------------------
    # 6. Interest Rate Risk
    # --------------------------------
    if 'int_rate_numeric' in df.columns:

        df['interest_segment'] = pd.qcut(
            df['int_rate_numeric'],
            q=4,
            labels=['Low Interest', 'Moderate', 'High', 'Very High']
        )

        interest_risk = df.groupby('interest_segment')['default'].mean() * 100

        print("\nDefault Rate by Interest Rate Segment:")
        print(interest_risk)


    # --------------------------------
    # 7. Save summary tables for Power BI
    # --------------------------------
    segment_default.to_csv("risk_segment_summary.csv")
    loan_amount_analysis.to_csv("loan_amount_risk_summary.csv")

    print("\nSummary tables saved for Power BI.")

    print("\n================ RISK SEGMENTATION COMPLETED ================\n")

    return df

# %%
if __name__ == "__main__":

    df = clean_csv_data(r'Dataset\accepted_2007_to_2018Q4.csv')

    perform_eda(df)

    df = feature_engineering(df)

    df = create_default_label(df)

    X_processed, y, preprocessor = data_preparation_pipeline(df)

    df = calculate_risk_score(df)

    df = risk_segmentation_analysis(df)

    print("Risk analysis pipeline completed.")


