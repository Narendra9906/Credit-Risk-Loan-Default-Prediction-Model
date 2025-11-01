import pandas as pd
import numpy as np
import datetime

def preprocess_data(raw_data_path, output_path):
    """
    Loads raw Lending Club data, cleans it, defines the target variable,
    and removes data leaks to create a clean dataset for modeling.
    """
    print("Starting data preprocessing...")
    
    # --- 1. Load Data ---
    # Load only a subset of columns to save memory. These are the ones
    # available at the time of application.
    # We also include 'loan_status' to create our target.
    
    # Columns available at application time
    features_to_load = [
        'loan_amnt', 'term', 'int_rate', 'installment', 'grade', 'sub_grade',
        'emp_length', 'home_ownership', 'annual_inc', 'verification_status',
        'issue_d', 'loan_status', 'purpose', 'title', 'zip_code', 'addr_state',
        'dti', 'delinq_2yrs', 'earliest_cr_line', 'fico_range_low', 'fico_range_high',
        'inq_last_6mths', 'mths_since_last_delinq', 'mths_since_last_record',
        'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc',
        'initial_list_status', 'application_type', 'mort_acc',
        'pub_rec_bankruptcies', 'tot_cur_bal', 'total_rev_hi_lim'
    ]
    
    try:
        # The main file from Kaggle is 'accepted_2007_to_2018q4.csv'
        df = pd.read_csv(raw_data_path, usecols=features_to_load, 
                         parse_dates=['issue_d', 'earliest_cr_line'])
        print(f"Raw data loaded: {df.shape}")
    except FileNotFoundError:
        print(f"Error: Raw data file not found at {raw_data_path}")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # --- 2. Define Target Variable (Default) ---
    # This is crucial. We only want loans that are 'Fully Paid' or 'Charged Off'.
    # 'Current' loans are not useful as we don't know their outcome.
    
    df['default'] = df['loan_status'].apply(lambda x: 1 if x in ['Charged Off', 'Default', 
                                                              'Does not meet the credit policy. Status:Charged Off'] else 0)
    
    # Filter to keep only completed loans
    df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off', 'Default',
                                   'Does not meet the credit policy. Status:Charged Off',
                                   'Does not meet the credit policy. Status:Fully Paid'])]
    
    print(f"Data filtered for completed loans: {df.shape}")

    # --- 3. Filter for 150K+ Dataset ---
    # The full dataset is huge. Let's use data from 2016-2017 to get a
    # dataset that matches your "150K+" description.
    df = df[df['issue_d'].dt.year.isin([2016, 2017])]
    print(f"Data filtered for 2016-2017: {df.shape}")

    # --- 4. Handle Missing Values (Example) ---
    # This is a complex step, here are a few key examples.
    
    # Fill numerical NaNs with 0 or median (domain-specific)
    df['mths_since_last_delinq'] = df['mths_since_last_delinq'].fillna(0)
    df['mths_since_last_record'] = df['mths_since_last_record'].fillna(0)
    df['emp_length'] = df['emp_length'].fillna('0 years')
    df['revol_util'] = df['revol_util'].fillna(df['revol_util'].median())
    df['mort_acc'] = df['mort_acc'].fillna(0)
    
    # Drop rows where critical data is missing
    df = df.dropna(subset=['dti', 'annual_inc', 'fico_range_low'])

    # --- 5. Clean/Convert Features ---
    # Convert string features to numerical
    df['emp_length'] = df['emp_length'].str.replace(r'\+ years', '').str.replace('< 1 year', '0').str.replace(' years', '').str.replace(' year', '').astype(int)
    df['term'] = df['term'].str.strip().str.replace(' months', '').astype(int)
    
    # --- 6. Drop Leaky/Unused Columns ---
    # We drop loan_status as we now have our 'default' target
    # We drop issue_d as we don't want to train on the loan date
    df = df.drop(columns=['loan_status', 'issue_d', 'title', 'zip_code', 'addr_state'])

    # --- 7. Save Cleaned Data ---
    df.to_csv(output_path, index=False)
    print(f"Preprocessing complete. Cleaned data saved to {output_path}")
    print(f"Final dataset shape: {df.shape}")
    print(f"Final default rate: {df['default'].mean() * 100:.2f}%")

if __name__ == "__main__":
    # Assumes you have the 'accepted_2007_to_2018q4.csv' in the same folder
    RAW_DATA_FILE = 'accepted_2007_to_2018q4.csv'
    CLEANED_DATA_FILE = 'cleaned_loan_data.csv'
    preprocess_data(RAW_DATA_FILE, CLEANED_DATA_FILE)
