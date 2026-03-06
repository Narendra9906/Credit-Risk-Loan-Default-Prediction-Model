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
    cleaned_df = clean_csv_data('your_dataset.csv', 'cleaned_dataset.csv')
    print("Data cleaning completed.")