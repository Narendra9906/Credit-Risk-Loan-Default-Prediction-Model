import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder

# --- 1. Feature Engineering Function (Your "15+ features") ---
def engineer_features(data):
    """
    Engineers 15+ new features from the cleaned Lending Club data.
    """
    df = data.copy()
    
    # 1. FICO score average
    df['fico_score_avg'] = (df['fico_range_low'] + df['fico_range_high']) / 2
    
    # 2. Credit History Length (in years)
    df['credit_history_length_yrs'] = (datetime.datetime.now().year - df['earliest_cr_line'].dt.year)
    
    # 3. Loan to Income Ratio
    df['loan_to_income_ratio'] = df['loan_amnt'] / (df['annual_inc'] + 1e-6)
    
    # 4. Revolving Balance to Income Ratio
    df['revol_bal_to_income_ratio'] = df['revol_bal'] / (df['annual_inc'] + 1e-6)
    
    # 5. Debt to Income Ratio (DTI) per dollar of income
    # (Assuming dti is a percentage, if not, adjust)
    df['dti_per_dollar'] = df['dti'] / (df['annual_inc'] + 1e-6)
    
    # 6. Total Credit to FICO Score
    df['total_acc_to_fico'] = df['total_acc'] / df['fico_score_avg']
    
    # 7. Open Accounts to Total Accounts Ratio
    df['open_acc_to_total_acc_ratio'] = df['open_acc'] / (df['total_acc'] + 1e-6)
    
    # 8. Revolving Utilization to FICO
    df['revol_util_to_fico'] = df['revol_util'] / df['fico_score_avg']
    
    # 9. Installment to Income Ratio
    df['installment_to_income_ratio'] = df['installment'] * 12 / (df['annual_inc'] + 1e-6)
    
    # 10. Total Balance to Income Ratio
    df['tot_cur_bal_to_income_ratio'] = df['tot_cur_bal'] / (df['annual_inc'] + 1e-6)
    
    # 11. Total Revolving Limit to Income
    df['total_rev_hi_lim_to_income'] = df['total_rev_hi_lim'] / (df['annual_inc'] + 1e-6)
    
    # 12. FICO x Interest Rate Interaction
    df['fico_x_int_rate'] = df['fico_score_avg'] * df['int_rate']
    
    # 13. FICO x DTI Interaction
    df['fico_x_dti'] = df['fico_score_avg'] * df['dti']
    
    # 14. Term x Interest Rate
    df['term_x_int_rate'] = df['term'] * df['int_rate']
    
    # 15. Mortgages to Total Accounts
    df['mort_acc_to_total_acc'] = df['mort_acc'] / (df['total_acc'] + 1e-6)

    # 16. Delinquencies in last 2 years (as boolean)
    df['has_delinq_2yrs'] = (df['delinq_2yrs'] > 0).astype(int)
    
    # 17. Public Records (as boolean)
    df['has_pub_rec'] = (df['pub_rec'] > 0).astype(int)

    return df

# --- 2. Load and Prepare Data ---
print("Loading cleaned data...")
try:
    df = pd.read_csv('cleaned_loan_data.csv', parse_dates=['earliest_cr_line'])
except FileNotFoundError:
    print("Error: 'cleaned_loan_data.csv' not found. Please run preprocess.py first.")
    exit()

print(f"Cleaned data loaded: {df.shape}")

# Handle categorical features
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Define Base Features (pre-engineering)
TARGET = 'default'
base_features = list(df.columns.drop([TARGET, 'earliest_cr_line']))
X_base = df[base_features]
y = df[TARGET]

# Split for Base Model
X_train_base, X_test_base, y_train, y_test = train_test_split(X_base, y, test_size=0.2, random_state=42, stratify=y)

# --- 3. Base Model (To show F1 Improvement) ---
print("\nTraining BASE model (before feature engineering)...")
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

model_base = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight,
    random_state=42
)
model_base.fit(X_train_base, y_train)
y_pred_base = model_base.predict(X_test_base)
f1_base = f1_score(y_test, y_pred_base)
print(f"Base Model F1-Score: {f1_base:.4f}")

# --- 4. Engineered Model (Final Model) ---
print("\nTraining FINAL model (with 15+ engineered features)...")
df_engineered = engineer_features(df)

# Define Engineered Features
engineered_features = list(df_engineered.columns.drop([TARGET, 'earliest_cr_line']))
X_eng = df_engineered[engineered_features]
y_eng = df_engineered[TARGET]

# Split for Final Model
X_train_eng, X_test_eng, y_train_eng, y_test_eng = train_test_split(X_eng, y_eng, test_size=0.2, random_state=42, stratify=y_eng)

# Train Final Model
model_final = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_estimators=200,  # Tune as needed
    max_depth=5,
    learning_rate=0.1
)
model_final.fit(X_train_eng, y_train_eng)

# --- 5. Model Evaluation (AUC-ROC & F1) ---
y_pred_proba_final = model_final.predict_proba(X_test_eng)[:, 1]
y_pred_final = model_final.predict(X_test_eng)

auc_final = roc_auc_score(y_test_eng, y_pred_proba_final)
f1_final = f1_score(y_test_eng, y_pred_final)
f1_improvement = ((f1_final - f1_base) / f1_base) * 100

print("\n--- Final Model Performance ---")
print(f"AUC-ROC: {auc_final:.4f}") # Your 88% goal
print(f"F1-Score (Base): {f1_base:.4f}")
print(f"F1-Score (Final): {f1_final:.4f}")
print(f"F1-Score Improvement: {f1_improvement:.2f}%") # Your 20% goal
print("\nClassification Report:")
print(classification_report(y_test_eng, y_pred_final))

# --- 6. SHAP Analysis (Finding Top 5 Drivers) ---
print("\nRunning SHAP analysis...")
explainer = shap.TreeExplainer(model_final)
shap_values = explainer.shap_values(X_test_eng.sample(1000)) # Sample for speed

# Get feature importance
shap_sum = np.abs(shap_values).mean(axis=0)
feature_importance = pd.DataFrame(shap_sum, index=engineered_features, columns=['SHAP_Importance'])
feature_importance = feature_importance.sort_values(by='SHAP_Importance', ascending=False)

print("\nTop 5 Drivers of Default (from SHAP):")
print(feature_importance.head(5))

# --- 7. Save Model & Artifacts ---
print("Saving model and artifacts...")
model_final.save_model('loan_default_model.json')

artifacts = {
    'base_features': base_features, # The original features for the form
    'engineer_features_fn': engineer_features,
    'label_encoders': label_encoders,
    'explainer': explainer,
    'engineered_features_list': engineered_features
}

with open('loan_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)

print("\nTraining and artifact saving complete.")
