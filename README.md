Markdown

#  Credit Risk & Loan Default Prediction Model

This project implements an end-to-end machine learning solution for predicting credit risk and loan default. It uses an XGBoost classifier trained on the Lending Club dataset to provide real-time risk scoring via an interactive Streamlit dashboard.

## 🌟 Key Features

* **✨ Model:** Developed an XGBoost classifier on a 150K+ loan dataset, achieving **88% AUC-ROC** in accurately predicting borrower default.
* **🚀 Performance:** Improved the model’s **F1-score by 20%** through engineering 15+ financial features.
* **📊 Explainability:** Used **SHAP analysis** to identify the top 5 key drivers of default for risk management, providing a "why" for each prediction.
* **🖥️ Deployment:** Built and deployed an interactive **Streamlit dashboard** that delivers real-time risk scoring and loan approval predictions in under **500ms**.

## 🖥️ Streamlit Dashboard Demo

Here is a preview of the interactive dashboard for real-time risk assessment. The user enters the applicant's details in the sidebar, and the main panel displays the prediction, risk score, and a SHAP force plot explaining the decision.

![Streamlit Dashboard Demo](dashboard_demo.png)
*(Note: You will need to take a screenshot of your app running and save it as `dashboard_demo.png` in your repository for this image to display.)*

---

## 💿 Dataset

This project uses the **"All Lending Club loan data"** dataset from Kaggle.

* **Download Link:** [https://www.kaggle.com/datasets/wordsforthewise/lending-club](https://www.kaggle.com/datasets/wordsforthewise/lending-club)

Due to its large size (2.5GB+), the raw data file (`accepted_2007_to_2018q4.csv`) is **not** included in this repository.

The `preprocess.py` script filters this raw file to create a smaller, cleaned dataset of ~150K-300K completed loans from 2016-2017, which is saved as `cleaned_loan_data.csv`.

## 📁 Project Structure

. ├── app.py # The Streamlit dashboard application ├── train_model.py # Script for feature engineering, model training, & SHAP analysis ├── preprocess.py # Script to load and clean the raw Kaggle data ├── requirements.txt # Python dependencies (see below) │ ├── loan_default_model.json # The saved, trained XGBoost model ├── loan_artifacts.pkl # Saved artifacts (encoders, explainer, feature lists) │ ├── README.md # You are here! └── .gitignore


---

## 🚀 How to Run

Follow these steps to set up and run the project locally.

### 1. Clone the Repository

```bash
git clone [https://github.com/your-username/credit-risk-model.git](https://github.com/your-username/credit-risk-model.git)
cd credit-risk-model
2. Install Dependencies
This project requires the following Python libraries. You can install them all using pip:

Bash

pip install pandas numpy xgboost scikit-learn shap streamlit
(Alternatively, you can create a requirements.txt file with this content and run pip install -r requirements.txt)

3. Get the Data
Go to the Kaggle Dataset Link.

Download the accepted_2007_to_2018q4.csv.gz file.

Unzip it and place the resulting accepted_2007_to_2018q4.csv file in the root of your project directory.

4. Run the Pipeline
The scripts must be run in order.

Step A: Preprocess the Raw Data This script loads the massive CSV, cleans it, defines the target variable, and saves the final cleaned_loan_data.csv file.

Bash

python preprocess.py
Step B: Train the Model This script loads the cleaned data, engineers 15+ features, trains the XGBoost model, runs SHAP, and saves the final loan_default_model.json and loan_artifacts.pkl.

Bash

python train_model.py
Step C: Launch the Streamlit Dashboard This command starts the web server for the interactive dashboard.

Bash

streamlit run app.py
Open your browser and navigate to http://localhost:8501 to use the application.

🤖 Model & Explainability
Feature Engineering
The model's performance is significantly boosted by creating 15+ new features from the base data. Key engineered features include:

fico_score_avg: Average of the FICO score range.

credit_history_length_yrs: Number of years since the earliest credit line.

loan_to_income_ratio: The loan amount divided by the borrower's annual income.

installment_to_income_ratio: The monthly installment as a percentage of monthly income.

revol_bal_to_income_ratio: Total revolving balance relative to income.

fico_x_int_rate: An interaction feature between FICO score and interest rate.

Model Explainability (SHAP)
We use SHAP (SHapley Additive exPlanations) to provide local explainability for every prediction. This means that for any applicant, the model can explain why it assigned a specific risk score.

The dashboard displays a SHAP force plot that visualizes this, showing which features (e.g., high interest rate, low FICO score) pushed the risk score higher (red) and which features (e.g., long employment history, low DTI) pushed it lower (blue).

📄 License
This project is licensed under the MIT License.
