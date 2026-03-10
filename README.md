# 💳 Credit Risk & Loan Default Analysis

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.9-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Power%20BI-Dashboard-F2C811?style=for-the-badge&logo=powerbi&logoColor=black"/>
  <img src="https://img.shields.io/badge/Status-Completed-2ea44f?style=for-the-badge"/>
</p>

---

## 📌 Project Overview

This project focuses on analyzing borrower financial data to identify **loan default risk patterns** and assist in **data-driven loan approval decisions**.

Using **Python for data analysis** and **Power BI for visualization**, the project builds a complete end-to-end analytics pipeline covering data cleaning, feature engineering, risk scoring, segmentation, and an interactive dashboard.

> 📊 Analyzes **150K+ loan records** to understand borrower behavior and financial risk indicators.

---

## 🎯 Project Goals

- Analyze borrower financial data to uncover default patterns
- Identify key factors influencing loan default
- Engineer financial risk indicators from raw data
- Generate normalized borrower **risk scores (0–100)**
- Segment borrowers into actionable risk groups
- Build an interactive **Power BI dashboard** for loan decision support

---

## 📂 Dataset

**Source:** Lending Club Loan Dataset

| Feature | Description |
|---|---|
| `loan_amnt` | Requested loan amount |
| `int_rate` | Interest rate on the loan |
| `annual_inc` | Borrower's annual income |
| `dti` | Debt-to-income ratio |
| `fico_score` | Borrower credit score |
| `grade` | Loan grade assigned by lender |
| `purpose` | Purpose of the loan |
| `loan_status` | Current status of the loan |

---

## 🔄 Project Workflow

```
Data Cleaning
     ↓
Exploratory Data Analysis (EDA)
     ↓
Feature Engineering
     ↓
Default Label Creation
     ↓
Data Preparation Pipeline (scikit-learn)
     ↓
Risk Score Calculation
     ↓
Risk Segmentation Analysis
     ↓
Power BI Dashboard
```

---

## 🧹 Data Cleaning

- Removed duplicate records
- Handled missing values
- Standardized column formats
- Converted percentage columns to numeric values
- Trimmed whitespace and verified data types

---

## 🔍 Exploratory Data Analysis (EDA)

Key analyses performed to identify borrower patterns and risk factors:

- Loan status distribution
- Loan amount distribution
- Debt-to-income ratio trends
- Interest rate vs. default risk
- Loan grade vs. default rate
- Correlation heatmap analysis

---

## ⚙️ Feature Engineering

**15+ financial features** were engineered to capture borrower repayment capacity and financial health:

| Feature | Description |
|---|---|
| `loan_to_income_ratio` | Loan amount relative to income |
| `installment_to_income_ratio` | Monthly burden on income |
| `credit_utilization_ratio` | Credit used vs. available |
| `avg_fico_score` | Average credit score |
| `credit_history_length` | Length of credit history |
| `log_loan_amount` | Log-transformed loan amount |
| `log_income` | Log-transformed annual income |
| `income_per_account` | Income spread across accounts |
| `loan_per_account` | Loan spread across accounts |

---

## 🏷️ Default Label Creation

Loan status values were converted into a **binary default label**:

| Loan Status | Default Label |
|---|:---:|
| Charged Off | `1` |
| Default | `1` |
| Late Payment | `1` |
| Fully Paid | `0` |
| Current | `0` |

---

## 🛠️ Data Preparation Pipeline

A **scikit-learn pipeline** was used to prepare the dataset for risk scoring:

```python
# Numerical feature scaling
StandardScaler()

# Categorical encoding
OneHotEncoder()
```

**Key features used:**
`loan_amnt` · `annual_inc` · `dti` · `installment` · `avg_fico_score` · `credit_utilization` · `loan_to_income_ratio` · `int_rate`

---

## 📊 Risk Score Calculation

A **custom weighted risk scoring mechanism** was built using key financial indicators:

| Risk Factor | Contribution |
|---|---|
| Debt-to-Income Ratio | High |
| Credit Utilization | High |
| Loan-to-Income Ratio | Medium |
| Installment Burden | Medium |
| Interest Rate | Medium |
| Credit Score | High |

> Scores normalized to a **0–100 scale** — higher score = higher default risk.

---

## 🚦 Risk Segmentation

Borrowers are classified into three risk tiers:

| Risk Score | Risk Category |
|:---:|:---:|
| 0 – 30 | 🟢 Low Risk |
| 30 – 60 | 🟡 Medium Risk |
| 60 – 100 | 🔴 High Risk |

---

## 📈 Risk Segmentation Insights

Additional analyses for targeted decision-making:

- Default rate by **risk segment**
- Default rate by **loan grade**
- Risk patterns by **loan purpose**
- **Income group** risk profiles
- **Interest rate** risk patterns

---

## 📊 Power BI Dashboard

The final processed dataset is visualized in an interactive **Power BI dashboard**:

| Dashboard Page | Content |
|---|---|
| Loan Portfolio Overview | Total loans, volume, status breakdown |
| Risk Distribution | Borrower risk score distribution |
| Default Rate by Grade | Grade-wise default comparison |
| Borrower Risk Segments | Risk tier breakdowns |
| Loan Purpose Risk | Purpose-wise default rates |
| Income vs Default Risk | Income group risk profiles |

> 📸 *Add your Power BI dashboard screenshot here*

---

## 📁 Project Structure

```
Credit-Risk-Loan-Default-Prediction-Model/
│
├── data/
│   ├── raw_dataset.csv
│   └── cleaned_dataset.csv
│
├── notebooks/
│   └── analysis.ipynb
│
├── scripts/
│   └── credit_risk_analysis.py
│
├── dashboard/
│   └── powerbi_dashboard.pbix
│
└── README.md
```

---

## 🚀 Installation & Setup

```bash
# Clone the repository
git clone https://github.com/Narendra9906/Credit-Risk-Loan-Default-Prediction-Model.git

# Navigate to project directory
cd Credit-Risk-Loan-Default-Prediction-Model

# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn
```

---

## ▶️ Running the Project

```bash
python credit_risk_analysis.py
```

**Generated output files:**

| File | Description |
|---|---|
| `cleaned_dataset.csv` | Cleaned loan records |
| `processed_credit_data.csv` | Feature-engineered dataset |
| `risk_segment_summary.csv` | Risk segment statistics |
| `loan_amount_risk_summary.csv` | Loan amount risk breakdown |

> Import these files into **Power BI** to build the dashboard.

---

## 🧰 Tools & Technologies

| Category | Tools |
|---|---|
| Programming | Python 3.9 |
| Data Analysis | Pandas, NumPy |
| ML Utilities | Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Dashboard | Power BI |

---

## 💡 Key Insights

- 📌 Borrowers with **high debt-to-income ratios** show significantly higher default probability
- 📌 **Low credit score borrowers** are at the greatest risk of default
- 📌 **Loan grades F and G** exhibit the highest default rates across all segments
- 📌 Loans for **debt consolidation and small business** carry elevated risk profiles

---

## 🔮 Future Improvements

- [ ] Integrate advanced ML credit scoring models (XGBoost, LightGBM)
- [ ] Automate risk reporting pipeline
- [ ] Build a real-time loan risk dashboard
- [ ] Implement portfolio-level risk monitoring

---

## 👤 Author

**Narendra Lal**  
Integrated M.Sc. Chemistry | National Institute of Technology, Rourkela

<p align="left">
  <img src="https://img.shields.io/badge/Python-Proficient-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/SQL-Proficient-4479A1?style=flat-square&logo=mysql&logoColor=white"/>
  <img src="https://img.shields.io/badge/Power%20BI-Proficient-F2C811?style=flat-square&logo=powerbi&logoColor=black"/>
  <img src="https://img.shields.io/badge/Machine%20Learning-Proficient-F7931E?style=flat-square&logo=scikit-learn&logoColor=white"/>
</p>

---

<p align="center">⭐ If you found this project helpful, please consider giving it a star!</p>
