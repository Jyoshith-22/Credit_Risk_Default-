# Credit Risk Default Prediction & Scoring Portal

This project implements an end-to-end Machine Learning solution to predict borrower loan default risk, segment customers, and provide real-time underwriting decisions. It is based on a credit application dataset containing demographics, income, loan request info, and historical credit behavior.

---

## 🚀 Key Features & Deliverables

1. **Jupyter Notebook (`Credit_Risk_Default_Prediction.ipynb`):** A fully documented, pre-run notebook containing Exploratory Data Analysis, outlier removal, missing value imputation, domain feature engineering, model training (Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM), Optuna hyperparameter optimization, and SHAP model explanations.
2. **PDF Case Study Report (`Credit_Risk_Prediction_Report.pdf`):** A publication-grade PDF report summarizing the project scope, preprocessing pipelines, model comparison tables, ROC curves, FICO credit score bands, and underwriting recommendations.
3. **Interactive Streamlit Dashboard (`app.py`):** An interactive portal showcasing portfolio analytics, model comparison metrics, and a **Real-Time Underwriting Calculator** to predict default probability and score bands for new applicants.
4. **Machine Learning Artifacts:** Serialized best-performing model (`best_credit_risk_model.pkl`) and preprocessing pipeline pipelines (`preprocessing_artifacts.pkl`).

---

## 📊 Model Performance Comparison

Optimized with K-Fold cross-validation, the Tuned LightGBM model outperforms all baselines:

| Model | ROC-AUC | PR-AUC | F1-Score | Precision | Recall | KS-Statistic |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM (Tuned)** | **0.9515** | **0.9082** | **0.8172** | **0.8509** | **0.7861** | **0.7650** |
| LightGBM (Default) | 0.9457 | 0.8979 | 0.8031 | 0.8100 | 0.7963 | 0.7445 |
| XGBoost | 0.9460 | 0.8991 | 0.8003 | 0.8096 | 0.7912 | 0.7450 |
| Random Forest | 0.9338 | 0.8844 | 0.8150 | 0.9037 | 0.7421 | 0.7278 |
| Decision Tree | 0.9021 | 0.8523 | 0.7710 | 0.7963 | 0.7473 | 0.6983 |
| Logistic Regression | 0.8764 | 0.7323 | 0.6411 | 0.5399 | 0.7890 | 0.6176 |

---

## 🛠️ How to Run the Project

### 1. Requirements Installation
To install the required libraries, execute:
```bash
pip install -r requirements.txt
```
*(Libraries needed: pandas, numpy, scikit-learn, xgboost, lightgbm, imbalanced-learn, shap, optuna, matplotlib, seaborn, reportlab, streamlit)*

### 2. Run the Streamlit Dashboard
Launch the interactive dashboard to run the underwriting calculator and view portfolio insights:
```bash
streamlit run app.py
```

### 3. Re-train & Optimize Models
To run the preprocessing, Optuna tuning, and plot generations from scratch:
```bash
python train_models.py
```

### 4. Compile the PDF Report
To re-generate the PDF report document containing the updated tables and figures:
```bash
python generate_pdf_report.py
```
