import nbformat as nbf
import os

def create_notebook():
    nb = nbf.v4.new_notebook()
    
    # Define cells
    cells = []
    
    # 1. Title
    cells.append(nbf.v4.new_markdown_cell(
        "# Credit Risk Default Prediction"
    ))
    
    # 2. Setup
    cells.append(nbf.v4.new_code_cell(
        "import os\n"
        "import pandas as pd\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "import joblib\n"
        "from sklearn.model_selection import train_test_split, StratifiedKFold\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.metrics import (\n"
        "    roc_auc_score, precision_recall_curve, auc, f1_score,\n"
        "    precision_score, recall_score, roc_curve, classification_report\n"
        ")\n"
        "from sklearn.linear_model import LogisticRegression\n"
        "from sklearn.tree import DecisionTreeClassifier\n"
        "from sklearn.ensemble import RandomForestClassifier\n"
        "import xgboost as xgb\n"
        "import lightgbm as lgb\n"
        "import optuna\n"
        "import shap\n\n"
        "sns.set_theme(style='whitegrid')\n"
        "plt.rcParams['figure.figsize'] = (10, 6)\n"
        "plt.rcParams['font.size'] = 11\n"
        "optuna.logging.set_verbosity(optuna.logging.WARNING)"
    ))
    
    # 3. Load Data
    cells.append(nbf.v4.new_markdown_cell(
        "## Load Data"
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "df = pd.read_csv('credit_risk_dataset.csv')\n"
        "print(df.shape)\n"
        "df.head()"
    ))
    
    # 4. EDA
    cells.append(nbf.v4.new_markdown_cell(
        "## Exploratory Data Analysis"
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "print(df['loan_status'].value_counts(normalize=True))\n\n"
        "print(df.isnull().sum())\n\n"
        "df['person_age'].describe()"
    ))
    
    # 5. Outliers and Cleaning
    cells.append(nbf.v4.new_markdown_cell(
        "## Data Cleaning"
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "df_clean = df[df['person_age'] <= 100].copy()\n"
        "df_clean = df_clean[df_clean['person_emp_length'] <= 60]\n"
        "print('Cleaned shape:', df_clean.shape)\n\n"
        "X = df_clean.drop(columns=['loan_status'])\n"
        "y = df_clean['loan_status']\n\n"
        "X_train, X_test, y_train, y_test = train_test_split(\n"
        "    X, y, test_size=0.2, random_state=42, stratify=y\n"
        ")"
    ))
    
    # 6. Imputation and Feature Engineering
    cells.append(nbf.v4.new_markdown_cell(
        "## Imputation & Feature Engineering"
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "emp_len_med = X_train['person_emp_length'].median()\n"
        "int_rate_med = X_train['loan_int_rate'].median()\n\n"
        "X_train['person_emp_length'] = X_train['person_emp_length'].fillna(emp_len_med)\n"
        "X_test['person_emp_length'] = X_test['person_emp_length'].fillna(emp_len_med)\n\n"
        "X_train['loan_int_rate'] = X_train['loan_int_rate'].fillna(int_rate_med)\n"
        "X_test['loan_int_rate'] = X_test['loan_int_rate'].fillna(int_rate_med)\n\n"
        "def eng_features(data):\n"
        "    df_feat = data.copy()\n"
        "    df_feat['interest_to_income_ratio'] = (df_feat['loan_amnt'] * df_feat['loan_int_rate'] / 100) / df_feat['person_income']\n"
        "    df_feat['emp_length_to_age_ratio'] = df_feat['person_emp_length'] / df_feat['person_age']\n"
        "    df_feat['cred_hist_to_age_ratio'] = df_feat['cb_person_cred_hist_length'] / df_feat['person_age']\n"
        "    return df_feat\n\n"
        "X_train = eng_features(X_train)\n"
        "X_test = eng_features(X_test)"
    ))
    
    # 7. Encoding and Scaling
    cells.append(nbf.v4.new_markdown_cell(
        "## Encoding & Scaling"
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "cat_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']\n"
        "num_cols = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', \n"
        "            'loan_percent_income', 'cb_person_cred_hist_length', 'interest_to_income_ratio', \n"
        "            'emp_length_to_age_ratio', 'cred_hist_to_age_ratio']\n\n"
        "X_train_encoded = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)\n"
        "X_test_encoded = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)\n\n"
        "X_train_encoded, X_test_encoded = X_train_encoded.align(X_test_encoded, join='left', axis=1, fill_value=0)\n\n"
        "dummy_cols = [c for c in X_train_encoded.columns if any(cat in c for cat in cat_cols)]\n"
        "X_train_encoded[dummy_cols] = X_train_encoded[dummy_cols].astype(int)\n"
        "X_test_encoded[dummy_cols] = X_test_encoded[dummy_cols].astype(int)\n\n"
        "scaler = StandardScaler()\n"
        "X_train_scaled = X_train_encoded.copy()\n"
        "X_test_scaled = X_test_encoded.copy()\n\n"
        "X_train_scaled[num_cols] = scaler.fit_transform(X_train_encoded[num_cols])\n"
        "X_test_scaled[num_cols] = scaler.transform(X_test_encoded[num_cols])"
    ))
    
    # 8. Modeling
    cells.append(nbf.v4.new_markdown_cell(
        "## Model Training"
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "neg_count = (y_train == 0).sum()\n"
        "pos_count = (y_train == 1).sum()\n"
        "scale_weight = neg_count / pos_count\n\n"
        "models = {}\n\n"
        "# Logistic Regression\n"
        "lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)\n"
        "lr.fit(X_train_scaled, y_train)\n"
        "models['Logistic Regression'] = lr\n\n"
        "# Decision Tree\n"
        "dt = DecisionTreeClassifier(class_weight='balanced', max_depth=8, random_state=42)\n"
        "dt.fit(X_train_scaled, y_train)\n"
        "models['Decision Tree'] = dt\n\n"
        "# Random Forest\n"
        "rf = RandomForestClassifier(class_weight='balanced', n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)\n"
        "rf.fit(X_train_scaled, y_train)\n"
        "models['Random Forest'] = rf\n\n"
        "# XGBoost\n"
        "xgb_model = xgb.XGBClassifier(scale_pos_weight=scale_weight, max_depth=6, n_estimators=200, learning_rate=0.05,\n"
        "                               random_state=42, eval_metric='logloss', n_jobs=-1)\n"
        "xgb_model.fit(X_train_scaled, y_train)\n"
        "models['XGBoost'] = xgb_model\n\n"
        "# LightGBM\n"
        "lgb_model = lgb.LGBMClassifier(scale_pos_weight=scale_weight, max_depth=6, n_estimators=200, learning_rate=0.05,\n"
        "                                random_state=42, n_jobs=-1, verbose=-1)\n"
        "lgb_model.fit(X_train_scaled, y_train)\n"
        "models['LightGBM (Default)'] = lgb_model"
    ))
    
    # 9. Optuna Tuning
    cells.append(nbf.v4.new_markdown_cell(
        "## Hyperparameter Tuning"
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "def objective(trial):\n"
        "    params = {\n"
        "        'objective': 'binary',\n"
        "        'metric': 'auc',\n"
        "        'scale_pos_weight': scale_weight,\n"
        "        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),\n"
        "        'num_leaves': trial.suggest_int('num_leaves', 20, 100),\n"
        "        'max_depth': trial.suggest_int('max_depth', 4, 10),\n"
        "        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),\n"
        "        'subsample': trial.suggest_float('subsample', 0.6, 1.0),\n"
        "        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),\n"
        "        'n_estimators': 150,\n"
        "        'random_state': 42,\n"
        "        'verbose': -1,\n"
        "        'n_jobs': -1\n"
        "    }\n\n"
        "    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)\n"
        "    scores = []\n"
        "    for train_idx, val_idx in cv.split(X_train_scaled, y_train):\n"
        "        X_tr, y_tr = X_train_scaled.iloc[train_idx], y_train.iloc[train_idx]\n"
        "        X_va, y_va = X_train_scaled.iloc[val_idx], y_train.iloc[val_idx]\n"
        "        \n"
        "        clf = lgb.LGBMClassifier(**params)\n"
        "        clf.fit(X_tr, y_tr)\n"
        "        preds = clf.predict_proba(X_va)[:, 1]\n"
        "        scores.append(roc_auc_score(y_va, preds))\n"
        "    return np.mean(scores)\n\n"
        "study = optuna.create_study(direction='maximize')\n"
        "study.optimize(objective, n_trials=10)\n"
        "print('Best params:', study.best_params)\n\n"
        "best_params = study.best_params\n"
        "best_params['scale_pos_weight'] = scale_weight\n"
        "best_params['n_estimators'] = 300\n"
        "best_params['random_state'] = 42\n"
        "best_params['verbose'] = -1\n"
        "best_params['n_jobs'] = -1\n\n"
        "best_lgb = lgb.LGBMClassifier(**best_params)\n"
        "best_lgb.fit(X_train_scaled, y_train)\n"
        "models['LightGBM (Tuned)'] = best_lgb"
    ))
    
    # 10. Evaluation & Comparison
    cells.append(nbf.v4.new_markdown_cell(
        "## Model Evaluation"
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "results_summary = []\n"
        "plt.figure(figsize=(10, 8))\n\n"
        "for name, model in models.items():\n"
        "    probs = model.predict_proba(X_test_scaled)[:, 1]\n"
        "    preds = model.predict(X_test_scaled)\n"
        "    \n"
        "    auc_roc = roc_auc_score(y_test, probs)\n"
        "    precision_vals, recall_vals, _ = precision_recall_curve(y_test, probs)\n"
        "    pr_auc = auc(recall_vals, precision_vals)\n"
        "    f1 = f1_score(y_test, preds)\n"
        "    prec = precision_score(y_test, preds)\n"
        "    rec = recall_score(y_test, preds)\n"
        "    \n"
        "    df_ks = pd.DataFrame({'y': y_test, 'prob': probs})\n"
        "    ks_stats = []\n"
        "    for thresh in np.linspace(0, 1, 100):\n"
        "        t_neg = df_ks[df_ks['y'] == 0]['prob']\n"
        "        t_pos = df_ks[df_ks['y'] == 1]['prob']\n"
        "        neg_pct = (t_neg >= thresh).mean()\n"
        "        pos_pct = (t_pos >= thresh).mean()\n"
        "        ks_stats.append(abs(pos_pct - neg_pct))\n"
        "    ks_stat = max(ks_stats)\n"
        "    \n"
        "    results_summary.append({\n"
        "        'Model': name,\n"
        "        'ROC-AUC': auc_roc,\n"
        "        'PR-AUC': pr_auc,\n"
        "        'F1-Score': f1,\n"
        "        'Precision': prec,\n"
        "        'Recall': rec,\n"
        "        'KS-Statistic': ks_stat\n"
        "    })\n"
        "    \n"
        "    fpr, tpr, _ = roc_curve(y_test, probs)\n"
        "    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_roc:.3f})')\n\n"
        "plt.plot([0, 1], [0, 1], 'k--', label='Random')\n"
        "plt.xlabel('FPR')\n"
        "plt.ylabel('TPR')\n"
        "plt.title('ROC Curves')\n"
        "plt.legend()\n"
        "plt.show()\n\n"
        "results_df = pd.DataFrame(results_summary)\n"
        "results_df"
    ))
    
    # 11. Model Interpretation
    cells.append(nbf.v4.new_markdown_cell(
        "## Feature Interpretability (SHAP)"
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "explainer = shap.TreeExplainer(best_lgb)\n"
        "shap_sample = X_test_scaled.sample(1000, random_state=42)\n"
        "shap_values = explainer(shap_sample)\n\n"
        "shap.summary_plot(shap_values, shap_sample, show=False)\n"
        "plt.title('SHAP Summary')\n"
        "plt.show()"
    ))
    
    # 12. Score Band Analysis
    cells.append(nbf.v4.new_markdown_cell(
        "## Credit Score Band Analysis"
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "test_probs = best_lgb.predict_proba(X_test_scaled)[:, 1]\n"
        "test_scores = 850 - (test_probs * 550)\n\n"
        "eval_df = pd.DataFrame({\n"
        "    'loan_status': y_test,\n"
        "    'default_prob': test_probs,\n"
        "    'credit_score': test_scores\n"
        "})\n\n"
        "def get_score_band(score):\n"
        "    if score >= 800: return 'Excellent (800-850)'\n"
        "    elif score >= 740: return 'Very Good (740-799)'\n"
        "    elif score >= 670: return 'Good (670-739)'\n"
        "    elif score >= 580: return 'Fair (580-669)'\n"
        "    else: return 'Poor (300-579)'\n\n"
        "eval_df['score_band'] = eval_df['credit_score'].apply(get_score_band)\n"
        "band_analysis = eval_df.groupby('score_band').agg(\n"
        "    total_accounts=('loan_status', 'count'),\n"
        "    defaults=('loan_status', 'sum'),\n"
        "    default_rate=('loan_status', 'mean')\n"
        ").reset_index()\n"
        "band_analysis['default_rate_pct'] = band_analysis['default_rate'] * 100\n\n"
        "plt.figure(figsize=(10, 6))\n"
        "ax = sns.barplot(data=band_analysis, x='score_band', y='default_rate_pct', palette='RdYlGn_r')\n"
        "plt.title('Default Rates across Credit Score Bands')\n"
        "plt.xlabel('Credit Score Band')\n"
        "plt.ylabel('Default Rate (%)')\n"
        "plt.xticks(rotation=15)\n"
        "for p in ax.patches:\n"
        "    ax.annotate(f'{p.get_height():.2f}%', (p.get_x() + p.get_width() / 2., p.get_height()),\n"
        "                ha='center', va='center', xytext=(0, 5), textcoords='offset points')\n"
        "plt.show()\n"
        "band_analysis"
    ))
    
    nb['cells'] = cells
    
    # Save notebook
    with open("Credit_Risk_Default_Prediction.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Notebook structure updated successfully.")

if __name__ == "__main__":
    create_notebook()
