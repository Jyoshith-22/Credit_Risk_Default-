import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, f1_score,
    precision_score, recall_score, roc_curve, classification_report, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
import optuna
import shap

# Set style for plots
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

def main():
    print("--- Phase 1: Data Loading & Preprocessing ---")
    
    # 1. Load data
    if not os.path.exists("credit_risk_dataset.csv"):
        raise FileNotFoundError("credit_risk_dataset.csv not found in current directory.")
        
    df = pd.read_csv("credit_risk_dataset.csv")
    print(f"Loaded dataset with shape: {df.shape}")
    
    # 2. Outlier Removal
    # From initial exploration, person_age max is 144, which is unrealistic.
    # Similarly, person_emp_length can have outliers.
    initial_len = len(df)
    df = df[df['person_age'] <= 100]
    df = df[df['person_emp_length'] <= 60]  # Standard assumption
    print(f"Removed {initial_len - len(df)} outliers. New shape: {df.shape}")
    
    # 3. Train-Test Split (before imputation/scaling to prevent data leakage)
    X = df.drop(columns=['loan_status'])
    y = df['loan_status']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train set size: {X_train.shape}, Test set size: {X_test.shape}")
    
    # Save test labels for evaluation checks
    test_data = X_test.copy()
    test_data['loan_status'] = y_test
    
    # 4. Impute Missing Values (using training set medians)
    emp_length_median = X_train['person_emp_length'].median()
    int_rate_median = X_train['loan_int_rate'].median()
    
    X_train['person_emp_length'] = X_train['person_emp_length'].fillna(emp_length_median)
    X_test['person_emp_length'] = X_test['person_emp_length'].fillna(emp_length_median)
    
    X_train['loan_int_rate'] = X_train['loan_int_rate'].fillna(int_rate_median)
    X_test['loan_int_rate'] = X_test['loan_int_rate'].fillna(int_rate_median)
    
    # 5. Feature Engineering
    def add_features(data):
        df_feat = data.copy()
        # Interest payment relative to income
        df_feat['interest_to_income_ratio'] = (df_feat['loan_amnt'] * df_feat['loan_int_rate'] / 100) / df_feat['person_income']
        # Employment length relative to age
        df_feat['emp_length_to_age_ratio'] = df_feat['person_emp_length'] / df_feat['person_age']
        # Credit history length relative to age
        df_feat['cred_hist_to_age_ratio'] = df_feat['cb_person_cred_hist_length'] / df_feat['person_age']
        return df_feat

    X_train = add_features(X_train)
    X_test = add_features(X_test)
    print("Added feature engineered columns: 'interest_to_income_ratio', 'emp_length_to_age_ratio', 'cred_hist_to_age_ratio'")
    
    # 6. Categorical Encoding (One-Hot Encoding)
    # We want to identify the categorical columns
    cat_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
    
    # We use pd.get_dummies and align train/test to ensure matching columns
    X_train_encoded = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
    X_test_encoded = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)
    
    # Align to ensure identical column ordering and drop mismatched columns
    X_train_encoded, X_test_encoded = X_train_encoded.align(X_test_encoded, join='left', axis=1, fill_value=0)
    
    # Convert dummy columns to integers (0 or 1)
    dummy_cols = [c for c in X_train_encoded.columns if any(cat in c for cat in cat_cols)]
    X_train_encoded[dummy_cols] = X_train_encoded[dummy_cols].astype(int)
    X_test_encoded[dummy_cols] = X_test_encoded[dummy_cols].astype(int)
    
    # 7. Scaling Numerical Columns
    num_cols = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 
                'loan_percent_income', 'cb_person_cred_hist_length', 'interest_to_income_ratio', 
                'emp_length_to_age_ratio', 'cred_hist_to_age_ratio']
    
    scaler = StandardScaler()
    X_train_scaled = X_train_encoded.copy()
    X_test_scaled = X_test_encoded.copy()
    
    X_train_scaled[num_cols] = scaler.fit_transform(X_train_encoded[num_cols])
    X_test_scaled[num_cols] = scaler.transform(X_test_encoded[num_cols])
    
    print("Preprocessed & Scaled numerical variables.")
    
    # Save preprocessing objects for deployment
    preprocessing_artifacts = {
        'scaler': scaler,
        'num_cols': num_cols,
        'cat_cols': cat_cols,
        'encoded_cols_order': X_train_encoded.columns.tolist(),
        'emp_length_median': emp_length_median,
        'int_rate_median': int_rate_median
    }
    joblib.dump(preprocessing_artifacts, "preprocessing_artifacts.pkl")
    print("Saved preprocessing artifacts to preprocessing_artifacts.pkl")
    
    print("\n--- Phase 2: Model Training ---")
    
    # Calculate scale_pos_weight for XGBoost/LightGBM
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_weight = neg_count / pos_count
    print(f"Target imbalance: {neg_count} non-defaults, {pos_count} defaults. Scale weight: {scale_weight:.2f}")
    
    models = {}
    
    # 1. Logistic Regression (Baseline)
    print("Training Logistic Regression...")
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    models['Logistic Regression'] = lr
    
    # 2. Decision Tree (Baseline)
    print("Training Decision Tree...")
    dt = DecisionTreeClassifier(class_weight='balanced', max_depth=8, random_state=42)
    dt.fit(X_train_scaled, y_train)
    models['Decision Tree'] = dt
    
    # 3. Random Forest (Advanced)
    print("Training Random Forest...")
    rf = RandomForestClassifier(class_weight='balanced', n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    models['Random Forest'] = rf
    
    # 4. XGBoost (Advanced)
    print("Training XGBoost...")
    xgb_model = xgb.XGBClassifier(scale_pos_weight=scale_weight, max_depth=6, n_estimators=200, learning_rate=0.05, 
                                   random_state=42, eval_metric='logloss', n_jobs=-1)
    xgb_model.fit(X_train_scaled, y_train)
    models['XGBoost'] = xgb_model
    
    # 5. LightGBM (Advanced - Default)
    print("Training LightGBM (Default)...")
    lgb_model = lgb.LGBMClassifier(scale_pos_weight=scale_weight, max_depth=6, n_estimators=200, learning_rate=0.05,
                                    random_state=42, n_jobs=-1, verbose=-1)
    lgb_model.fit(X_train_scaled, y_train)
    models['LightGBM (Default)'] = lgb_model

    # 6. Hyperparameter Tuning using Optuna for LightGBM
    print("Starting Optuna Tuning for LightGBM...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    def objective(trial):
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'scale_pos_weight': scale_weight,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'n_estimators': 150,
            'random_state': 42,
            'verbose': -1,
            'n_jobs': -1
        }
        
        # Stratified K-Fold
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        auc_scores = []
        
        for train_idx, val_idx in cv.split(X_train_scaled, y_train):
            X_tr, y_tr = X_train_scaled.iloc[train_idx], y_train.iloc[train_idx]
            X_va, y_va = X_train_scaled.iloc[val_idx], y_train.iloc[val_idx]
            
            clf = lgb.LGBMClassifier(**params)
            clf.fit(X_tr, y_tr)
            preds = clf.predict_proba(X_va)[:, 1]
            auc_scores.append(roc_auc_score(y_va, preds))
            
        return np.mean(auc_scores)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=15)
    print(f"Optuna completed. Best CV ROC-AUC: {study.best_value:.4f}")
    print(f"Best Params: {study.best_params}")
    
    # Train final model with best parameters
    best_params = study.best_params
    best_params['scale_pos_weight'] = scale_weight
    best_params['n_estimators'] = 300
    best_params['random_state'] = 42
    best_params['verbose'] = -1
    best_params['n_jobs'] = -1
    
    print("Training Best LightGBM Model...")
    best_lgb = lgb.LGBMClassifier(**best_params)
    best_lgb.fit(X_train_scaled, y_train)
    models['LightGBM (Tuned)'] = best_lgb
    
    # Save the best model
    joblib.dump(best_lgb, "best_credit_risk_model.pkl")
    print("Saved best model to best_credit_risk_model.pkl")
    
    print("\n--- Phase 3: Model Evaluation ---")
    
    # We will compute metrics for all models
    results_summary = []
    
    plt.figure(figsize=(10, 8))
    for name, model in models.items():
        # Predict probability
        probs = model.predict_proba(X_test_scaled)[:, 1]
        preds = model.predict(X_test_scaled)
        
        # Metrics
        auc_roc = roc_auc_score(y_test, probs)
        
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, probs)
        pr_auc = auc(recall_vals, precision_vals)
        
        f1 = f1_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        
        # KS Statistic
        # KS is the max separation between the CDF of defaults and non-defaults
        df_ks = pd.DataFrame({'y': y_test, 'prob': probs})
        ks_stats = []
        for thresh in np.linspace(0, 1, 100):
            t_neg = df_ks[df_ks['y'] == 0]['prob']
            t_pos = df_ks[df_ks['y'] == 1]['prob']
            neg_pct = (t_neg >= thresh).mean()
            pos_pct = (t_pos >= thresh).mean()
            ks_stats.append(abs(pos_pct - neg_pct))
        ks_stat = max(ks_stats)
        
        results_summary.append({
            'Model': name,
            'ROC-AUC': auc_roc,
            'PR-AUC': pr_auc,
            'F1-Score': f1,
            'Precision': prec,
            'Recall': rec,
            'KS-Statistic': ks_stat
        })
        
        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_test, probs)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_roc:.3f})")
        
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Model Comparison')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('model_roc_curves.png', dpi=300)
    plt.close()
    
    # Plot PR curves
    plt.figure(figsize=(10, 8))
    for name, model in models.items():
        probs = model.predict_proba(X_test_scaled)[:, 1]
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, probs)
        pr_auc = auc(recall_vals, precision_vals)
        plt.plot(recall_vals, precision_vals, label=f"{name} (PR-AUC = {pr_auc:.3f})")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves - Model Comparison')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig('model_pr_curves.png', dpi=300)
    plt.close()
    
    # Results DataFrame
    results_df = pd.DataFrame(results_summary)
    results_df.to_csv("model_evaluation_metrics.csv", index=False)
    print("\nModel Performance Metrics:")
    print(results_df.to_string(index=False))
    
    # 8. Feature Importance (Tuned LightGBM)
    print("\nGenerating Feature Importance Plot...")
    importance_df = pd.DataFrame({
        'Feature': X_train_scaled.columns,
        'Importance': best_lgb.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(data=importance_df.head(15), x='Importance', y='Feature', palette='viridis')
    plt.title('Top 15 Feature Importances (LightGBM Tuned)')
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300)
    plt.close()
    
    # 9. Model Interpretation with SHAP (Tuned LightGBM)
    print("\nComputing SHAP values (takes a few moments)...")
    # Using tree explainer for lightgbm
    explainer = shap.TreeExplainer(best_lgb)
    # Explain a sample of 1000 test points for speed
    shap_sample = X_test_scaled.sample(min(1000, len(X_test_scaled)), random_state=42)
    shap_values = explainer(shap_sample)
    
    # Plot SHAP summary and save
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, shap_sample, show=False)
    plt.title("SHAP Summary Plot - Tuned LightGBM", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig('shap_summary.png', dpi=300)
    plt.close()
    
    # 10. Credit Score Band / Risk Band Analysis (Bonus Task)
    print("\nPerforming Credit Risk Band Analysis...")
    # Calculate probabilities of default
    test_probs = best_lgb.predict_proba(X_test_scaled)[:, 1]
    
    # Create Risk Scores (custom mapping: 300 to 850 score, similar to FICO)
    # High default probability -> Low credit score
    test_scores = 850 - (test_probs * 550)
    
    eval_df = pd.DataFrame({
        'loan_status': y_test,
        'default_prob': test_probs,
        'credit_score': test_scores
    })
    
    # Bin credit scores into FICO bands
    def get_score_band(score):
        if score >= 800: return '1. Excellent (800-850)'
        elif score >= 740: return '2. Very Good (740-799)'
        elif score >= 670: return '3. Good (670-739)'
        elif score >= 580: return '4. Fair (580-669)'
        else: return '5. Poor (300-579)'
        
    eval_df['score_band'] = eval_df['credit_score'].apply(get_score_band)
    
    band_analysis = eval_df.groupby('score_band').agg(
        total_accounts=('loan_status', 'count'),
        defaults=('loan_status', 'sum'),
        default_rate=('loan_status', 'mean')
    ).reset_index()
    
    band_analysis['default_rate_pct'] = band_analysis['default_rate'] * 100
    band_analysis.to_csv("credit_score_band_analysis.csv", index=False)
    print("\nCredit Score Band Analysis:")
    print(band_analysis.to_string(index=False))
    
    # Plot default rate by score bands
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=band_analysis, x='score_band', y='default_rate_pct', palette='RdYlGn_r')
    plt.title('Default Rates across Credit Score Bands')
    plt.xlabel('Credit Score Band (FICO Mapping)')
    plt.ylabel('Observed Default Rate (%)')
    plt.xticks(rotation=15)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 5), textcoords='offset points')
    plt.tight_layout()
    plt.savefig('risk_segmentation.png', dpi=300)
    plt.close()
    
    print("\n--- Pipeline execution finished successfully! All assets saved. ---")

if __name__ == "__main__":
    main()
