# src/utils.py
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

from visualization import plot_feature_importances, plot_actual_vs_predicted
from preprocessing import load_and_clean_data
from features import select_features


def train_and_save_best_models():
    print("\n🚀 Starting AQI Model Training Pipeline...\n")

    # -------------------------
    # 1. Load & clean data
    # -------------------------
    # data_path is relative to this file
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "aqi_dataset.csv"))
    df, label_encoder = load_and_clean_data(data_path)  # returns (df, le)
    print()

    # -------------------------
    # 2. Feature selection
    # -------------------------
    X, y_reg, y_clf = select_features(df)

    # -------------------------
    # 3. Train Regression Models
    # -------------------------
    print("\n--- Training Regression Models ---\n")
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)

    # instantiate models
    rf_reg = RandomForestRegressor(random_state=42)
    svr_reg = SVR()
    xgb_reg = XGBRegressor(random_state=42, n_estimators=200)

    # fit
    rf_reg.fit(X_train_r, y_train_r)
    svr_reg.fit(X_train_r, y_train_r)
    xgb_reg.fit(X_train_r, y_train_r)

    # evaluate — compute RMSE as sqrt(MSE)
    print("\n--- Regression Model Evaluation ---\n")
    reg_results = {}
    for name, model in [("RandomForest", rf_reg), ("SVR", svr_reg), ("XGBoost", xgb_reg)]:
        y_pred = model.predict(X_test_r)
        r2 = r2_score(y_test_r, y_pred)
        mse = mean_squared_error(y_test_r, y_pred)         
        rmse = float(np.sqrt(mse))
        reg_results[name] = {"r2": r2, "rmse": rmse}
        print(f"{name} Results:\nR² Score: {r2:.4f}\nRMSE: {rmse:.4f}\n")

    # pick best by R²
    best_reg_name = max(reg_results.keys(), key=lambda k: reg_results[k]["r2"])
    best_reg = {"RandomForest": rf_reg, "SVR": svr_reg, "XGBoost": xgb_reg}[best_reg_name]
    print(f"✅ Best Regression Model: {best_reg_name} (R²={reg_results[best_reg_name]['r2']:.4f})\n")

    try:
        y_pred_best = best_reg.predict(X_test_r)
        plot_actual_vs_predicted(y_test_r, y_pred_best)
        plot_feature_importances(best_reg, X.columns, title=f"{best_reg_name} Regressor - Feature Importance")
    except Exception as e:
        print("⚠️ Plotting skipped (non-interactive environment).", e)

    # -------------------------
    # 4. Train Classification Models
    # -------------------------
    print("\n--- Training Classification Models ---\n")
    if y_clf is None:
        if "AQI_Bucket" in df.columns:
            le = LabelEncoder()
            y_clf = le.fit_transform(df["AQI_Bucket"])
        else:
            raise ValueError("No classification target found ('AQI_Bucket' or encoded)")

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_clf, test_size=0.2, random_state=42)

    # SMOTE balancing
    print("📘 Applying SMOTE for class balancing...")
    smote = SMOTE(random_state=42)
    X_train_c_bal, y_train_c_bal = smote.fit_resample(X_train_c, y_train_c)

    rf_clf = RandomForestClassifier(random_state=42)
    xgb_clf = XGBClassifier(random_state=42, n_estimators=200, use_label_encoder=False, eval_metric="mlogloss")

    rf_clf.fit(X_train_c_bal, y_train_c_bal)
    xgb_clf.fit(X_train_c_bal, y_train_c_bal)

    print("\n================ CLASSIFICATION RESULTS ================\n")
    clf_results = {}
    for name, model in [("RandomForestClassifier", rf_clf), ("XGBClassifier", xgb_clf)]:
        y_pred_c = model.predict(X_test_c)
        acc = accuracy_score(y_test_c, y_pred_c)
        clf_results[name] = {"accuracy": acc}
        print(f"🔹 {name} Accuracy: {acc:.4f}\n")
        try:
            if label_encoder is not None:
                y_true_labels = label_encoder.inverse_transform(y_test_c)
                y_pred_labels = label_encoder.inverse_transform(y_pred_c)
                print(classification_report(y_true_labels, y_pred_labels))
            else:
                print(classification_report(y_test_c, y_pred_c))
        except Exception:
            print(classification_report(y_test_c, y_pred_c))

    # pick best classifier by accuracy
    best_clf_name = max(clf_results.keys(), key=lambda k: clf_results[k]["accuracy"])
    best_clf = {"RandomForestClassifier": rf_clf, "XGBClassifier": xgb_clf}[best_clf_name]
    print(f"✅ Best Classification Model: {best_clf_name} (Accuracy={clf_results[best_clf_name]['accuracy']:.4f})\n")

    # -------------------------
    # 5. Save best models + label encoder
    # -------------------------
    save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "webapp", "saved_models"))
    os.makedirs(save_dir, exist_ok=True)

    joblib.dump(best_reg, os.path.join(save_dir, "regressor.pkl"))
    joblib.dump(best_clf, os.path.join(save_dir, "classifier.pkl"))
    if isinstance(label_encoder, LabelEncoder):
        joblib.dump(label_encoder, os.path.join(save_dir, "label_encoder.pkl"))
    else:
        try:
            le_to_save = LabelEncoder()
            le_to_save.fit(df["AQI_Bucket"])
            joblib.dump(le_to_save, os.path.join(save_dir, "label_encoder.pkl"))
        except Exception:
            pass

    print("\n💾 Models saved to:", save_dir)
    print("🎯 Pipeline complete — regression & classification models trained and saved.\n")
    return best_reg, best_clf


if __name__ == "__main__":
    train_and_save_best_models()
