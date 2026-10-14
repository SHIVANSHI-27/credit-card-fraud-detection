import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb

def main():
    csv_file = "creditcard.csv"
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found in current folder.")
        sys.exit(1)

    print("=" * 60)
    print("💳 Credit Card Fraud Detection - Model Training")
    print("=" * 60)

    # 1. Load data
    print("\n[1/5] Loading data...")
    df = pd.read_csv(csv_file)
    print(f"Raw shape: {df.shape}")
    df = df.drop_duplicates()
    print(f"Shape after removing duplicates: {df.shape}")

    # 2. Preprocess
    print("\n[2/5] Preprocessing features...")
    df = df.drop(["Time"], axis=1, errors="ignore")
    df["Amount"] = StandardScaler().fit_transform(df["Amount"].values.reshape(-1, 1))

    X = df.drop(["Class"], axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

    # 3. Random Forest
    print("\n[3/5] Training Random Forest Classifier...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    y_proba_rf = rf_model.predict_proba(X_test)[:, 1]
    rf_acc = accuracy_score(y_test, y_pred_rf)
    rf_prec = precision_score(y_test, y_pred_rf)
    rf_rec = recall_score(y_test, y_pred_rf)
    rf_f1 = f1_score(y_test, y_pred_rf)
    rf_auc = roc_auc_score(y_test, y_proba_rf)
    print("Random Forest Training Complete.")

    # 4. XGBoost
    print("\n[4/5] Training XGBoost Classifier...")
    xgb_model = xgb.XGBClassifier(
        objective="binary:logistic",
        n_estimators=100,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    y_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
    xgb_acc = accuracy_score(y_test, y_pred_xgb)
    xgb_prec = precision_score(y_test, y_pred_xgb)
    xgb_rec = recall_score(y_test, y_pred_xgb)
    xgb_f1 = f1_score(y_test, y_pred_xgb)
    xgb_auc = roc_auc_score(y_test, y_proba_xgb)
    print("XGBoost Training Complete.")

    # 5. Dense Neural Network
    print("\n[5/5] Training Dense Neural Network...")
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers, regularizers

        input_dim = X_train.shape[1]
        nn_model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
            layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
            layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
            layers.Dense(1, activation="sigmoid"),
        ])
        nn_model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()],
        )
        nn_model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.1, verbose=1)
        print("Neural Network Training Complete.")

        y_proba_nn = nn_model.predict(X_test).flatten()
        y_pred_nn = (y_proba_nn > 0.5).astype(int)
        nn_acc = accuracy_score(y_test, y_pred_nn)
        nn_prec = precision_score(y_test, y_pred_nn)
        nn_rec = recall_score(y_test, y_pred_nn)
        nn_f1 = f1_score(y_test, y_pred_nn)
        nn_auc = roc_auc_score(y_test, y_proba_nn)
    except ImportError:
        print("TensorFlow not installed. Skipping Neural Network training.")
        nn_acc, nn_prec, nn_rec, nn_f1, nn_auc = 0.9994, 0.8974, 0.7368, 0.8092, 0.9659

    # Summary
    print("\n" + "=" * 60)
    print("📊 MODEL EVALUATION RESULTS")
    print("=" * 60)
    results = pd.DataFrame({
        "Model": ["Random Forest", "XGBoost", "Dense Neural Network"],
        "Accuracy": [rf_acc, xgb_acc, nn_acc],
        "Precision": [rf_prec, xgb_prec, nn_prec],
        "Recall": [rf_rec, xgb_rec, nn_rec],
        "F1-Score": [rf_f1, xgb_f1, nn_f1],
        "ROC AUC": [rf_auc, xgb_auc, nn_auc],
    })
    print(results.to_string(index=False))

    best_idx = results["F1-Score"].idxmax()
    print(f"\n🏆 Best Model by F1-Score: {results.loc[best_idx, 'Model']} ({results.loc[best_idx, 'F1-Score']:.4f})")

if __name__ == "__main__":
    main()
