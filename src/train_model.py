import os
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, accuracy_score
from data_prep import load_and_prepare_data

MODEL_SAVE_PATH = os.environ.get("MODEL_PATH", "./models/xgb_alzheimer_classifier.joblib")
RANDOM_STATE = 42

def compute_metrics(y_true, y_pred):
    """Computes F1 and Accuracy scores for XGBoost models."""
    f1 = f1_score(y_true, y_pred, average="weighted")
    accuracy = accuracy_score(y_true, y_pred)
    return {"accuracy": accuracy, "f1": f1}

def train_xgb_classifier():
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    print(f"Data split: Training samples={len(X_train)}, Testing samples={len(X_test)}")

    print("Initializing XGBoost Classifier...")
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=RANDOM_STATE,
        min_child_weight=10,
        gamma=20,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    print("Starting model training...")
    model.fit(X_train, y_train)
    print("Training completed.")

    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)

    print("\n--- Evaluation Results (XGBoost) ---")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score (Weighted): {metrics['f1']:.4f}")
    print("------------------------------------\n")

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"Model saved successfully to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_xgb_classifier()  