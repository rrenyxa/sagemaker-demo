import argparse
import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train():
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=5)

    # SageMaker paths
    # SageMaker mounts each input channel into a folder with the channel name
    if os.path.exists("/opt/ml/input/config/resourceconfig.json"):
        TRAIN_DIR = "/opt/ml/input/data/train"
        VAL_DIR = "/opt/ml/input/data/validation"
        MODEL_DIR = "/opt/ml/model"
    else:
        # Local development paths
        TRAIN_DIR = "data/train"
        VAL_DIR = "data/validation"
        MODEL_DIR = "data/model"

    args, _ = parser.parse_known_args()

    # 1. Load datasets
    # Training data
    train_df = pd.read_csv(os.path.join(TRAIN_DIR, "train.csv"), header=None)
    y_train = train_df.iloc[:, 0]
    X_train = train_df.iloc[:, 1:]

    # Validation data
    val_df = pd.read_csv(os.path.join(VAL_DIR, "validation.csv"), header=None)
    y_val = val_df.iloc[:, 0]
    X_val = val_df.iloc[:, 1:]

    print(f"Training started with n_estimators={args.n_estimators}, max_depth={args.max_depth}")
    print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}")

    # 2. Train Model
    model = RandomForestClassifier(
        n_estimators=args.n_estimators, 
        max_depth=args.max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)

    # 3. Quick Validation Check
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    
    train_acc = accuracy_score(y_train, train_preds)
    val_acc = accuracy_score(y_val, val_preds)

    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")

    # 4. Save the model
    # Note: SageMaker expects the model artifact to be in /opt/ml/model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_output_path = os.path.join(MODEL_DIR, "model.joblib")
    joblib.dump(model, model_output_path)
    
    print(f"Model saved to {model_output_path}")

if __name__ == "__main__":
    train()