import json
import pandas as pd
import numpy as np
import os
import argparse
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def preprocess():
    parser = argparse.ArgumentParser()
    # We now take two ratios for more flexibility
    parser.add_argument("--test-split-ratio", type=float, default=0.2)
    parser.add_argument("--val-split-ratio", type=float, default=0.2)
    args, _ = parser.parse_known_args()

    # --- PATH RESOLUTION ---
    if os.path.exists("/opt/ml/processing/input"):
        BASE_INPUT_DIR = "/opt/ml/processing/input"
        BASE_OUTPUT_DIR = "/opt/ml/processing"
    else:
        BASE_INPUT_DIR = "data/raw"
        BASE_OUTPUT_DIR = "data"

    input_data_path = os.path.join(BASE_INPUT_DIR, "penguins.csv")
    df = pd.read_csv(input_data_path)
    
    # 1. Basic Cleaning
    df.dropna(subset=["species", "sex"], inplace=True)

    # 2. Feature Definition
    target = "species"
    numeric_features = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    categorical_features = ["island", "sex"]

    # 3. Pipeline Definition
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    # 4. Triple Split (Train, Validation, Test)
    X = df.drop(target, axis=1)
    class_names = sorted(df[target].unique())
    class_to_id = {class_name: class_id for class_id, class_name in enumerate(class_names)}
    id_to_class = {str(class_id): class_name for class_name, class_id in class_to_id.items()}
    y_encoded = df[target].map(class_to_id).to_numpy()

    # First split: Separate Test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y_encoded, test_size=args.test_split_ratio, random_state=42
    )

    # Second split: Separate Train and Validation from the rest
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=args.val_split_ratio, random_state=42
    )

    # Transform all sets
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_val_transformed = preprocessor.transform(X_val)
    X_test_transformed = preprocessor.transform(X_test)

    # 5. Format Data (Target in the first column)
    train_dataset = np.column_stack([y_train, X_train_transformed])
    val_dataset = np.column_stack([y_val, X_val_transformed])
    test_dataset = np.column_stack([y_test, X_test_transformed])

    # 6. Save Artifacts
    paths = {
        "train": os.path.join(BASE_OUTPUT_DIR, "train"),
        "validation": os.path.join(BASE_OUTPUT_DIR, "validation"),
        "test": os.path.join(BASE_OUTPUT_DIR, "test"),
        "preprocessor": os.path.join(BASE_OUTPUT_DIR, "preprocessor"),
        "metadata": os.path.join(BASE_OUTPUT_DIR, "metadata"),
    }

    for p in paths.values():
        os.makedirs(p, exist_ok=True)

    pd.DataFrame(train_dataset).to_csv(os.path.join(paths["train"], "train.csv"), index=False, header=False)
    pd.DataFrame(val_dataset).to_csv(os.path.join(paths["validation"], "validation.csv"), index=False, header=False)
    pd.DataFrame(test_dataset).to_csv(os.path.join(paths["test"], "test.csv"), index=False, header=False)
    
    joblib.dump(preprocessor, os.path.join(paths["preprocessor"], "preprocessor.joblib"))
    with open(os.path.join(paths["metadata"], "label_mapping.json"), "w", encoding="utf-8") as file:
        json.dump(
            {"class_to_id": class_to_id, "id_to_class": id_to_class},
            file,
            indent=2,
            sort_keys=True,
        )

    print(f"Preprocessing finished.")
    print(f"Class mapping: {class_to_id}")
    print(f"Train: {train_dataset.shape}, Val: {val_dataset.shape}, Test: {test_dataset.shape}")

if __name__ == "__main__":
    preprocess()
