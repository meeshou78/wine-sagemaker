import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df):
    X = df.drop('quality', axis=1)
    y = df['quality']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.4f}")

def save_model(model, model_dir):
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "wine_model.pkl"))

if __name__ == "__main__":
    # Point to the SageMaker input channel
    input_path = "/opt/ml/input/data/train/wine_quality_dataset.csv"
    output_path = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

    df = load_data(input_path)
    X_train, X_test, y_train, y_test = preprocess_data(df)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model, output_path)

