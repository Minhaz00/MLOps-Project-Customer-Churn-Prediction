from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import mlflow
from mlflow.models import infer_signature
import pickle
import os

X_train = pd.read_csv("/root/code/MLOps-Project-Customer-Churn-Prediction/Data/X_train.csv")
X_test = pd.read_csv("/root/code/MLOps-Project-Customer-Churn-Prediction/Data/X_test.csv")
y_train = pd.read_csv("/root/code/MLOps-Project-Customer-Churn-Prediction/Data/y_train.csv")
y_test = pd.read_csv("/root/code/MLOps-Project-Customer-Churn-Prediction/Data/y_test.csv")

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Customer Churn Prediction")

with mlflow.start_run(run_name="random_forest"):
    # Create and train model
    model_rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=0.16, random_state=42)
    
    # Log parameters
    mlflow.log_params({
        "n_estimators": 300,
        "min_samples_leaf": 0.16,
        "random_state": 42
    })
    
    # Train model
    model_rf.fit(X_train, y_train)
    
    # Make predictions
    pred_rf = model_rf.predict(X_test)
    
    # Calculate and log accuracy
    rf = round(accuracy_score(y_test, pred_rf) * 100, 2)
    mlflow.log_metric("accuracy", rf)
    
    # Log classification report
    clf_report = classification_report(y_test, pred_rf)
    with open("rf_classification_report.txt", "w") as f:
        f.write(clf_report)
    mlflow.log_artifact("rf_classification_report.txt")
    
    # Create and log confusion matrix
    plt.figure(figsize=(8, 6))
    cm3 = confusion_matrix(y_test, pred_rf)
    sns.heatmap(cm3 / np.sum(cm3), annot=True, fmt='.2%', cmap="Reds")
    plt.title("Random Forest Confusion Matrix")
    plt.savefig("rf_confusion_matrix.png")
    plt.close()
    mlflow.log_artifact("rf_confusion_matrix.png")
    
    # Calculate and log additional metrics
    precision = precision_score(y_test, pred_rf, average="weighted")
    recall = recall_score(y_test, pred_rf, average="weighted")
    f1 = f1_score(y_test, pred_rf, average="weighted")
    mlflow.log_metrics({
        "precision": precision,
        "recall": recall,
        "f1": f1
    })
    
    # Log feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model_rf.feature_importances_
    }).sort_values('importance', ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
    plt.title("Top 10 Feature Importance - Random Forest")
    plt.savefig("rf_feature_importance.png")
    plt.close()
    mlflow.log_artifact("rf_feature_importance.png")
    
    # Log the model
    signature = infer_signature(X_train, pred_rf)
    mlflow.sklearn.log_model(model_rf, "random_forest_model", signature=signature)

    # Save the model locally as a pickle file
    model_dir = "Model"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "random_forest_model.pkl")
    with open(model_path, "wb") as model_file:
        pickle.dump(model_rf, model_file)

    print(f"Model saved locally at: {model_path}")

print(f"Random Forest Metrics: Accuracy={rf}%, Precision={precision}, Recall={recall}, F1={f1}")