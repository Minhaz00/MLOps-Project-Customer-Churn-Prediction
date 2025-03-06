from xgboost import XGBClassifier
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

X_train = pd.read_csv("/root/code/MLOps-Project-Customer-Churn-Prediction/Data/X_train.csv")
X_test = pd.read_csv("/root/code/MLOps-Project-Customer-Churn-Prediction/Data/X_test.csv")
y_train = pd.read_csv("/root/code/MLOps-Project-Customer-Churn-Prediction/Data/y_train.csv")
y_test = pd.read_csv("/root/code/MLOps-Project-Customer-Churn-Prediction/Data/y_test.csv")

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Customer Churn Prediction")

with mlflow.start_run(run_name="xgboost"):
    # Create and train model
    model_xgb = XGBClassifier(max_depth=8, n_estimators=125, random_state=0, 
                              learning_rate=0.03, n_jobs=5)
    
    # Log parameters
    mlflow.log_params({
        "max_depth": 8,
        "n_estimators": 125,
        "learning_rate": 0.03,
        "random_state": 0,
        "n_jobs": 5
    })
    
    # Train model
    model_xgb.fit(X_train, y_train)
    
    # Make predictions
    pred_xgb = model_xgb.predict(X_test)
    
    # Calculate and log accuracy
    xgb = round(accuracy_score(y_test, pred_xgb) * 100, 2)
    mlflow.log_metric("accuracy", xgb)
    
    # Log classification report
    clf_report = classification_report(y_test, pred_xgb)
    with open("xgb_classification_report.txt", "w") as f:
        f.write(clf_report)
    mlflow.log_artifact("xgb_classification_report.txt")
    
    # Log confusion matrix
    plt.figure(figsize=(8, 6))
    cm4 = confusion_matrix(y_test, pred_xgb)
    sns.heatmap(cm4 / np.sum(cm4), annot=True, fmt='.2%', cmap="Reds")
    plt.title("XGBoost Confusion Matrix")
    plt.savefig("xgb_confusion_matrix.png")
    plt.close()
    mlflow.log_artifact("xgb_confusion_matrix.png")
    
    # Log model
    signature = infer_signature(X_train, pred_xgb)
    mlflow.sklearn.log_model(model_xgb, "xgboost_model", signature=signature)

    # Calculate precision, recall, and F1 score
    precision = round(precision_score(y_test, pred_xgb) * 100, 2)
    recall = round(recall_score(y_test, pred_xgb) * 100, 2)
    f1 = round(f1_score(y_test, pred_xgb) * 100, 2)

    # Log the additional metrics
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Print the metrics
    print(f"XGBoost Metrics: Accuracy={xgb}%, Precision={precision}%, Recall={recall}%, F1={f1}%")
    print(f"XGBoost Metrics: Accuracy={xgb}%, Precision={precision}, Recall={recall}, F1={f1}")