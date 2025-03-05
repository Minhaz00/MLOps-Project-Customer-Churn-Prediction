from sklearn.svm import SVC
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

X_train = pd.read_csv("../Data/X_train.csv")
X_test = pd.read_csv("../Data/X_test.csv")
y_train = pd.read_csv("../Data/y_train.csv")
y_test = pd.read_csv("../Data/y_test.csv")

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Customer Churn Prediction")

with mlflow.start_run(run_name="svm_classifier"):
    # Create and train model
    model_svm = SVC(kernel='rbf', random_state=42)
    
    # Log parameters
    mlflow.log_params({
        "kernel": "rbf",
        "random_state": 42
    })
    
    # Train model
    model_svm.fit(X_train, y_train)
    
    # Make predictions
    pred_svm = model_svm.predict(X_test)
    
    # Calculate and log accuracy
    sv = round(accuracy_score(y_test, pred_svm)*100, 2)
    mlflow.log_metric("accuracy", sv)
    
    # Log classification report
    clf_report = classification_report(y_test, pred_svm)
    with open("svm_classification_report.txt", "w") as f:
        f.write(clf_report)
    mlflow.log_artifact("svm_classification_report.txt")
    
    # Create and log confusion matrix
    plt.figure(figsize=(8, 6))
    cm6 = confusion_matrix(y_test, pred_svm)
    sns.heatmap(cm6/np.sum(cm6), annot=True, fmt='0.2%', cmap="Reds")
    plt.title("SVM Classifier Confusion Matrix")
    plt.savefig("svm_confusion_matrix.png")
    plt.close()
    mlflow.log_artifact("svm_confusion_matrix.png")
    
    # Log additional metrics
    mlflow.log_metrics({
        "precision": precision_score(y_test, pred_svm),
        "recall": recall_score(y_test, pred_svm),
        "f1": f1_score(y_test, pred_svm)
    })
    
    # Log the model
    signature = infer_signature(X_train, pred_svm)
    mlflow.sklearn.log_model(model_svm, "svm_model", signature=signature)

print(f"SVM Classifier Accuracy: {sv}%")
print("\nClassification Report:")
print(classification_report(y_test, pred_svm))