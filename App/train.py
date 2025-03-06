from sklearn.linear_model import LogisticRegression
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
import pickle
import os

X_train = pd.read_csv("./X_train.csv")
X_test = pd.read_csv("./X_test.csv")
y_train = pd.read_csv("./y_train.csv")
y_test = pd.read_csv("./y_test.csv")

# Create and train model
model_lg = LogisticRegression(max_iter=120, random_state=0, n_jobs=20)

# Train model
model_lg.fit(X_train, y_train)

# Make predictions
pred_lg = model_lg.predict(X_test)

# Calculate and print accuracy
lg = round(accuracy_score(y_test, pred_lg) * 100, 2)
print(f"Logistic Regression Accuracy: {lg}%")

# Log classification report to file
clf_report = classification_report(y_test, pred_lg)
print("\nClassification Report:")
print(clf_report)

# Save classification report to a text file
with open("lg_classification_report.txt", "w") as f:
    f.write(clf_report)

# Create and log confusion matrix
plt.figure(figsize=(8, 6))
cm1 = confusion_matrix(y_test, pred_lg)
sns.heatmap(cm1 / np.sum(cm1), annot=True, fmt='.2%', cmap="Reds")
plt.title("Logistic Regression Confusion Matrix")
plt.savefig("lg_confusion_matrix.png")
plt.close()

# Calculate and log additional metrics
precision = precision_score(y_test, pred_lg)
recall = recall_score(y_test, pred_lg)
f1 = f1_score(y_test, pred_lg)

# Save the model locally as a pickle file
model_dir = "./"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "model.pkl")
with open(model_path, "wb") as model_file:
    pickle.dump(model_lg, model_file)

print(f"Model saved locally at: {model_path}")
