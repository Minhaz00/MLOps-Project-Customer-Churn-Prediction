import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Customer Churn Prediction")

df= pd.read_csv("../Data/Telco-Customer-Churn.csv")

# Drop customerID column
df = df.drop('customerID', axis=1)

# Convert TotalCharges to numeric, handling errors
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors="coerce")

# Drop rows with missing TotalCharges


# Reset index
df.drop(df[df['TotalCharges'].isnull()]
.index, inplace=True)
df.reset_index(drop=True, inplace=True)
df.replace('No internet service', 'No', inplace=True)
df.replace('No phone service', 'No', inplace=True)

# Display unique values in categorical columns
for i in df.columns:
    if df[i].dtypes=="object":
        print(f'{i} : {df[i].unique()}')
        print("****************************************************")

# Convert gender to numeric
df['gender'].replace({'Female':1,'Male':0}, inplace=True)

# One-hot encoding for multi-category variables
# Handle variables with more than 2 categories
more_than_2 = ['InternetService' ,'Contract' ,'PaymentMethod']
df = pd.get_dummies(data=df, columns=more_than_2)

# Feature scaling for numerical columns
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# Scale continuous variables
large_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
df[large_cols] = scaler.fit_transform(df[large_cols])

# Convert binary categories to numeric
two_cate = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
            'TechSupport', 'StreamingTV', 'StreamingMovies', 
            'PaperlessBilling', 'Churn']
for i in two_cate:
    df[i].replace({"No":0, "Yes":1}, inplace=True)

print("Preparing features and target...")
X = df.drop('Churn', axis=1)
y = df['Churn']

print("\nFeature set shape:", X.shape)
print("Target shape:", y.shape)

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.33, 
    random_state=42
)

print("\nTraining set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# Save the preprocessed data
X_train.to_csv("../Data/X_train.csv", index=False)
X_test.to_csv("../Data/X_test.csv", index=False)
y_train.to_csv("../Data/y_train.csv", index=False)
y_test.to_csv("../Data/y_test.csv", index=False)