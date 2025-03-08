import boto3
from botocore.exceptions import NoCredentialsError

# AWS S3 bucket details
BUCKET_NAME = 'poridhi-mlflow-models-123-d862805'
MODEL_FILE_PATH = '/root/code/MLOps-Project-Customer-Churn-Prediction/Model/logistic_regression_model.pkl'  # Local path to your .pkl file
S3_OBJECT_NAME = 'model.pkl'  # Path in S3 bucket

# Initialize the S3 client
s3 = boto3.client('s3')

# Upload the model to S3
try:
    s3.upload_file(MODEL_FILE_PATH, BUCKET_NAME, S3_OBJECT_NAME)
    print(f"Model successfully uploaded to s3://{BUCKET_NAME}/{S3_OBJECT_NAME}")
except FileNotFoundError:
    print("The specified model file was not found.")
except NoCredentialsError:
    print("AWS credentials not available.")
except Exception as e:
    print(f"An error occurred: {e}")
