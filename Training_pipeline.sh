#!/bin/bash

# Start MLflow Docker container
echo "Starting MLflow..."


docker-compose -f Docker/docker-compose-mlflow.yaml up -d

# Wait for MLflow to be ready
sleep 10

# Run preprocessing
echo "Running data preprocessing..."
python3 Preprocessing/Preprocessing.py

# Train models
echo "Training models..."
python3 Training/Logistic_regression.py
python3 Training/Random_forest.py
python3 Training/SVM.py
python3 Training/Xgboost.py

# Register best model
echo "Registering the best model..."
python3 Training/Reg_best_model.py

echo "Training pipeline completed successfully!"
