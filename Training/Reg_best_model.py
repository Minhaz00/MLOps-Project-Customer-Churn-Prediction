from mlflow.tracking import MlflowClient
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Customer Churn Prediction")

def register_best_model(experiment_name="Customer Churn Prediction"):
    """
    Registers the best-performing model from the MLflow experiment
    to the model registry and transitions it to the 'Production' stage.
    """
    client = MlflowClient()
    
    # Retrieve experiment details
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment '{experiment_name}' not found.")
    
    # Identify the best run based on a key metric (e.g., accuracy)
    best_run = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.accuracy DESC"]
    )[0]
    
    # Register the model from the best run
    model_uri = f"runs:/{best_run.info.run_id}/model"
    model_name = "customer_churn_prediction_model"
    model_version = mlflow.register_model(model_uri, model_name)
    
    # Transition the model to the 'Production' stage
    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage="Production"
    )
    
    print(f"Model {model_name} version {model_version.version} is now in 'Production' stage.")
    return model_version

# Call the function after model training and comparison
best_model_version = register_best_model("Customer Churn Prediction")
print(f"Registered model version: {best_model_version.version}")