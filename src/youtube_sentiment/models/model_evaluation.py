import numpy as np
import pandas as pd
import pickle
import logging
import yaml
import mlflow
from pathlib import Path
import os
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import seaborn as sns
import json

import git
from youtube_sentiment.config import Tags

import mlflow.sklearn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from mlflow.models import infer_signature


load_dotenv()
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")

# Get git information
repo = git.Repo(search_parent_directories=True)
git_sha = repo.head.object.hexsha
branch = repo.active_branch.name

# logging configuration
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_evaluation_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)  # Fill any NaN values
        logger.debug('Data loaded and NaNs filled from %s', file_path)
        return df
    except Exception as e:
        logger.error('Error loading data from %s: %s', file_path, e)
        raise


def load_model(model_path: str):
    """Load the trained model."""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug('Model loaded from %s', model_path)
        return model
    except Exception as e:
        logger.error('Error loading model from %s: %s', model_path, e)
        raise


def load_vectorizer(vectorizer_path: str) -> TfidfVectorizer:
    """Load the saved TF-IDF vectorizer."""
    try:
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)
        logger.debug('TF-IDF vectorizer loaded from %s', vectorizer_path)
        return vectorizer
    except Exception as e:
        logger.error('Error loading vectorizer from %s: %s', vectorizer_path, e)
        raise


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters loaded from %s', params_path)
        return params
    except Exception as e:
        logger.error('Error loading parameters from %s: %s', params_path, e)
        raise


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray):
    """Evaluate the model and log classification metrics and confusion matrix."""
    try:
        # Predict and calculate classification metrics
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        logger.debug('Model evaluation completed')

        return report, cm
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise


def log_confusion_matrix(root_dir, cm, dataset_name):
    """Log confusion matrix as an artifact."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {dataset_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    # Save confusion matrix plot as a file and log it to MLflow
    cm_file_path = os.path.join(root_dir, f'resources/confusion_matrix_{dataset_name}.png')
    plt.savefig(cm_file_path)
    mlflow.log_artifact(cm_file_path)
    plt.close()

def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save the model run ID and path to a JSON file."""
    try:
        # Create a dictionary with the info you want to save
        model_info = {
            'run_id': run_id,
            'model_path': model_path
        }
        # Save the dictionary as a JSON file
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logger.debug('Model info saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model info: %s', e)
        raise

def get_root_directory() -> str:
    """Get the root directory (three levels up from this script's location)."""
    file_path = Path(__file__).resolve()
    project_root = file_path.parents[3]
    return str(project_root)

def log_parameters(env_config: dict, active_env: str, params: dict) -> None:
    # Get model and vectorizer type from active environment
    model_type = env_config['Model']
    vectorizer_type = env_config['vectorizers']

    # Log environment info
    mlflow.log_param("environment", active_env)
    mlflow.log_param("model_type", model_type)
    mlflow.log_param("vectorizer_type", vectorizer_type)

    # Log only the relevant model parameters (for the selected model)
    if model_type in params['Model']:
        model_params = params['Model'][model_type]['params']
        for param_name, param_value in model_params.items():
            mlflow.log_param(f"model.{param_name}", param_value)

    # Log only the relevant vectorizer parameters
    if vectorizer_type in params['vectorizers']:
        vectorizer_params = params['vectorizers'][vectorizer_type]
        for param_name, param_value in vectorizer_params.items():
            # Handle list parameters like ngram_range
            if isinstance(param_value, list):
                param_value = str(tuple(param_value))
            mlflow.log_param(f"vectorizer.{param_name}", param_value)



def main():

    # Load parameters from YAML file
    root_dir = get_root_directory()
    params = load_params(os.path.join(root_dir, 'params.yaml'))

        
    # Get active environment and its configuration
    active_env = params['active_environment']
    env_config = params['environments'][active_env]
    
    # Get MLflow experiment name from the active environment
    experiment_name = env_config['monitoring']['experiment_name']
    logger.debug(f"Using experiment name from config: {experiment_name}")

    mlflow.set_experiment(experiment_name)
    mlflow.set_tracking_uri(tracking_uri)

    # Create structured tags
    tags = Tags(git_sha=git_sha, branch=branch)

    with mlflow.start_run(tags=tags.to_dict()) as run:
        try:

            # Log parameters
            log_parameters(env_config , active_env, params)

            # Load model and vectorizer
            model = load_model(os.path.join(root_dir, 'models/lgbm_model.pkl'))
            vectorizer = load_vectorizer(os.path.join(root_dir, 'models/tfidf_vectorizer.pkl'))

            # Load test data for signature inference
            test_data = load_data(os.path.join(root_dir, 'data/interim/test_processed.csv'))

            # Prepare test data
            X_test_tfidf = vectorizer.transform(test_data['clean_comment'].values)
            y_test = test_data['category'].values

            # Create a DataFrame for signature inference (using first few rows as an example)
            input_example = pd.DataFrame(X_test_tfidf.toarray()[:5], columns=vectorizer.get_feature_names_out())  # <--- Added for signature

            # Infer the signature
            signature = infer_signature(input_example, model.predict(X_test_tfidf[:5]))  # <--- Added for signature

            # Log model with signature
            model_info = mlflow.sklearn.log_model(
                model,
                "lgbm_model",
                signature=signature,  # <--- Added for signature
                input_example=input_example  # <--- Added input_example
            )

            # Save model info
            logger.debug('Model info saved to %s', model_info.model_uri)
            # model_path = "lgbm_model"
            save_model_info(
                run.info.run_id, 
                model_info.model_uri, 
                os.path.join(root_dir, 'resources', 'experiment_info.json')
            )

            # Log the vectorizer as an artifact
            mlflow.log_artifact(os.path.join(root_dir, 'models/tfidf_vectorizer.pkl'))

            eval_data = pd.DataFrame(X_test_tfidf.toarray(), columns=vectorizer.get_feature_names_out())
            eval_data['target'] = y_test  # Add target column
            
            # Run MLflow's built-in evaluation
            result = mlflow.models.evaluate(
                model=model_info.model_uri,  # URI to the logged model
                data=eval_data,              # Evaluation dataset with features and target
                targets="target",            # Target column name
                model_type="classifier",
                evaluators=["default"],
            )
            mlflow.log_metrics({
                "eval_accuracy": result.metrics.get("accuracy_score", 0),
                "eval_f1_weighted": result.metrics.get("f1_score_weighted", 0),
                "eval_precision_weighted": result.metrics.get("precision_score_weighted", 0),
                "eval_recall_weighted": result.metrics.get("recall_score_weighted", 0)
            })

            # Log the evaluation results summary
            logger.info(f"MLflow evaluation completed with metrics: {result.metrics}")
            print(f"EVALUATION COMPLETE - Metrics: {result.metrics}")
    

        except Exception as e:
            logger.error(f"Failed to complete model evaluation: {e}")
            print(f"Error: {e}")

if __name__ == '__main__':
    main()