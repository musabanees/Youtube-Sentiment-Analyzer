import json
import mlflow
import logging
import os
import yaml
import git
from pathlib import Path
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

from youtube_sentiment.config import Tags

# Set up MLflow tracking URI
load_dotenv()
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(tracking_uri)

# Get git information
repo = git.Repo(search_parent_directories=True)
git_sha = repo.head.object.hexsha
branch = repo.active_branch.name

# logging configuration
logger = logging.getLogger('model_registration')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_registration_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def get_root_directory() -> Path:
    """Get the project root directory (three levels up from this script)."""
    file_path = Path(__file__).resolve()
    project_root = file_path.parents[3]
    return project_root

def load_params(file_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(file_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug(f"Parameters retrieved from {file_path}")
        return params
    except Exception as e:
        logger.error(f"Failed to load parameters: {e}")
        raise

def load_experiment_info(file_path: str) -> dict:
    """Load experiment information from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            info = json.load(file)
        logger.debug(f"Experiment info loaded from {file_path}")
        return info
    except Exception as e:
        logger.error(f"Failed to load experiment info: {e}")
        raise

def register_model(run_id: str, model_name: str, model_alias: str, tags: dict = None) -> str:
    """Register model in MLflow Model Registry.
    
    Args:
        run_id: The MLflow run ID containing the model
        model_name: Name to register the model under
        model_alias: Alias to assign to the model
        tags: Optional tags to attach to the model
        
    Returns:
        The version number of the registered model
    """
    try:
        logger.info(f"🔄 Registering model '{model_name}' from run {run_id}...")
        
        # Register the model in MLflow
        ## To-do: add the model name in the params.yaml file under model_registry
        ## Get the latest model by Tag (Latest), then register the model with its run ID
        registered_model = mlflow.register_model(
            model_uri=f"runs:/{run_id}/lgbm_model",  # Using lgbm_model as logged in model_evaluation.py
            name=model_name,
            tags=tags,
        )
        
        # Get the new version number
        latest_version = registered_model.version
        logger.info(f"✅ Model registered as version {latest_version}")
        
        # Set up a client to work with the model registry
        client = MlflowClient()
        
        # Set an alias for the latest model
        client.set_registered_model_alias(
            name=model_name,
            alias=model_alias,
            version=latest_version,
        )
        logger.info(f"✅ Set alias '{model_alias}' to version {latest_version}")

        # Transition the model to Production stage
        client.transition_model_version_stage(
            name=model_name,
            version=latest_version,
            stage="Production"
        )
        logger.info(f"✅ Transitioned model version {latest_version} to Production stage")
        
        return latest_version
        
    except Exception as e:
        logger.error(f"❌ Failed to register model: {e}")
        raise

def main():
    try:
        # Get project root and load parameters
        root_dir = get_root_directory()
        params = load_params(os.path.join(root_dir, 'params.yaml'))
        
        # Get model registry configuration from production environment
        registry_config = params['environments']['prd']['model_registry']
        model_name = registry_config['name']
        model_alias = registry_config['alias']
        
        # Load experiment info to get the run_id
        run_id = registry_config['run_id']
        
        # Prepare tags for the model
        tags = Tags(git_sha=git_sha, branch=branch)
        tags_dict = tags.to_dict() 

        # Register the model
        version = register_model(run_id, model_name, model_alias, tags_dict)

        # Update the params.yaml file with the new version if needed
        # This would require additional code to write back to params.yaml
        
        logger.info(f"🎉 Model registration complete! Version: {version}")
        
    except Exception as e:
        logger.error(f"Failed to complete model registration: {e}")
        raise

if __name__ == "__main__":
    main()