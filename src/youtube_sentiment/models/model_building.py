import numpy as np
import pandas as pd
import os
import pickle
import yaml
import logging
from pathlib import Path

import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer

# logging configuration
logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_building_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)  # Fill any NaN values
        logger.debug('Data loaded and NaNs filled from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def apply_tfidf(train_data: pd.DataFrame, **vectorizer_params) -> tuple:
    """Apply TF-IDF with configurable parameters to the data.
    
    Args:
        train_data: DataFrame containing the text data
        **vectorizer_params: All parameters to pass to TfidfVectorizer
    """
    try:

        # Safety conversion right before instantiating the vectorizer
        if 'ngram_range' in vectorizer_params and not isinstance(vectorizer_params['ngram_range'], tuple):
            vectorizer_params['ngram_range'] = tuple(vectorizer_params['ngram_range'])
            logger.debug(f"Converted ngram_range to tuple: {vectorizer_params['ngram_range']}")
        
        # Instantiate vectorizer with all parameters from config
        vectorizer = TfidfVectorizer(**vectorizer_params)

        X_train = train_data['clean_comment'].values
        y_train = train_data['category'].values

        # Perform TF-IDF transformation
        X_train_tfidf = vectorizer.fit_transform(X_train)

        logger.debug(f"TF-IDF transformation complete. Shape: {X_train_tfidf.shape}")
        logger.debug(f"Parameters used: {vectorizer_params}")

        # Create models directory if it doesn't exist
        os.makedirs(os.path.join(get_root_directory(), 'models'), exist_ok=True)
        
        # Save the vectorizer in the models directory
        with open(os.path.join(get_root_directory(), 'models/tfidf_vectorizer.pkl'), 'wb') as f:
            pickle.dump(vectorizer, f)

        logger.debug('TF-IDF vectorizer saved successfully')
        return X_train_tfidf, y_train
    except Exception as e:
        logger.error('Error during TF-IDF transformation: %s', e)
        raise

def save_model(model, file_path: str) -> None:
    """Save the trained model to a file."""
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug('Model saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model: %s', e)
        raise

def get_root_directory() -> str:
    """Get the root directory (three levels up from this script's location)."""
    file_path = Path(__file__).resolve()
    project_root = file_path.parents[3]
    return str(project_root)


def main():
    try:
        # Get root directory and resolve the path for params.yaml
        root_dir = get_root_directory()

        # Load parameters from the root directory
        params = load_params(os.path.join(root_dir, 'params.yaml'))
        
        # Get active environment
        active_env = params['active_environment']
        logger.info(f"Using active environment: {active_env}")
        
        # Get environment-specific configuration
        env_config = params['environments'][active_env]
        
        # Get selected model type from active environment
        selected_model = env_config['Model']
        logger.info(f"Selected model: {selected_model}")
        
        # Get selected vectorizer type from active environment
        vectorizer_type = env_config['vectorizers']
        logger.info(f"Selected vectorizer: {vectorizer_type}")
        
        # Load the preprocessed training data from the interim directory
        train_data = load_data(os.path.join(root_dir, 'data/interim/train_processed.csv'))
        
        # Get vectorizer parameters
        if vectorizer_type in params['vectorizers']:
            vectorizer_params = params['vectorizers'][vectorizer_type]
            logger.debug(f"Using {vectorizer_type} parameters: {vectorizer_params}")
        else:
            logger.warning(f"No config found for {vectorizer_type}, using defaults")
            vectorizer_params = {'max_features': 1000, 'ngram_range': (1, 3)}
        
        # Apply vectorizer to data
        X_train_tfidf, y_train = apply_tfidf(train_data, **vectorizer_params)

        # Get LightGBM model parameters directly
        model_params = params['Model'][selected_model]['params']
        logger.info(f"Training LightGBM with parameters: {model_params}")

        best_model = lgb.LGBMClassifier(**model_params)

        # Train the model
        best_model.fit(X_train_tfidf, y_train)
        logger.info("LightGBM model training completed")

        # Create models directory if it doesn't exist
        os.makedirs(os.path.join(root_dir, 'models'), exist_ok=True)
        
        # Save the trained model
        save_model(best_model, os.path.join(root_dir, 'models/lgbm_model.pkl'))
        logger.info("Model saved as lgbm_model.pkl")

    except Exception as e:
        logger.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
