# %%
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend before importing pyplot

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import mlflow
import numpy as np
import re
import os
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MlflowClient
import matplotlib.dates as mdates
import pickle
import yaml
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('youtube_sentiment_api')


load_dotenv()
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
# %%
# Define the preprocessing function
def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    try:
        # Convert to lowercase
        comment = comment.lower()

        # Remove trailing and leading whitespaces
        comment = comment.strip()

        # Remove newline characters
        comment = re.sub(r'\n', ' ', comment)

        # Remove non-alphanumeric characters, except punctuation
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        # Remove stopwords but retain important ones for sentiment analysis
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        print(f"Error in preprocessing comment: {e}")
        return comment


def load_model_and_vectorizer(alias="latest-model"):
    """
    Load both model and vectorizer from MLflow artifacts by stage.
    Returns model and vectorizer separately.
    """
    # Load model info from params.yaml
    with open("../params.yaml", "r") as f:
        params = yaml.safe_load(f)
        
    # Extract model registry info from production environment
    registry_config = params["environments"]["prd"]["model_registry"]
    model_name = registry_config["name"]
    expected_run_id = registry_config.get("run_id")
    
    # Get tracking URI from environment variable (.env file)
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(tracking_uri)

    # Create MLflow client
    client = MlflowClient()
    
    try:
        # Get the model version by alias
        logger.info(f"Looking up model '{model_name}' with alias '{alias}'")
        model_version = client.get_model_version_by_alias(name=model_name, alias=alias)
        print(f"Retrieved model version: {model_version}")
        # Log details of the retrieved model
        version = model_version.version
        run_id = model_version.run_id
        logger.info(f"Found model version {version} with alias '{alias}' (run_id: {run_id})")
        
        # Check if the run_id matches the one in params.yaml
        if expected_run_id:
            if expected_run_id == run_id:
                logger.info(f"✅ Run ID match confirmed: {run_id}")
                print(f"Run ID match confirmed: Model version {version} run ID ({run_id}) matches params.yaml")
            else:
                logger.warning(f"⚠️ Run ID mismatch! Expected: {expected_run_id}, Actual: {run_id}")
                print(f"WARNING: Run ID mismatch! Expected: {expected_run_id}, Actual: {run_id}")
        else:
            logger.info(f"No run_id specified in params.yaml for comparison. Using model with run_id: {run_id}")
        
        # Create the model URI using the model version information
        model_uri = f"models:/{model_version.name}@{alias}"
        logger.info(f"Loading model from URI: {model_uri}")
        model = mlflow.pyfunc.load_model(model_uri)
        
        # Load vectorizer from local path
        vectorizer_path = "../models/tfidf_vectorizer.pkl" 
        with open(vectorizer_path, 'rb') as file:
            vectorizer = pickle.load(file)
        logger.info(f"Loaded vectorizer from local path: {vectorizer_path}")

    except Exception as e:
        logger.error(f"Error loading model or vectorizer: {e}")
        raise
        
    return model, vectorizer
        

# def load_model(model_path, vectorizer_path):
#     """Load the trained model."""
#     try:
#         with open(model_path, 'rb') as file:
#             model = pickle.load(file)
        
#         with open(vectorizer_path, 'rb') as file:
#             vectorizer = pickle.load(file)
      
#         return model, vectorizer
#     except Exception as e:
#         raise


# # Initialize the model and vectorizer
# model, vectorizer = load_model("../models/lgbm_model.pkl", "../models/tfidf_vectorizer.pkl")
# logger.info(f"Model: {model}")
# logger.info(f"Vectorizer: {vectorizer}")

# Initialize the model and vectorizer
model, vectorizer = load_model_and_vectorizer()  # Update paths and versions as needed

# %%

@app.route('/')
def home():
    return "Welcome to our flask api"



@app.route('/predict_with_timestamps', methods=['POST'])
def predict_with_timestamps():
    data = request.json
    comments_data = data.get('comments')
    
    if not comments_data:
        return jsonify({"error": "No comments provided"}), 400

    try:
        comments = [item['text'] for item in comments_data]
        timestamps = [item['timestamp'] for item in comments_data]

        # Preprocess each comment
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        
        # Create a DataFrame for MLflow prediction
        input_df = pd.DataFrame({"text": preprocessed_comments})
        
        logger.info(f"Comment: {input_df['text'].values}")
        
        # Transform comments using the vectorizer (this was missing!)
        transformed_comments = vectorizer.transform(preprocessed_comments)
        
        # Convert the sparse matrix to dense format
        feature_names = vectorizer.get_feature_names_out()
        
        # Convert to DataFrame with proper column names
        input_df = pd.DataFrame(
            transformed_comments.toarray(),
            columns=feature_names
        )
        
        # Use the already loaded model
        predictions = model.predict(input_df)
        
        # Convert predictions to strings for consistency
        predictions = [str(pred) for pred in predictions]
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    
    # Return the response with original comments, predicted sentiments, and timestamps
    response = [{"comment": comment, "sentiment": sentiment, "timestamp": timestamp} for comment, sentiment, timestamp in zip(comments, predictions, timestamps)]
    return jsonify(response)


@app.route('/predict', methods=['POST'])
def predict_mlflow():
    data = request.json
    comments = data.get('comments')

    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    try:
        # Preprocess each comment
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        
        # Create a DataFrame for MLflow prediction
        input_df = pd.DataFrame({"text": preprocessed_comments})
        
        logger.info(f"Comment: {input_df['text'].values}")
        
        # Transform comments using the vectorizer (this was missing!)
        transformed_comments = vectorizer.transform(preprocessed_comments)
        
        # Convert the sparse matrix to dense format
        feature_names = vectorizer.get_feature_names_out()
        
        # Convert to DataFrame with proper column names
        input_df = pd.DataFrame(
            transformed_comments.toarray(),
            columns=feature_names
        )
        
        # Use the already loaded model
        predictions = model.predict(input_df)

        # Convert to list if needed
        if hasattr(predictions, 'tolist'):
            predictions = predictions.tolist()
            
    except Exception as e:
        return jsonify({"error": f"MLflow prediction failed: {str(e)}"}), 500
    
    # Return the same response format as your existing endpoint
    response = [{"comment": comment, "sentiment": sentiment} for comment, sentiment in zip(comments, predictions)]
    return jsonify(response)



@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    try:
        data = request.get_json()
        sentiment_counts = data.get('sentiment_counts')
        
        if not sentiment_counts:
            return jsonify({"error": "No sentiment counts provided"}), 400

        # Prepare data for the pie chart
        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [
            int(sentiment_counts.get('1', 0)),
            int(sentiment_counts.get('0', 0)),
            int(sentiment_counts.get('-1', 0))
        ]
        if sum(sizes) == 0:
            raise ValueError("Sentiment counts sum to zero")
        
        colors = ['#36A2EB', '#C9CBCF', '#FF6384']  # Blue, Gray, Red

        # Generate the pie chart
        plt.figure(figsize=(6, 6))
        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=140,
            textprops={'color': 'w'}
        )
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # Save the chart to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True)
        img_io.seek(0)
        plt.close()

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_chart: {e}")
        return jsonify({"error": f"Chart generation failed: {str(e)}"}), 500

@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    try:
        data = request.get_json()
        comments = data.get('comments')

        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        # Preprocess comments
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]

        # Combine all comments into a single string
        text = ' '.join(preprocessed_comments)

        # Generate the word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='black',
            colormap='Blues',
            stopwords=set(stopwords.words('english')),
            collocations=False
        ).generate(text)

        # Save the word cloud to a BytesIO object
        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format='PNG')
        img_io.seek(0)

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_wordcloud: {e}")
        return jsonify({"error": f"Word cloud generation failed: {str(e)}"}), 500

@app.route('/generate_trend_graph', methods=['POST'])
def generate_trend_graph():
    try:
        data = request.get_json()
        sentiment_data = data.get('sentiment_data')

        if not sentiment_data:
            return jsonify({"error": "No sentiment data provided"}), 400

        # Convert sentiment_data to DataFrame
        df = pd.DataFrame(sentiment_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Set the timestamp as the index
        df.set_index('timestamp', inplace=True)

        # Ensure the 'sentiment' column is numeric
        df['sentiment'] = df['sentiment'].astype(int)

        # Map sentiment values to labels
        sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}

        # Resample the data over monthly intervals and count sentiments
        monthly_counts = df.resample('M')['sentiment'].value_counts().unstack(fill_value=0)

        # Calculate total counts per month
        monthly_totals = monthly_counts.sum(axis=1)

        # Calculate percentages
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        # Ensure all sentiment columns are present
        for sentiment_value in [-1, 0, 1]:
            if sentiment_value not in monthly_percentages.columns:
                monthly_percentages[sentiment_value] = 0

        # Sort columns by sentiment value
        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        # Plotting
        plt.figure(figsize=(12, 6))

        colors = {
            -1: 'red',     # Negative sentiment
            0: 'gray',     # Neutral sentiment
            1: 'green'     # Positive sentiment
        }

        for sentiment_value in [-1, 0, 1]:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[sentiment_value],
                marker='o',
                linestyle='-',
                label=sentiment_labels[sentiment_value],
                color=colors[sentiment_value]
            )

        plt.title('Monthly Sentiment Percentage Over Time')
        plt.xlabel('Month')
        plt.ylabel('Percentage of Comments (%)')
        plt.grid(True)
        plt.xticks(rotation=45)

        # Format the x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))

        plt.legend()
        plt.tight_layout()

        # Save the trend graph to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG')
        img_io.seek(0)
        plt.close()

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_trend_graph: {e}")
        return jsonify({"error": f"Trend graph generation failed: {str(e)}"}), 500



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
