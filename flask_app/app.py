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
import mlflow.artifacts
import matplotlib.dates as mdates
import pickle
import yaml
import logging
from dotenv import load_dotenv
from pathlib import Path
import boto3

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('youtube_sentiment_api')


load_dotenv()
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
PROJECT_ROOT = Path(__file__).parent.parent

with open(PROJECT_ROOT / "params.yaml", "r") as f:
    params = yaml.safe_load(f)

vectorizer_path = PROJECT_ROOT / "models" / "tfidf_vectorizer.pkl"
with open(vectorizer_path, 'rb') as file:
    vectorizer = pickle.load(file)
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

def load_model_and_vectorizer():
    """Load model and vectorizer from same MLflow run"""
    try:
        prd_config = params["environments"]["prd"]
        s3_config = prd_config["s3"]
        registry_config = prd_config["model_registry"]
        
        bucket = s3_config["bucket"]
        experiment_id = registry_config["experiment_id"]
        run_id = registry_config["run_id"]
        model_artifact = registry_config["model_artifact_name"]
        vectorizer_artifact = registry_config["vectorizer_artifact_name"]
        
        # Base path for this run's artifacts
        base_s3_uri = f"s3://{bucket}/{experiment_id}/{run_id}/artifacts"
        
        # Load model
        model_s3_uri = f"{base_s3_uri}/{model_artifact}"
        logger.info(f"Loading model from: {model_s3_uri}")
        model = mlflow.pyfunc.load_model(model_s3_uri)
        logger.info("✅ Model loaded")
        
        # Load vectorizer (same run)
        vectorizer_s3_uri = f"{base_s3_uri}/{vectorizer_artifact}"
        logger.info(f"Loading vectorizer from: {vectorizer_s3_uri}")
        
        local_path = mlflow.artifacts.download_artifacts(vectorizer_s3_uri)
        with open(local_path, 'rb') as f:
            vectorizer = pickle.load(f)
        logger.info("✅ Vectorizer loaded")
        
        return model, vectorizer
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise

model, vectorizer = load_model_and_vectorizer()  

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
