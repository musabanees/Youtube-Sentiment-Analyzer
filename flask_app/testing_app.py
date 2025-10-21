# %%
import requests
import json
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

# Your Flask app URL (when running locally)
BASE_URL = "http://127.0.0.1:5000"

# %%
def test_home():
    response = requests.get(f"{BASE_URL}/")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
    
test_home()

# %%
def test_predict():
    # Sample data for testing
    data = {
        "comments": [
            "I love this product!",
            "This is the worst thing I've ever bought.",
            "It's okay, not great but not terrible."
        ]
    }
    response = requests.post(f"{BASE_URL}/predict", json=data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")

# print("Testing /predict endpoint:")
# test_predict()

# %%
def test_predict_with_timestamps():
    # Sample data with timestamps
    data = {
        "comments": [
            {"text": "This video is amazing!", "timestamp": "2025-10-20T10:00:00"},
            {"text": "Not sure what to think.", "timestamp": "2025-10-20T10:15:00"},
            {"text": "This is terrible.", "timestamp": "2025-10-20T10:30:00"}
        ]
    }
    
    # Send the request
    response = requests.post(
        f"{BASE_URL}/predict_with_timestamps", 
        json=data,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
print("Testing /predict_with_timestamps endpoint:")
test_predict_with_timestamps()