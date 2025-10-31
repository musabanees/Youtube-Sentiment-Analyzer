#!/usr/bin/env python3
"""
API Endpoint Tests for YouTube Sentiment Analysis Service
Run with: python -m pytest tests/test_api.py -v
"""

import requests
import json
import pytest

# Your Flask app URL (when running locally)
BASE_URL = "http://127.0.0.1:5000"

class TestAPISentiment:
    """Test class for API endpoints"""

    def test_predict_endpoint(self):
        """Test the predict endpoint"""
        data = {
            "comments": [
                "I love this product!",
                "This is the worst thing I've ever bought.",
                "It's okay, not great but not terrible."
            ]
        }
        response = requests.post(f"{BASE_URL}/predict", json=data)
        assert response.status_code == 200

        result = response.json()
        assert isinstance(result, list)
        assert len(result) == 3

        for item in result:
            assert "comment" in item
            assert "sentiment" in item
            assert item["comment"] in data["comments"]

        print("✅ Predict endpoint test passed")

    def test_predict_with_timestamps_endpoint(self):
        """Test the predict with timestamps endpoint"""
        data = {
            "comments": [
                {"text": "This video is amazing!", "timestamp": "2025-10-20T10:00:00"},
                {"text": "Not sure what to think.", "timestamp": "2025-10-20T10:15:00"},
                {"text": "This is terrible.", "timestamp": "2025-10-20T10:30:00"}
            ]
        }

        response = requests.post(f"{BASE_URL}/predict_with_timestamps", json=data)
        assert response.status_code == 200

        result = response.json()
        assert isinstance(result, list)
        assert len(result) == 3

        for item in result:
            assert "comment" in item
            assert "sentiment" in item
            assert "timestamp" in item

        print("✅ Predict with timestamps endpoint test passed")