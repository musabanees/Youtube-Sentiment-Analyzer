// popup.js

document.addEventListener("DOMContentLoaded", async () => {
  const outputDiv = document.getElementById("output");
  const API_KEY = 'AIzaSyAA3gUGMaiKAHJnnE8juwgVURrqHPsoWRo';
  // const API_URL = 'http://my-elb-2062136355.us-east-1.elb.amazonaws.com:80';   
  const API_URL = 'http://localhost:5000/';
  // const API_URL = 'http://23.20.221.231:8080/';

  // Get the current tab's URL
  chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
    const url = tabs[0].url;
    const youtubeRegex = /^https:\/\/(?:www\.)?youtube\.com\/watch\?v=([\w-]{11})/;
    const match = url.match(youtubeRegex);

    if (match && match[1]) {
      const videoId = match[1];
      outputDiv.innerHTML = `<div class="section-title">YouTube Video ID</div><p>${videoId}</p><p>Fetching comments...</p>`;

      const comments = await fetchComments(videoId);
      if (comments.length === 0) {
        outputDiv.innerHTML += "<p>No comments found for this video.</p>";
        return;
      }

      outputDiv.innerHTML += `<p>Fetched ${comments.length} comments. Performing sentiment analysis...</p>`;
      const predictions = await getSentimentPredictions(comments);

      if (predictions) {
        // Process the predictions to get sentiment counts and sentiment data
        const sentimentCounts = { "1": 0, "0": 0, "-1": 0 };
        const sentimentData = []; // For trend graph
        const totalSentimentScore = predictions.reduce((sum, item) => sum + parseInt(item.sentiment), 0);
        predictions.forEach((item, index) => {
          sentimentCounts[item.sentiment]++;
          sentimentData.push({
            timestamp: item.timestamp,
            sentiment: parseInt(item.sentiment)
          });
        });

        // Compute metrics
        const totalComments = comments.length;
        const uniqueCommenters = new Set(comments.map(comment => comment.authorId)).size;
        const totalWords = comments.reduce((sum, comment) => sum + comment.text.split(/\s+/).filter(word => word.length > 0).length, 0);
        const avgWordLength = (totalWords / totalComments).toFixed(2);
        const avgSentimentScore = (totalSentimentScore / totalComments).toFixed(2);

        // Normalize the average sentiment score to a scale of 0 to 10
        const normalizedSentimentScore = (((parseFloat(avgSentimentScore) + 1) / 2) * 10).toFixed(1);

        // Calculate percentages
        const positivePercent = ((sentimentCounts["1"] / totalComments) * 100).toFixed(1);
        const neutralPercent = ((sentimentCounts["0"] / totalComments) * 100).toFixed(1);
        const negativePercent = ((sentimentCounts["-1"] / totalComments) * 100).toFixed(1);

        // Calculate engagement quality score
        const engagementScore = (
          (sentimentCounts["1"] * 1.0) + 
          (sentimentCounts["0"] * 0.5) + 
          (sentimentCounts["-1"] * 0.0)
        ) / totalComments * 100;

        // Determine benchmark
        const benchmarkText = normalizedSentimentScore >= 7 ? "Excellent" :
                              normalizedSentimentScore >= 5 ? "Good" :
                              normalizedSentimentScore >= 3 ? "Mixed" : "Poor";
        const benchmarkClass = normalizedSentimentScore >= 7 ? "excellent" :
                               normalizedSentimentScore >= 5 ? "good" :
                               normalizedSentimentScore >= 3 ? "mixed" : "poor";

        // Add the Hero Metrics section
        outputDiv.innerHTML += `
          <div class="hero-section">
            <div class="hero-metric">
              <div class="hero-value">${normalizedSentimentScore}/10</div>
              <div class="hero-label">Overall Sentiment</div>
              <div class="hero-badge ${benchmarkClass}">${benchmarkText}</div>
            </div>
            <div class="hero-metric">
              <div class="hero-value">${positivePercent}%</div>
              <div class="hero-label">Positive Comments</div>
            </div>
          </div>
        `;

        // Add Quick Stats Grid
        outputDiv.innerHTML += `
          <div class="stats-grid">
            <div class="stat-card">
              <div class="stat-icon">Comments</div>
              <div class="stat-value">${totalComments}</div>
              <div class="stat-label">Total Comments</div>
            </div>
            <div class="stat-card">
              <div class="stat-icon">Users</div>
              <div class="stat-value">${uniqueCommenters}</div>
              <div class="stat-label">Unique Commenters</div>
            </div>
            <div class="stat-card">
              <div class="stat-icon">Quality</div>
              <div class="stat-value">${engagementScore.toFixed(0)}%</div>
              <div class="stat-label">Engagement Quality</div>
            </div>
          </div>
        `;

        // Add AI-Generated Insights
        outputDiv.innerHTML += `
          <div class="insights-section">
            <div class="section-title">Key Insights</div>
            <ul class="insights-list">
              ${generateInsights(sentimentCounts, totalComments, sentimentData, positivePercent, negativePercent)}
            </ul>
          </div>
        `;

        // Add Sentiment Breakdown with visual bar
        outputDiv.innerHTML += `
          <div class="sentiment-breakdown">
            <div class="section-title">Sentiment Distribution</div>
            <div class="sentiment-bar-container">
              <div class="sentiment-bar">
                <div class="positive-bar" style="width: ${positivePercent}%">${positivePercent}%</div>
                <div class="neutral-bar" style="width: ${neutralPercent}%">${neutralPercent}%</div>
                <div class="negative-bar" style="width: ${negativePercent}%">${negativePercent}%</div>
              </div>
            </div>
            <div class="legend">
              <span class="legend-item positive">Positive (${sentimentCounts["1"]})</span>
              <span class="legend-item neutral">Neutral (${sentimentCounts["0"]})</span>
              <span class="legend-item negative">Negative (${sentimentCounts["-1"]})</span>
            </div>
          </div>
        `;

        // Add the Sentiment Analysis Results section with a placeholder for the chart
        outputDiv.innerHTML += `
          <div class="section">
            <div class="section-title">Sentiment Analysis Results</div>
            <p>See the pie chart below for sentiment distribution.</p>
            <div id="chart-container"></div>
          </div>`;

        // Fetch and display the pie chart inside the chart-container div
        await fetchAndDisplayChart(sentimentCounts);

        // Add the Sentiment Trend Graph section
        outputDiv.innerHTML += `
          <div class="section">
            <div class="section-title">Sentiment Trend Over Time</div>
            <div id="trend-graph-container"></div>
          </div>`;

        // Fetch and display the sentiment trend graph
        await fetchAndDisplayTrendGraph(sentimentData);

        // Add the Word Cloud section
        outputDiv.innerHTML += `
          <div class="section">
            <div class="section-title">Comment Wordcloud</div>
            <div id="wordcloud-container"></div>
          </div>`;

        // Fetch and display the word cloud inside the wordcloud-container div
        await fetchAndDisplayWordCloud(comments.map(comment => comment.text));
      }
    } else {
      outputDiv.innerHTML = "<p>This is not a valid YouTube URL.</p>";
    }
  });

  async function fetchComments(videoId) {
    let comments = [];
    let pageToken = "";
    try {
      while (comments.length < 500) {
        const response = await fetch(`https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId=${videoId}&maxResults=100&pageToken=${pageToken}&key=${API_KEY}`);
        const data = await response.json();
        if (data.items) {
          data.items.forEach(item => {
            const commentText = item.snippet.topLevelComment.snippet.textOriginal;
            const timestamp = item.snippet.topLevelComment.snippet.publishedAt;
            const authorId = item.snippet.topLevelComment.snippet.authorChannelId?.value || 'Unknown';
            comments.push({ text: commentText, timestamp: timestamp, authorId: authorId });
          });
        }
        pageToken = data.nextPageToken;
        if (!pageToken) break;
      }
    } catch (error) {
      console.error("Error fetching comments:", error);
      outputDiv.innerHTML += "<p>Error fetching comments.</p>";
    }
    return comments;
  }

  async function getSentimentPredictions(comments) {
    try {
      const response = await fetch(`${API_URL}/predict_with_timestamps`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ comments })
      });
      const result = await response.json();
      if (response.ok) {
        return result; // The result now includes sentiment and timestamp
      } else {
        throw new Error(result.error || 'Error fetching predictions');
      }
    } catch (error) {
      console.error("Error fetching predictions:", error);
      outputDiv.innerHTML += "<p>Error fetching sentiment predictions.</p>";
      return null;
    }
  }

  async function fetchAndDisplayChart(sentimentCounts) {
    try {
      const response = await fetch(`${API_URL}/generate_chart`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sentiment_counts: sentimentCounts })
      });
      if (!response.ok) {
        throw new Error('Failed to fetch chart image');
      }
      const blob = await response.blob();
      const imgURL = URL.createObjectURL(blob);
      const img = document.createElement('img');
      img.src = imgURL;
      img.style.width = '100%';
      img.style.marginTop = '20px';
      // Append the image to the chart-container div
      const chartContainer = document.getElementById('chart-container');
      chartContainer.appendChild(img);
    } catch (error) {
      console.error("Error fetching chart image:", error);
      outputDiv.innerHTML += "<p>Error fetching chart image.</p>";
    }
  }

  async function fetchAndDisplayWordCloud(comments) {
    try {
      const response = await fetch(`${API_URL}/generate_wordcloud`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ comments })
      });
      if (!response.ok) {
        throw new Error('Failed to fetch word cloud image');
      }
      const blob = await response.blob();
      const imgURL = URL.createObjectURL(blob);
      const img = document.createElement('img');
      img.src = imgURL;
      img.style.width = '100%';
      img.style.marginTop = '20px';
      // Append the image to the wordcloud-container div
      const wordcloudContainer = document.getElementById('wordcloud-container');
      wordcloudContainer.appendChild(img);
    } catch (error) {
      console.error("Error fetching word cloud image:", error);
      outputDiv.innerHTML += "<p>Error fetching word cloud image.</p>";
    }
  }

  async function fetchAndDisplayTrendGraph(sentimentData) {
    try {
      const response = await fetch(`${API_URL}/generate_trend_graph`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sentiment_data: sentimentData })
      });
      if (!response.ok) {
        throw new Error('Failed to fetch trend graph image');
      }
      const blob = await response.blob();
      const imgURL = URL.createObjectURL(blob);
      const img = document.createElement('img');
      img.src = imgURL;
      img.style.width = '100%';
      img.style.marginTop = '20px';
      // Append the image to the trend-graph-container div
      const trendGraphContainer = document.getElementById('trend-graph-container');
      trendGraphContainer.appendChild(img);
    } catch (error) {
      console.error("Error fetching trend graph image:", error);
      outputDiv.innerHTML += "<p>Error fetching trend graph image.</p>";
    }
  }
});

// Helper function to generate AI-like insights
function generateInsights(sentimentCounts, totalComments, sentimentData, positivePercent, negativePercent) {
  const insights = [];
  
  // Sentiment dominance insight
  if (positivePercent > 70) {
    insights.push(`<li><strong>Highly positive reception!</strong> ${positivePercent}% of viewers are loving this content.</li>`);
  } else if (negativePercent > 60) {
    insights.push(`<li><strong>Controversial content:</strong> ${negativePercent}% negative sentiment indicates strong viewer disagreement.</li>`);
  } else if (Math.abs(positivePercent - negativePercent) < 20) {
    insights.push(`<li><strong>Balanced opinions:</strong> Viewers are split with mixed reactions (${positivePercent}% positive vs ${negativePercent}% negative).</li>`);
  }
  
  // Engagement quality insight
  const engagementScore = (sentimentCounts["1"] * 1.5 + sentimentCounts["0"] * 0.5 + sentimentCounts["-1"] * 1.2) / totalComments * 20;
  if (engagementScore > 75) {
    insights.push(`<li><strong>High engagement:</strong> Comments show strong emotional investment from viewers.</li>`);
  } else if (engagementScore < 40) {
    insights.push(`<li><strong>Low engagement:</strong> Consider more interactive content to boost viewer participation.</li>`);
  }
  
  // Community health insight
  if (positivePercent > 50 && negativePercent < 25) {
    insights.push(`<li><strong>Healthy community:</strong> Positive and constructive discussion environment.</li>`);
  }
  
  // Content performance insight
  if (totalComments > 100) {
    insights.push(`<li><strong>High activity:</strong> ${totalComments} comments indicate strong viewer interest.</li>`);
  } else if (totalComments < 20) {
    insights.push(`<li><strong>Limited feedback:</strong> Consider promoting discussion with engaging questions.</li>`);
  }
  
  return insights.length > 0 ? insights.join('') : '<li>Analysis complete - check the metrics above for detailed insights.</li>';
}
