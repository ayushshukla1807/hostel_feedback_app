# Step 2.1: Import necessary tools
from flask import Flask, request, render_template, jsonify
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Step 2.2: Prepare the AI tool (Sentiment Analyzer)

# This block ensures the AI's language pack is downloaded.
# It's like telling the computer: "Hey, for analyzing feelings, you'll need this specific language data."
# It tries to find it first, and if it can't, it downloads it.
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except nltk.downloader.DownloadError:
    nltk.download('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# Step 2.3: Initialize the Flask Application
# This is like saying, "Okay, let's start our web application."
app = Flask(__name__)

# This creates an instance of our sentiment analyzer, ready to go.
analyzer = SentimentIntensityAnalyzer()

# Step 2.4: Define the "Homepage" route
# This tells Flask: "When someone visits the main address of our app (like www.myfeedbackapp.com/),
# show them the index.html page."
@app.route('/')
def index():
    # Flask looks for 'index.html' inside the 'templates' folder.
    return render_template('index.html')

# Step 2.5: Define the "Analyze Feedback" route
# This route is specifically for when the user clicks the "Analyze Feedback" button.
# 'methods=['POST']' means it expects data to be sent TO it, not just visited.
@app.route('/analyze', methods=['POST'])
def analyze_feedback():
    # Step 2.5.1: Get the data sent from the webpage
    # The webpage (index.html) will send the feedback text in a JSON format.
    # We need to get that text.
    data = request.get_json()
    feedback_text = data.get('feedback', '') # Get the 'feedback' value, or empty string if not found

    # Step 2.5.2: Check if feedback was actually provided
    if not feedback_text.strip(): # .strip() removes any accidental spaces
        # If no feedback, send an error message back to the webpage
        return jsonify({'error': 'No feedback provided'}), 400 # 400 means "Bad Request"

    # Step 2.5.3: Use the AI to analyze the feedback
    # This is where the magic happens! The analyzer looks at the text.
    scores = analyzer.polarity_scores(feedback_text)

    # Step 2.5.4: Determine the overall sentiment
    # The 'scores' dictionary has 'pos', 'neg', 'neu', and 'compound'.
    # The 'compound' score is a single value from -1 (most negative) to +1 (most positive).
    # We use simple thresholds to classify the sentiment.
    sentiment = "Neutral"
    if scores['compound'] >= 0.05:
        sentiment = "Positive"
    elif scores['compound'] <= -0.05:
        sentiment = "Negative"

    # Step 2.5.5: Send the results back to the webpage
    # The webpage's JavaScript will receive this JSON data.
    return jsonify({
        'text': feedback_text,
        'sentiment': sentiment,
        'scores': scores # You can show these scores if you want, or just the sentiment label
    })

# Step 2.6: How to run your application
# This block ensures that when you run THIS file (app.py), the web server starts.
if __name__ == '__main__':
    # app.run() starts the local web server.
    # debug=True is super helpful during development:
    # - It automatically restarts the server when you save changes to app.py.
    # - It shows detailed error messages in your browser if something goes wrong.
    # REMEMBER: For a real deployment, you'll turn debug=False.
    app.run(debug=True)