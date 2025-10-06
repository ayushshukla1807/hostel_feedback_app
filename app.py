from flask import Flask, request, render_template, jsonify
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# --- Corrected NLTK Data Download Logic ---
# This block ensures the necessary NLTK data (VADER lexicon for sentiment analysis) is available.
# It checks if the data is already downloaded, and if not, it downloads it.
try:
    # Attempt to find the data. If it exists, we do nothing.
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    # If 'vader_lexicon' is not found (which causes LookupError),
    # then we explicitly download it.
    print("NLTK VADER lexicon not found. Downloading...")
    nltk.download('vader_lexicon')
    print("NLTK VADER lexicon downloaded.")
# --------------------------------------------

app = Flask(__name__)
analyzer = SentimentIntensityAnalyzer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_feedback():
    data = request.get_json()
    feedback_text = data.get('feedback', '')

    if not feedback_text.strip():
        return jsonify({'error': 'No feedback provided'}), 400

    scores = analyzer.polarity_scores(feedback_text)
    sentiment = "Neutral"
    if scores['compound'] >= 0.05:
        sentiment = "Positive"
    elif scores['compound'] <= -0.05:
        sentiment = "Negative"

    return jsonify({
        'text': feedback_text,
        'sentiment': sentiment,
        'scores': scores
    })

if __name__ == '__main__':
    # Important: For deployment, debug=False is essential.
    # For local testing, debug=True is helpful.
    # You can either remove debug=True or set it to False if you want to test production-like behavior.
    # For now, keep it True for local testing.
    app.run(debug=True)