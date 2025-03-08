from flask import Flask, request, render_template, jsonify
import joblib
import re
import string
from nltk.corpus import stopwords
import nltk
import logging

# Download NLTK stopwords (only needed once)
nltk.download('stopwords')

# Initialize Flask app
app = Flask(__name__)

# Set up logging
logging.basicConfig(filename='app.log', level=logging.INFO)

# Load the model and vectorizer
try:
    model = joblib.load('sentiment_model.pkl')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

try:
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    print("Vectorizer loaded successfully!")
except Exception as e:
    print(f"Error loading vectorizer: {e}")
    vectorizer = None

# Text cleaning function
def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# Home route
@app.route('/', methods=['GET', 'POST'])
def home():
    logging.info("Home page accessed.")
    if request.method == 'POST':
        logging.info("Prediction requested.")
        if model is None or vectorizer is None:
            return "Model or vectorizer not loaded. Please check the server logs."
        # Get user input from the form
        user_input = request.form['text']
        if not user_input.strip():
            return render_template('index.html', error="Please enter some text.")
        # Clean the input text
        cleaned_input = clean_text(user_input)
        # Convert text to numerical features
        input_vectorized = vectorizer.transform([cleaned_input])
        # Make a prediction
        prediction = model.predict(input_vectorized)[0]
        # Get prediction probabilities
        probabilities = model.predict_proba(input_vectorized)[0]
        confidence = probabilities[prediction]  # Confidence for the predicted class
        # Map prediction to sentiment
        sentiment = "Positive" if prediction == 1 else "Negative"
        # Return the result to the frontend
        return render_template('index.html', sentiment=sentiment, confidence=confidence, user_input=user_input)
    # Render the form for GET requests
    return render_template('index.html')

# Feedback route
@app.route('/feedback', methods=['POST'])
def feedback():
    feedback_data = request.json
    with open('feedback.txt', 'a') as f:
        f.write(f"{feedback_data}\n")
    return jsonify({"status": "success"})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

