from flask import Flask, render_template, request
import joblib
import re

from newspaper import Article

app = Flask(__name__)

# -----------------------------------
# Load trained model and vectorizer
# -----------------------------------
model = joblib.load("model/fake_news_model.pkl")

vectorizer = joblib.load("model/vectorizer.pkl")


# -----------------------------------
# Text cleaning function
# -----------------------------------
def clean_text(text):

    text = str(text).lower()

    # remove URLs
    text = re.sub(r"http\S+", "", text)

    # remove special characters
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# -----------------------------------
# Extract article from URL
# -----------------------------------
def extract_news_from_url(url):

    article = Article(url)

    article.download()

    article.parse()

    return article.title + " " + article.text


# -----------------------------------
# Home route
# -----------------------------------
@app.route("/")
def home():

    return render_template("index.html")


# -----------------------------------
# Prediction route
# -----------------------------------
@app.route("/predict", methods=["POST"])
def predict():

    news = request.form["news"]

    # -----------------------------------
    # If input is URL
    # -----------------------------------
    if news.startswith("http://") or news.startswith("https://"):

        try:

            news = extract_news_from_url(news)

        except:

            return render_template(
                "index.html",
                prediction_text="Could not extract article from URL"
            )

    # -----------------------------------
    # Clean text
    # -----------------------------------
    cleaned = clean_text(news)

    # -----------------------------------
    # Convert to vector
    # -----------------------------------
    vector = vectorizer.transform([cleaned])

    # -----------------------------------
    # Predict class
    # -----------------------------------
    prediction = model.predict(vector)[0]

    # -----------------------------------
    # Get probabilities
    # -----------------------------------
    probability = model.predict_proba(vector)[0]

    fake_score = round(probability[0] * 100, 2)

    real_score = round(probability[1] * 100, 2)

    # -----------------------------------
    # Smarter prediction logic
    # -----------------------------------
    if real_score >= 75:

        result = "LIKELY REAL NEWS"

        authenticity = real_score

    elif fake_score >= 75:

        result = "LIKELY FAKE NEWS"

        authenticity = fake_score

    else:

        result = "UNCERTAIN RESULT"

        authenticity = max(real_score, fake_score)

    # -----------------------------------
    # Render result
    # -----------------------------------
    return render_template(
        "index.html",
        prediction_text=result,
        fake_score=fake_score,
        real_score=real_score,
        authenticity=authenticity
    )


# -----------------------------------
# Run Flask app
# -----------------------------------
if __name__ == "__main__":

    app.run(debug=True)