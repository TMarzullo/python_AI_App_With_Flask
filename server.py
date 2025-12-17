
from flask import Flask, render_template, request, Response
from EmotionDetection.emotion_detection import emotion_detector

app = Flask(
    __name__,
    template_folder="oaqjp-final-project-emb-ai/templates",
    static_folder="oaqjp-final-project-emb-ai/static",
)

# Serve the provided UI
@app.route("/")
def index():
    return render_template("index.html")

# Ensure the Flask decorator path is /emotionDetector
@app.route("/emotionDetector", methods=["GET", "POST"])
def emotionDetector():
    # Accept text either as a GET query param, JSON body, or form field
    text = request.args.get("textToAnalyze")
    if not text and request.is_json:
        payload = request.get_json(silent=True) or {}
        text = payload.get("textToAnalyze")
    if not text:
        text = request.form.get("textToAnalyze", "")

    result = emotion_detector(text or "")

    # Build the output
    message = (
        "For the given statement, the system response is "
        f"'anger': {result['anger']}, "
        f"'disgust': {result['disgust']}, "
        f"'fear': {result['fear']}, "
        f"'joy': {result['joy']} and "
        f"'sadness': {result['sadness']}. "
        f"The dominant emotion is {result['dominant_emotion']}."
    )

    return Response(message, mimetype="text/plain")

if __name__ == "__main__":
    # Run on localhost:5000
    app.run(host="0.0.0.0", port=5000)
