from flask import Flask, render_template, request, jsonify
from googleapiclient.discovery import build
from textblob import TextBlob
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# 🔑 Replace with your YouTube Data API key
YOUTUBE_API_KEY = "YOUR_YOUTUBE_API_KEY"

def extract_video_id(url):
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    return None

def get_comments(video_id, max_results=100):
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=max_results,
        textFormat="plainText"
    )
    response = request.execute()

    comments = []
    for item in response["items"]:
        comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        comments.append(comment)

    return comments

def analyze_sentiment(comments):
    pos, neg, neu = 0, 0, 0

    for comment in comments:
        polarity = TextBlob(comment).sentiment.polarity
        if polarity > 0:
            pos += 1
        elif polarity < 0:
            neg += 1
        else:
            neu += 1

    return pos, neg, neu

def create_graph(pos, neg, neu):
    plt.figure()
    plt.bar(["Positive", "Negative", "Neutral"], [pos, neg, neu])
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.title("YouTube Comment Sentiment")

    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return graph_url

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    youtube_url = request.form["url"]
    video_id = extract_video_id(youtube_url)

    comments = get_comments(video_id)
    pos, neg, neu = analyze_sentiment(comments)

    graph = create_graph(pos, neg, neu)

    return jsonify({
        "positive": pos,
        "negative": neg,
        "neutral": neu,
        "graph": graph
    })

if __name__ == "__main__":
    app.run(debug=True)

