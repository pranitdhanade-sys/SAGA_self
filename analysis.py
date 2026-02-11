from youtube_comment_downloader import YoutubeCommentDownloader
from textblob import TextBlob

def get_comments(youtube_url, limit=200):
    downloader = YoutubeCommentDownloader()
    comments = downloader.get_comments_from_url(youtube_url)

    comment_list = []
    for idx, comment in enumerate(comments):
        if idx >= limit:
            break
        comment_list.append(comment['text'])
    return comment_list


def analyze_sentiment(comments):
    results = {"positive": 0, "negative": 0, "neutral": 0}

    for text in comments:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity

        if polarity > 0:
            results["positive"] += 1
        elif polarity < 0:
            results["negative"] += 1
        else:
            results["neutral"] += 1

    return results

