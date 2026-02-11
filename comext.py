# youtube_comment_extractor_secure.py

import os
import csv
from googleapiclient.discovery import build

VIDEO_ID = "Geq1_M3DrQ4"   # Put your video ID here
MAX_COMMENTS = 500


def get_api_key():
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Set YOUTUBE_API_KEY environment variable.")
    return api_key


def get_comments(video_id, max_comments):
    youtube = build("youtube", "v3", developerKey=get_api_key())

    comments = []
    next_page_token = None

    while len(comments) < max_comments:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            pageToken=next_page_token,
            textFormat="plainText"
        )

        response = request.execute()

        for item in response.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]

            comments.append({
                "author": snippet["authorDisplayName"],
                "comment": snippet["textDisplay"],
                "likes": snippet["likeCount"]
            })

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return comments


def save_to_csv(comments, filename="comments.csv"):
    with open(filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["author", "comment", "likes"])
        writer.writeheader()
        writer.writerows(comments)


if __name__ == "__main__":
    comments = get_comments(VIDEO_ID, MAX_COMMENTS)
    save_to_csv(comments)
    print(f"Extracted {len(comments)} comments securely.")

