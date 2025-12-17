
import requests
import json

def emotion_detector(text):
    # Prebuild the empty result for blank input
    empty_result = {
        "anger": 0.0,
        "disgust": 0.0,
        "fear": 0.0,
        "joy": 0.0,
        "sadness": 0.0,
        "dominant_emotion": None,
    }

    text_to_analyze = text
    url = "https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
    headers = {
        "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock",
        "Content-Type": "application/json",
    }
    payload = {
        "raw_document": {
            "text": text_to_analyze
        }
    }

    # Make the POST request
    response = requests.post(url, headers=headers, json=payload, timeout=15)
    if response.status_code == 400:
        return (empty_result)

    data = response.json()
    emotion_block = data["emotionPredictions"][0]["emotion"]

    # Build the required output with selected emotions
    emotions = {
        "anger": float(emotion_block.get("anger", 0.0)),
        "disgust": float(emotion_block.get("disgust", 0.0)),
        "fear": float(emotion_block.get("fear", 0.0)),
        "joy": float(emotion_block.get("joy", 0.0)),
        "sadness": float(emotion_block.get("sadness", 0.0)),
    }

    # Determine dominant emotion (highest score)
    dominant_emotion = max(emotions, key=lambda k: emotions[k]) if emotions else None

    # Final output format
    result = {
        **emotions,
        "dominant_emotion": dominant_emotion,
    }

    return (result)

if __name__ == "__main__":
    sample = "I love this new technology."
    try:
        analyzed_text = emotion_detector(sample)
        print(analyzed_text)
    except Exception as e:
        print("Error:", e)
