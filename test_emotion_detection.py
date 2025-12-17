
import json
import unittest
from unittest.mock import patch

# Import the function from your package module
from EmotionDetection.emotion_detection import emotion_detector


def build_sample_response(anger=0.01, disgust=0.01, fear=0.01, joy=0.01, sadness=0.01, text=""):
    """Construct a sample API JSON payload matching the Watson response shape.
    The aggregated emotion block is under data['emotionPredictions'][0]['emotion'].
    """
    return {
        "emotionPredictions": [{
            "emotion": {
                "anger": anger,
                "disgust": disgust,
                "fear": fear,
                "joy": joy,
                "sadness": sadness,
            },
            "target": "",
            "emotionMentions": [{
                "span": {
                    "begin": 0,
                    "end": len(text),
                    "text": text,
                },
                "emotion": {
                    "anger": anger,
                    "disgust": disgust,
                    "fear": fear,
                    "joy": joy,
                    "sadness": sadness,
                },
            }],
        }],
        "producerId": {"name": "Ensemble Aggregated Emotion Workflow", "version": "0.0.1"},
    }


class FakeResponse:
    """Minimal fake response to mimic requests.Response for our tests."""
    def __init__(self, payload_dict):
        self._payload = payload_dict
        self.text = json.dumps(payload_dict)

    def json(self):
        # Emulate requests.Response.json()
        return self._payload

    def raise_for_status(self):
        # No-op: Pretend the HTTP request succeeded
        pass



class TestEmotionDetection(unittest.TestCase):
    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_joy_statement(self, mock_post):
        self._run_case(mock_post, "I am glad this happened", "joy",
                       dict(joy=0.95, anger=0.02, disgust=0.01, fear=0.01, sadness=0.01))

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_anger_statement(self, mock_post):
        self._run_case(mock_post, "I am really mad about this", "anger",
                       dict(anger=0.95, joy=0.02, disgust=0.01, fear=0.01, sadness=0.01))

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_disgust_statement(self, mock_post):
        self._run_case(mock_post, "I feel disgusted just hearing about this", "disgust",
                       dict(disgust=0.95, joy=0.02, anger=0.01, fear=0.01, sadness=0.01))

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_sadness_statement(self, mock_post):
        self._run_case(mock_post, "I am so sad about this", "sadness",
                       dict(sadness=0.95, joy=0.02, anger=0.01, disgust=0.01, fear=0.01))

    @patch("EmotionDetection.emotion_detection.requests.post")
    def test_fear_statement(self, mock_post):
        self._run_case(mock_post, "I am really afraid that this will happen", "fear",
                       dict(fear=0.95, joy=0.02, anger=0.01, disgust=0.01, sadness=0.01))

    # Helper shared by all methods
    def _run_case(self, mock_post, statement, expected_dom, scores):
        sample = build_sample_response(
            anger=scores.get("anger", 0.01),
            disgust=scores.get("disgust", 0.01),
            fear=scores.get("fear", 0.01),
            joy=scores.get("joy", 0.01),
            sadness=scores.get("sadness", 0.01),
            text=statement,
        )
        mock_post.return_value = FakeResponse(sample)
        result = emotion_detector(statement)

