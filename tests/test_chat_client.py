"""
Unit tests for ChatClient error handling and chat completion workflows.
"""
import unittest
from unittest.mock import patch, MagicMock
from src.chat_client import ChatClient


class TestChatClient(unittest.TestCase):

    def setUp(self):
        self.client = ChatClient()
        self.client.base_url = "https://api.openai.com/v1"
        self.client.api_key = "test-dummy-key"
        self.client.model = "gemini-2.5-flash"

    @patch("src.chat_client.requests.post")
    def test_successful_chat_completion(self, mock_post):
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "chatcmpl-123",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Alert deduplication prevents alert fatigue and speeds up root cause analysis."
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 25,
                "completion_tokens": 15,
                "total_tokens": 40
            }
        }
        mock_post.return_value = mock_response

        messages = [
            {"role": "system", "content": "You are Alert_IQ Assistant."},
            {"role": "user", "content": "How does alert deduplication help?"}
        ]
        result = self.client.chat_completion(messages)

        self.assertIsNotNone(result)
        self.assertIn("Alert deduplication", result)
        mock_post.assert_called_once()

    @patch("src.chat_client.requests.post")
    def test_401_authentication_error_handling(self, mock_post):
        mock_response = MagicMock()
        mock_response.ok = False
        mock_response.status_code = 401
        mock_response.text = '{"error": {"message": "Invalid API key"}}'
        mock_post.return_value = mock_response

        messages = [{"role": "user", "content": "Hello"}]
        result = self.client.chat_completion(messages)

        self.assertIsNone(result)

    @patch("src.chat_client.requests.post")
    def test_429_rate_limit_error_handling(self, mock_post):
        mock_response = MagicMock()
        mock_response.ok = False
        mock_response.status_code = 429
        mock_response.text = '{"error": {"message": "Rate limit reached"}}'
        mock_post.return_value = mock_response

        messages = [{"role": "user", "content": "Hello"}]
        result = self.client.chat_completion(messages)

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
