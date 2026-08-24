"""
Chat Completion Client for OpenAI-Compatible APIs
Configures environment variables, logs request/response payloads, and handles API errors cleanly.
"""
import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("ChatClient")


def load_env_file(env_path: Optional[str] = None) -> None:
    """
    Lightweight .env file parser to populate os.environ without third-party dependencies.
    """
    target = Path(env_path) if env_path else Path(__file__).resolve().parent.parent / ".env"
    if not target.exists():
        # Fallback to current working directory .env if present
        target = Path(".env")

    if target.exists():
        with open(target, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip("\"'")
                # Don't overwrite if already set in environment
                if key not in os.environ:
                    os.environ[key] = val


class ChatClient:
    """
    Client for OpenAI-compatible chat completion APIs (Gemini, OpenAI, Ollama, Groq, etc.).
    """

    def __init__(self, env_file: Optional[str] = None):
        load_env_file(env_file)

        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.model = os.getenv("OPENAI_MODEL", "gemini-2.5-flash")

        if not self.api_key and "localhost" not in self.base_url and "127.0.0.1" not in self.base_url:
            logger.warning("OPENAI_API_KEY is not set. Requests to authenticated endpoints may fail.")

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = 1000
    ) -> Optional[str]:
        """
        Sends a chat completion request to the configured API.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in completion.

        Returns:
            The model's text response string, or None if the request failed.
        """
        endpoint = f"{self.base_url}/chat/completions" if not self.base_url.endswith("/chat/completions") else self.base_url
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        # Task 3: Log outgoing request payload
        logger.info("Sending chat completion request to: %s", endpoint)
        logger.info("Using Model: %s", self.model)
        logger.info("Request Payload:\n%s", json.dumps(payload, indent=2))

        try:
            response = requests.post(endpoint, headers=headers, json=payload, timeout=30)

            # Task 4: Catch and handle HTTP errors clearly
            if response.status_code == 401:
                logger.error(
                    "❌ Authentication Error (401): Invalid API key or unauthorized request.\n"
                    "👉 Check your OPENAI_API_KEY in .env."
                )
                return None
            elif response.status_code == 429:
                logger.error(
                    "❌ Rate Limit / Quota Exceeded (429): You have exceeded your current API rate limit or quota.\n"
                    "👉 Please wait before retrying or check your API billing / quotas."
                )
                return None
            elif response.status_code == 404:
                logger.error(
                    "❌ Endpoint or Model Not Found (404): The requested URL or model '%s' does not exist.\n"
                    "👉 Response: %s", self.model, response.text
                )
                return None
            elif not response.ok:
                logger.error(
                    "❌ API Request Failed with status code %d: %s",
                    response.status_code,
                    response.text
                )
                return None

            data = response.json()

            # Task 3: Log incoming response payload and token usage
            usage = data.get("usage", {})
            logger.info("Response received successfully (Status: %d)", response.status_code)
            logger.info("Token Usage: Prompt Tokens=%s, Completion Tokens=%s, Total Tokens=%s",
                        usage.get("prompt_tokens", "N/A"),
                        usage.get("completion_tokens", "N/A"),
                        usage.get("total_tokens", "N/A"))
            logger.info("Full Response Payload:\n%s", json.dumps(data, indent=2))

            # Task 2: Extract model text reply
            choices = data.get("choices", [])
            if choices and "message" in choices[0] and "content" in choices[0]["message"]:
                reply_content = choices[0]["message"]["content"]
                return reply_content
            else:
                logger.warning("No message content found in API choices response.")
                return None

        except requests.exceptions.Timeout:
            logger.error("❌ Connection Timeout: The API request timed out after 30 seconds.")
            return None
        except requests.exceptions.ConnectionError as e:
            logger.error("❌ Connection Error: Could not reach the API endpoint at %s. Details: %s", self.base_url, e)
            return None
        except Exception as e:
            logger.error("❌ Unexpected Error occurred: %s", str(e))
            return None


def main():
    """
    Executes a sample chat completion request demonstrating system and user prompts.
    """
    print("=" * 60)
    print("🤖 Alert_IQ - Chat Completion Client Initialization")
    print("=" * 60)

    client = ChatClient()
    print(f"📡 Configured Base URL : {client.base_url}")
    print(f"🧠 Configured Model    : {client.model}")
    print(f"🔑 API Key Configured  : {'Yes (Masked)' if client.api_key else 'No'}")
    print("-" * 60)

    # Task 2: System and User prompt
    messages = [
        {"role": "system", "content": "You are Alert_IQ Assistant, an intelligent incident monitoring and alert response agent."},
        {"role": "user", "content": "Explain in two sentences how automated alert deduplication improves incident response time."}
    ]

    print("🚀 Sending Chat Completion Request...")
    reply = client.chat_completion(messages=messages)

    print("-" * 60)
    if reply:
        print("💬 Model Response:")
        print(reply)
    else:
        print("⚠️ No response received due to an error (see log details above).")
    print("=" * 60)


if __name__ == "__main__":
    main()
