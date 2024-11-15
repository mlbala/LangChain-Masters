import openai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Example API call using the new interface
try:
    # Test an API call
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "What is the capital of France?"}]
    )
    print("Response content:", response["choices"][0]["message"]["content"])
except openai.error.AuthenticationError as e:
    print("Authentication error:", str(e))
except openai.error.RateLimitError as e:
    print("Rate limit error:", str(e))
except openai.error.OpenAIError as e:
    print("OpenAI error:", str(e))
except Exception as e:
    print("Unexpected error:", str(e))
