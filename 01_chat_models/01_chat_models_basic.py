from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os

# Load environment variables from .env file
load_dotenv()

# Initialize ChatOpenAI model with a valid model name
model = ChatOpenAI(model="gpt-4o-mini")  # Replace with "gpt-4" if you have access

# Use the invoke method
try:
    result = model.invoke("What is any nuber divided by 0?")
    print("Full result:", result)
    print("Content Only:", result.content)
except Exception as e:
    print("Error:", e)
