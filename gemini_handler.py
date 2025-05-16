import google.generativeai as genai
from dotenv import load_dotenv
import os

def configure_gemini():
    # Load environment variables from .env file
    load_dotenv()

    # Get the API key from the environment
    api_key = os.getenv("GEMINI_API_KEY")

    if api_key is None:
        raise ValueError("API Key is not set in the environment variables.")

    # Configure the Gemini API client with the API key
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-pro")

def ask_gemini(model, prompt):
    # Generate content using the Gemini model
    response = model.generate_content(prompt)
    return response.text
