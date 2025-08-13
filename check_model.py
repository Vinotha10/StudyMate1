import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load your environment variables if stored in a .env file
load_dotenv()

# Configure your Gemini API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# List available models
models = genai.list_models()
for m in models:
    print(m.name)