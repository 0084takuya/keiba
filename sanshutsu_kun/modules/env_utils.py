import os
from dotenv import load_dotenv
import openai

def load_env_and_set_api():
    """
    .envをロードし、OpenAI APIキーをセットする
    """
    load_dotenv()
    openai.api_key = os.getenv('OPENAI_API_KEY') 