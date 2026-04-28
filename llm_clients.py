import os
from groq import Groq
import google.generativeai as genai
from interfaces import LLMInterface
from config import GROQ_API_KEY, GOOGLE_API_KEY

class GroqLLM(LLMInterface):
    def __init__(self, model_name="llama-3.3-70b-versatile"):
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name,
            temperature=0.0,
        )
        return response.choices[0].message.content

class GeminiLLM(LLMInterface):
    def __init__(self, model_name="gemini-1.5-flash"):
        genai.configure(api_key=GOOGLE_API_KEY)
        self.model = genai.GenerativeModel(model_name)

    def generate(self, prompt: str) -> str:
        response = self.model.generate_content(prompt)
        return response.text