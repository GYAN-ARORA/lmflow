from typing import List
from ..base import ChatLM
from groq import Groq

class GroqChatLM(ChatLM):
    def __init__(self, groq_api_key, model='llama3-8b-8192', max_tokens=1024, temperature=0.7, *args, **kwargs):
        self.client = Groq(api_key=groq_api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        super().__init__(*args, **kwargs)

    def respond(self, prompt: List[dict]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=prompt
        )
        return response.choices[0].message.content
