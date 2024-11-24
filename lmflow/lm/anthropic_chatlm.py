from typing import List
from ..base import ChatLM
from anthropic import Anthropic


class AnthropicChatLM(ChatLM):
    def __init__(self, anthropic_api_key, model='claude-3-opus-20240229', max_tokens=1024, *args, **kwargs):
        self.client = Anthropic(api_key=anthropic_api_key)
        self.model = model
        self.max_tokens = max_tokens
        super().__init__(*args, **kwargs)

    def respond(self, prompt: List[dict]) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=prompt
        )
        return response.content[0].text
