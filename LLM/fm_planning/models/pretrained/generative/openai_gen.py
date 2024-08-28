from typing import Any, Optional, List, Dict

import os
import time
import openai
from openai.types.chat.chat_completion import ChatCompletion

from LLM.fm_planning.models.pretrained.base import PretrainedModel


class OpenAIGenerativeModel(PretrainedModel[List[Dict[str, Any]]]):
    """OpenAI generative model."""

    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs: Any):
        """Construct OpenAI generative model."""
        super().__init__()
        self._model = model
        self._api_key = (
            api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        )
        if api_key is None:
            raise ValueError("API key must be provided.")
        self._api_key = api_key
        self._client = openai.OpenAI(api_key=self._api_key)
        self._kwargs = kwargs

    def forward(self, x: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute completion for a prompt."""
        # Send request.
        success = False
        while not success:
            try:
                response: ChatCompletion = self._client.chat.completions.create(
                    messages=x,
                    model=self._model,
                    **self._kwargs,
                )
                success = True
            except:
                time.sleep(1)
                continue

        response = response.model_dump()
        return response
