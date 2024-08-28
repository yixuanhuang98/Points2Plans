from typing import Optional, List, Any

import os
import time
import openai
from openai.types.embedding import Embedding

from LLM.fm_planning.models.pretrained.base import PretrainedModel


class OpenAIEmbeddingModel(PretrainedModel[List[str]]):
    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs: Any):
        super().__init__()
        self._model = model
        self._api_key = (
            api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        )
        if api_key is None:
            raise ValueError("API key must be provided.")
        self._api_key = api_key
        self._client = openai.OpenAI(api_key=self._api_key)

    def forward(self, x: List[str]) -> List[Embedding]:
        """Compute embeddings for a prompt."""
        success = False
        while not success:
            try:
                response = self._client.embeddings.create(input=x, model=self._model)
                success = True
            except:
                time.sleep(1)
                continue

        embeddings = [embedding for embedding in response.data]
        return embeddings
