from typing import Any

import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel  # type: ignore

from LLM.fm_planning.utils import typing, tensors
from LLM.fm_planning.models.pretrained.base import PretrainedModel
from .utils import mean_pooling


class HuggingFaceEmbeddingModel(PretrainedModel[str]):
    def __init__(self, model: str, device: str = "auto", **kwargs: Any):
        super().__init__()
        self._model_name = model
        self._device = tensors.device(device)
        self._model = AutoModel.from_pretrained(model).to(self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)

    @torch.no_grad()
    def forward(self, x: str) -> typing.Tensor:
        """Compute embeddings for a prompt."""

        # BERT models.
        if self._model_name in ["bert-base-uncased", "bert-large-uncased"]:
            encoded_input = self._tokenizer(x, return_tensors="pt").to(self._device)
            output = self._model(**encoded_input)
            embedding = output[1]

        # MPNet model.
        elif self._model_name == "sentence-transformers/all-mpnet-base-v2":
            encoded_input = self._tokenizer(
                x, padding=True, truncation=True, return_tensors="pt"
            ).to(self._device)
            output = self._model(**encoded_input)
            embedding = mean_pooling(output, encoded_input["attention_mask"])

        # Mistral (7B) model.
        elif self._model_name == "intfloat/e5-mistral-7b-instruct":
            encoded_input = self._tokenizer(
                [x], return_attention_mask=False, padding=False, truncation=True
            )
            encoded_input["input_ids"][0] += [self._tokenizer.eos_token_id]
            encoded_input = self._tokenizer.pad(
                encoded_input,
                padding=True,
                return_attention_mask=True,
                return_tensors="pt",
            ).to(self._device)
            output = self._model(**encoded_input)
            embedding = output.last_hidden_state[:, -1, :]

        # Llama 2 (7B) models.
        elif self._model_name in [
            "meta-llama/Llama-2-7b-hf",
            "meta-llama/Llama-2-7b-chat-hf",
        ]:
            encoded_input = self._tokenizer(x, return_tensors="pt").to(self._device)
            output = self._model(**encoded_input)
            embedding = mean_pooling(output, encoded_input["attention_mask"])

        embedding = embedding.squeeze()
        embedding = F.normalize(embedding, dim=0).cpu().numpy()
        return embedding
