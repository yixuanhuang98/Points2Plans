from typing import Dict, Any

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

from LLM.fm_planning.models.pretrained.base import PretrainedModel


class LlamaGenerativeModel(PretrainedModel[str]):
    """Llama generative model."""

    def __init__(self, model: str, device: str = "auto", **kwargs: Any):
        """Construct Llama generative model."""
        super().__init__()
        self._tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model,
            torch_dtype=torch.float16,
            device_map=device,
            **kwargs,
        )
        self.device: torch.device = self._model.device

    def forward(self, x: str) -> str:
        """Compute completion for a prompt."""
        # Send request.
        input_ids: torch.Tensor = self._tokenizer.encode(
            x, add_special_tokens=True, return_tensors="pt"
        )
        response_ids = self._model.generate(input_ids.to(self.device))[0]
        response = self._tokenizer.decode(
            response_ids[input_ids.shape[-1] :], skip_special_tokens=True
        )

        return response
