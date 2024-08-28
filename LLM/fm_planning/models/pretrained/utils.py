from typing import Any, Dict, Union, Optional

import pathlib

from LLM.fm_planning import models
from LLM.fm_planning.models.pretrained import PretrainedModel
from LLM.fm_planning.models.pretrained.generative import *
from LLM.fm_planning.models.pretrained.embeddings import *
from LLM.fm_planning.utils import configs


class PretrainedModelFactory(configs.Factory[PretrainedModel]):
    """Pretrained model factory."""

    def __init__(
        self,
        config: Union[str, pathlib.Path, Dict[str, Any]],
        api_key: Optional[str] = None,
        device: str = "auto",
    ):
        """Creates the pretrained model factory from a config.

        Args:
            config: Config path or dict.
            api_key: Pretrained model API key.
            device: Torch device.
        """
        super().__init__(config, "model", models)

        if issubclass(self.cls, OpenAIGenerativeModel):
            if "api_key" not in self.kwargs:
                assert api_key is not None, "OpenAI API key must be provided."
                self.kwargs["api_key"] = api_key

        elif issubclass(self.cls, LlamaGenerativeModel):
            self.kwargs["device"] = device

        elif issubclass(self.cls, OpenAIEmbeddingModel):
            if "api_key" not in self.kwargs:
                assert api_key is not None, "OpenAI API key must be provided."
                self.kwargs["api_key"] = api_key

        elif issubclass(self.cls, HuggingFaceEmbeddingModel):
            self.kwargs["device"] = device

        else:
            raise ValueError(f"Invalid model type: {self.cls}")


def load(
    config: Union[str, pathlib.Path, Dict[str, Any]],
    device: str = "auto",
    **kwargs,
) -> PretrainedModel:
    """Loads the pretrained model factory from a config.

    Args:
        config: Config path or dict.
        device: Torch device.
        **kwargs: Optional model constructor kwargs.

    Returns:
        Pretrained model instance.
    """
    pretrained_model_factory = PretrainedModelFactory(
        config=config,
        device=device,
    )
    return pretrained_model_factory(**kwargs)


def load_config(path: Union[str, pathlib.Path]) -> Dict[str, Any]:
    """Loads a pretrained model config from path.

    Args:
        path: Path to the config, config directory.

    Returns:
        Pretrained model config dict.
    """
    return configs.load_config(path, "model")
