from typing import Generic, Any, TypeVar

import abc

from LLM.fm_planning.utils.typing import ModelBatchType


class PretrainedModel(abc.ABC, Generic[ModelBatchType]):
    """Abstract base class for pretrained models."""

    @abc.abstractmethod
    def forward(self, x: ModelBatchType) -> Any:
        """Compute forward pass."""
        pass


PretrainedModelType = TypeVar("PretrainedModelType", bound=PretrainedModel)
