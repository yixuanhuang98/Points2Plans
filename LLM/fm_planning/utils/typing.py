from typing import TypeVar, Union, Mapping

import numpy as np
import torch


Scalar = Union[float, int, bool]
scalars = (float, int, bool)
Tensor = Union[np.ndarray, torch.Tensor]
ArrayType = TypeVar("ArrayType", np.ndarray, torch.Tensor)
StateType = TypeVar("StateType")
ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
ModelBatchType = TypeVar("ModelBatchType")
DatasetBatchType = TypeVar("DatasetBatchType", bound=Mapping)
