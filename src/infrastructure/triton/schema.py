import numpy as np
from numpy import typing as npt
from pydantic import BaseModel, ConfigDict


class EmbeddingsModelInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_ids: npt.NDArray[np.int64]
    attention_mask: npt.NDArray[np.int64]
    token_type_ids: npt.NDArray[np.int64]
