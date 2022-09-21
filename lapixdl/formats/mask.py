from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image


@dataclass
class Mask:
    # FIXME: typing also can be a list[list[int]]
    categorical: np.ndarray

    def __post_init__(self) -> None:
        self.categorical: np.ndarray = np.array(self.categorical, dtype=np.uint8)

        if not len(self.categorical.shape) == 2:
            raise ValueError('Unexpected shape. The categorical mask needs to be a 2D array.')

    @property
    def height(self) -> int:
        return int(self.categorical.shape[0])

    @property
    def width(self) -> int:
        return int(self.categorical.shape[1])

    @property
    def unique_ids(self) -> set[int]:
        return set(np.unique(self.categorical))

    def save(
        self,
        filename: str,
        **kwargs: Any
    ) -> None:
        Image.fromarray(self.categorical).save(filename, **kwargs)
