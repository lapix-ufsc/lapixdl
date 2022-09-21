from __future__ import annotations

import numpy as np

from lapixdl.formats.annotation import BBox


def bin_mask_from_bb(mask_shape: tuple[int, int], bbox: BBox):
    '''
      Draws a mask from a bounding box
    '''
    return draw_rectangle(np.zeros(mask_shape, int),
                          bbox.upper_left_point,
                          bbox.bottom_right_point,
                          1)


def draw_rectangle(img: np.ndarray, pt1: tuple[int, int], pt2: tuple[int, int], fill: int):
    cp = img.copy()
    cp[slice(pt1[0], pt2[0] + 1), slice(pt1[1], pt2[1] + 1)] = fill
    return cp


def mask_categorical(shape: tuple[int, int]) -> np.ndarray:
    h = int(shape[0])
    w = int(shape[1])
    out = np.zeros(shape, dtype=np.uint8)
    out[:h // 2, :w // 2] = 1
    out[:h // 2, w // 2:] = 2
    out[h // 2:, :w // 2] = 3
    out[h // 2:, w // 2:] = 4
    return np.array(out, dtype=np.uint8)
