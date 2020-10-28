from typing import Tuple

import cv2
import numpy as np

from lapixdl.evaluation.model import BBox


def bin_mask_from_bb(mask_shape: Tuple[int, int], bbox: BBox):
    '''
      Draws a mask from a bounding box
    '''
    return cv2.rectangle(np.zeros(mask_shape, np.int),
                         bbox.upper_left_point,
                         bbox.bottom_right_point,
                         1, -1)
