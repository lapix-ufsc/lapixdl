from __future__ import annotations

import os

import numpy as np
import pytest
from PIL import Image

from lapixdl.formats.mask import Mask
from testing.utils import mask_categorical


@pytest.fixture
def shape():
    return (1200, 1600)


@pytest.fixture
def mask(shape):
    return Mask(mask_categorical(shape))


def test_wrong_init():
    with pytest.raises(ValueError):
        Mask([1, 1])


def test_shape(mask, shape):
    assert mask.height == shape[0]
    assert mask.width == shape[1]


def test_unique_ids(mask):
    assert mask.unique_ids == {1, 2, 3, 4}


def test_save(mask, tmpdir):
    out_path = tmpdir.join('test.png')
    mask.save(str(out_path))
    assert out_path.isfile()

    msk = Image.open(str(out_path)).convert('L')
    assert np.array_equal(mask.categorical, msk)
