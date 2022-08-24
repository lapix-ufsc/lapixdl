import pytest
from lapixdl.formats import labelbox

import pandas as pd


def test_load(labelbox_filename, labelbox_raw):
    labelbox_df = labelbox.load(labelbox_filename)

    assert labelbox_df.shape == (2, 5)
    assert labelbox_df.equals(pd.DataFrame(labelbox_raw))


def test_validate(labelbox_raw):
    assert labelbox.validate(labelbox_raw) is True


def test_validate_wrong_data():
    with pytest.raises(TypeError):
        labelbox.validate(0)

    with pytest.raises(KeyError):
        labelbox.validate([{}])
