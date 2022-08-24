import pytest
from lapixdl.formats import lapix
from lapixdl.base import FileTypeError


def test_lapix_load(lapix_filename):
    lapix_df = lapix.load(lapix_filename)
    assert lapix_df.shape == (2, 8)


def test_lapix_load_wrong_filename():
    with pytest.raises(FileTypeError):
        lapix.load('example_wrong.json')


def test_lapix_save(lapix_raw, tmpdir):

    filename = tmpdir.join('out_example.parquet.gzip')

    lapix.save(lapix_raw, str(filename))

    assert len(tmpdir.listdir()) == 1
