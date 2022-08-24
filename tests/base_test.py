from lapixdl.base import basename


def test_basename():
    filename = 'dir/example/file.json'

    assert basename(filename, False) == 'file'
    assert basename(filename, True) == 'file.json'
