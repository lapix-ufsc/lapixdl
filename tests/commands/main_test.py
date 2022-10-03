from __future__ import annotations

import pytest

from lapixdl.commands.main import main


def test_main(lapix_filename, tmpdir):
    out = main(['converter', '--to-masks', '-i', lapix_filename, '-o', str(tmpdir)])
    assert out == 0


def test_main_help():
    with pytest.raises(SystemExit) as pytest_wrapped_e1:
        main(['help'])
    assert pytest_wrapped_e1.value.code == 0

    with pytest.raises(SystemExit) as pytest_wrapped_e2:
        main([])
    assert pytest_wrapped_e2.value.code == 0


def test_main_help_other_command():
    with pytest.raises(SystemExit) as pytest_wrapped_e1:
        main(['help', 'converter'])

    assert pytest_wrapped_e1.value.code == 0


def test_main_wrong_command():
    with pytest.raises(SystemExit) as pytest_wrapped_e1:
        main(['wrong-command'])

    assert pytest_wrapped_e1.value.code == 2
