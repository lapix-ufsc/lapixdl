from __future__ import annotations

import argparse

import pytest


@pytest.fixture
def parser():
    return argparse.ArgumentParser(prog='lapixdl tests')


@pytest.fixture
def subparser(parser):
    return parser.add_subparsers(dest='command')
