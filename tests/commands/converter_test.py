from __future__ import annotations

from lapixdl.commands.converter import converter_command
from lapixdl.commands.converter import converter_command_parser
from lapixdl.commands.converter import main
from lapixdl.commands.converter import to_masks


def test_converter_command_parser(subparser):
    parser_converter = converter_command_parser(subparser)
    args = parser_converter.parse_args(['--to-masks', '-i', 'in-path', '-o', 'out-path'])

    assert args.to_masks
    assert len(args.in_path) > 0
    assert len(args.out_path) > 0
    assert hasattr(args, 'func')
    assert parser_converter.prog == 'lapixdl tests converter'

    parser_converter = converter_command_parser()
    assert parser_converter.prog == 'lapixdl dataset converter command'


def test_to_masks(lapix_filename, tmpdir):
    outdir = tmpdir.mkdir('output_A/')
    to_masks(lapix_filename, str(outdir), '.png', None, 1)
    assert len(outdir.listdir()) == 2


def test_converter_command(subparser, lapix_filename, tmpdir):
    assert converter_command(None) == 1

    parser = converter_command_parser(subparser)

    outdir = tmpdir.mkdir('output_A/')
    args = parser.parse_args(['--to-masks', '-i', lapix_filename, '-o', str(outdir)])
    out = converter_command(args)
    assert out == 0
    assert len(outdir.listdir()) == 2


def test_main(lapix_filename, tmpdir):
    out = main(['--to-mask', '-i', lapix_filename, '-o', str(tmpdir)])
    assert out == 0
