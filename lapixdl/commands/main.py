from __future__ import annotations

import argparse
import sys
from typing import Sequence

from lapixdl.commands.converter import converter_command_parser


def main(argv: Sequence[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog='LapixDL CLI tools',
        usage='lapixdl <command> [<args>]',
        description='This tool will help you to convert datasets between different formats.'
    )

    subparsers = parser.add_subparsers(
        dest='command',
        help='lapixdl command helpers',
    )

    # Register commands
    converter_command_parser(subparsers)

    help = subparsers.add_parser(
        'help',
        help='Show help for a specific command.',
    )
    help.add_argument(
        'help_cmd',
        nargs='?',
        help='Command to show help for.',
    )

    if len(argv) == 0:
        argv = ['help']

    args = parser.parse_args(argv)

    if args.command == 'help' and args.help_cmd:
        parser.parse_args([args.help_cmd, '--help'])
    elif args.command == 'help':
        parser.parse_args(['--help'])

    if not hasattr(args, 'func'):  # pragma: no cover
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == '__main__':
    raise SystemExit(main())
