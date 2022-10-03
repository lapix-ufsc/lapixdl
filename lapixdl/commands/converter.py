from __future__ import annotations

import argparse
import os
from typing import Sequence

from typing_extensions import TypeAlias

from lapixdl.convert import save_lapixdf_as_masks
from lapixdl.formats import lapix


SUBPARSER_T: TypeAlias = 'argparse._SubParsersAction[argparse.ArgumentParser]'


def converter_command_parser(subparsers: SUBPARSER_T | None = None) -> argparse.ArgumentParser:
    if subparsers is not None:
        parser = subparsers.add_parser('converter')
    else:
        parser = argparse.ArgumentParser(prog='lapixdl dataset converter command')

    group_output = parser.add_argument_group('Define the output format desired')
    mexg_output = group_output.add_mutually_exclusive_group(required=True)
    mexg_output.add_argument(
        '--to-masks',
        action='store_true',
        help='Convert the input data into semantic segmentation masks. This conversion is allowed from lapixdl format.',
    )

    converter_group = parser.add_argument_group('General parameters for the converters command')
    converter_group.add_argument('-i', '--in-path', help='Path for the input file.', required=True, metavar='INPUT_PATH')
    converter_group.add_argument('-o', '--out-path', help='Path for the output dir.', required=True, metavar='OUTPUT_DIR')
    converter_group.add_argument('-j', '--jobs', help='Number of processes/jobs to be used.', default=1)

    masks_group = parser.add_argument_group('Parameters for the conversion of masks')
    masks_group.add_argument('--draw-order', nargs='+', type=int, default=None)
    masks_group.add_argument('--mask-extension', type=str, default='.png')

    if subparsers is not None:
        parser.set_defaults(func=converter_command)

    return parser


def to_masks(
    in_path: str,
    out_path: str,
    mask_extension: str,
    draw_order: tuple[int, ...] | None,
    processes: int,
) -> int:

    print(f'Loading annotations from {in_path}')
    lapix_df = lapix.load(in_path)

    print('Generating the masks...')
    save_lapixdf_as_masks(lapix_df, out_path, mask_extension, draw_order, processes)

    return 0


def converter_command(args: argparse.Namespace | None = None) -> int:
    if args is None:
        return 1

    # if args.to_masks:
    draw_order = tuple(args.draw_order) if isinstance(args.draw_order, list) else None

    return to_masks(
        in_path=os.path.abspath(args.in_path),
        out_path=os.path.abspath(args.out_path),
        mask_extension=args.mask_extension,
        draw_order=draw_order,
        processes=args.jobs
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = converter_command_parser()
    args = parser.parse_args(argv)
    return converter_command(args)


if __name__ == '__main__':
    raise SystemExit(main())
