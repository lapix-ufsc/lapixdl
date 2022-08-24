from __future__ import annotations

import os


class FileTypeError(RuntimeError):
    pass


def basename(
    filename: str,
    with_extension: bool = False
) -> str:

    bn = os.path.basename(filename)
    if with_extension:
        return bn
    else:
        return os.path.splitext(bn)[0]
