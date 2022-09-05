from __future__ import annotations

import os


class FileTypeError(RuntimeError):
    """A runtime error when an unexpected file type was encountered."""
    pass


def basename(
    filename: str,
    with_extension: bool = False
) -> str:
    """Gets the basename from a full filename

    Args:
        filename (str): The full filename of the file
        with_extension (bool): To determine if the return is the basename
    with (True) or without the extension (False).

    Returns:
        str: The basename from the filename.
    """

    bn = os.path.basename(filename)
    if with_extension:
        return bn
    else:
        return os.path.splitext(bn)[0]
