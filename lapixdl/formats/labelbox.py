from __future__ import annotations

import json
from typing import Any

import pandas as pd


EXPECTED_DATA = set({'ID', 'External ID', 'Reviews', 'Label'})


def load(
    filename: str,
) -> pd.DataFrame:
    """Loads the labelbox JSON file into a pandas dataframe

    Args
        filename (str): The full filename for the JSON file.

    Returns: A pandas DataFrame with the data from labelbox JSON file.
    """
    with open(filename) as f:
        raw = json.load(f)

    validate(raw)

    return pd.DataFrame(raw)


def validate(
    raw: list[dict[str, Any]]
) -> bool:
    """Validate if the content of labelbox file is valid

    Args:
        raw (list[dict[str, Any]]): A list with the content from the
    labelbox JSON file.

    Returns:
        bool: True if the content data is valid.
    """
    if not isinstance(raw, list):
        raise TypeError('A list of dictionaries representing the raw data from LabelBox was expected.')

    for it in raw:
        if not all(i in it for i in EXPECTED_DATA) and 'Skipped' not in it:
            raise KeyError(f'The annotation values need to have `Skipped` or all of {EXPECTED_DATA} keys.')

    return True
