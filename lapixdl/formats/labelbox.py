
from __future__ import annotations

from typing import Any
import json
import pandas as pd


def load(
    filename: str,
) -> pd.DataFrame:
    with open(filename) as f:
        raw = json.load(f)

    validate(raw)

    return pd.DataFrame(raw)


def validate(
    raw: list[dict[str, Any]]
) -> bool:
    if not isinstance(raw, list):
        raise TypeError('A list of dictionaries representing the raw data from LabelBox was expected.')

    expected_data = set({'ID', 'External ID', 'Reviews', 'Label'})
    for it in raw:
        if not all(i in it for i in expected_data):
            if 'Skipped' not in it:
                raise KeyError(f'Not found expected values need to have `Skipped` or {expected_data}')

    return True
