from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class Table:
    """Class for keeping track of table"""

    schema: set[str]
    values: dict
    dataframe: Optional[pd.DataFrame] = None
