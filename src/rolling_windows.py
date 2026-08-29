"""
Rolling-window definitions for out-of-sample evaluation.

Fifteen consecutive one-year test windows spanning 2006–2020 are used.  Each
window has a 4-year training period immediately preceding the test year.
"""

from __future__ import annotations

import pandas as pd
from typing import List, Tuple

WINDOWS: List[Tuple[int, int, int]] = [
    # (train_start_year, train_end_year, test_year)
    (2002, 2005, 2006),
    (2003, 2006, 2007),
    (2004, 2007, 2008),
    (2005, 2008, 2009),
    (2006, 2009, 2010),
    (2007, 2010, 2011),
    (2008, 2011, 2012),
    (2009, 2012, 2013),
    (2010, 2013, 2014),
    (2011, 2014, 2015),
    (2012, 2015, 2016),
    (2013, 2016, 2017),
    (2014, 2017, 2018),
    (2015, 2018, 2019),
    (2016, 2019, 2020),
]


def get_window_dates(
    train_start_year: int,
    train_end_year: int,
    test_year: int,
) -> Tuple[str, str, str, str]:
    """
    Convert integer years to ISO date strings.

    Returns
    -------
    tuple of (train_start, train_end, test_start, test_end)
    """
    return (
        f"{train_start_year}-01-01",
        f"{train_end_year}-12-31",
        f"{test_year}-01-01",
        f"{test_year}-12-31",
    )


def split_series(
    rv: pd.Series,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
) -> Tuple[pd.Series, pd.Series]:
    """
    Slice *rv* into training and test sub-series.

    Parameters
    ----------
    rv : pd.Series
        Full realized variance series with a ``DatetimeIndex``.
    train_start, train_end : str
        ISO date bounds for the training period (inclusive).
    test_start, test_end : str
        ISO date bounds for the test period (inclusive).

    Returns
    -------
    train : pd.Series
    test : pd.Series
    """
    train = rv.loc[train_start:train_end]
    test = rv.loc[test_start:test_end]
    return train, test
