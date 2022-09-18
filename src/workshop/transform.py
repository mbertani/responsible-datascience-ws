# Inspired by: https://calmcode.io/pandas-pipe/calm.html

import logging
import datetime as dt
import pandas as pd

from functools import wraps
from typing import List

logger = logging.getLogger(__name__)

def log_step(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        tic = dt.datetime.now()
        result = func(*args, **kwargs)
        time_taken = str(dt.datetime.now() - tic)
        print(f"Executed step {func.__name__} shape={result.shape} took {time_taken}s")
        return result
    return wrapper


@log_step
def start_pipeline(dataf):
    return dataf.copy()

@log_step
def cols_to_int(frame: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Convert a set of columns in a dataframe to int

    Args:
        frame (pd.DataFrame): the input dataframe.
        cols (List[str]): a list of columns to convert.

    Returns:
        pd.DataFrame: the dataframe with converted columns.
    """
    assert set(cols).issubset(frame.columns)

    for column in cols:
        frame[column] = frame[column].apply(int)
    return frame

@log_step
def cols_to_str(frame: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Convert a set of columns in a dataframe to str

    Args:
        frame (pd.DataFrame): the input dataframe.
        cols (List[str]): a list of columns to convert.

    Returns:
        pd.DataFrame: the dataframe with converted columns.
    """
    assert set(cols).issubset(frame.columns)

    for column in cols:
        frame[column] = frame[column].apply(str)
    return frame