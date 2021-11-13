import heapq
import random
import numpy as np
import pandas as pd

def select_from_df(df: pd.DataFrame, n) -> pd.DataFrame:
        
    na_rates = df.isna().sum() / df.shape[0]
    name_normal = lambda name: not any(set(range(10)) & set(name[-1]))

    best_many = heapq.nsmallest(min(10*n, len(df.columns)), na_rates.index, key=lambda ticker: na_rates[ticker] if name_normal(ticker) else 2.)
    best_few = random.choices(best_many, k=n)

    _selection = df[best_few]
    return _selection

def logify(df: pd.DataFrame) -> pd.DataFrame:
    """Apply ffill(5), dropna, diff and again dropna on df"""
    _logification = df.ffill(limit=5).bfill(limit=5).dropna()
    _logification = _logification.apply(np.log).diff().dropna()
    return _logification

def prep_select(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return logify(select_from_df(df, n))
