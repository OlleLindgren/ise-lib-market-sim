import heapq
import random
from typing import List, Tuple

import numpy as np
import pandas as pd
from datacache import fingerprint


def select_from_df(df: pd.DataFrame, n) -> pd.DataFrame:

    na_rates = df.isna().sum() / df.shape[0]

    name_normal = lambda name: not any(set(range(10)) & set(name[-1]))

    n_tickers_many = min(10 * n, len(df.columns))

    # Find a selection of tickers with non-digit names, and with low rates of NA
    best_many = heapq.nsmallest(
        n_tickers_many,
        na_rates.index,
        key=lambda ticker: na_rates[ticker] if name_normal(ticker) else 2.0 + na_rates[ticker],
    )
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


# Dictionary to keep cached results from select_from_history in
SELECTION_CACHES = {}


def select_from_history(
    n: int, history: pd.DataFrame, *, horizon: int = 500, nmax_internal: int = 1000
) -> Tuple[List[str], pd.Series, pd.DataFrame]:
    """Select promising portfolio components from historical stock prices

    Args:
        n (int): The number of components (tickers/stocks) to select
        history (pd.DataFrame): Pricing history of a number of stocks
        horizon (int, optional): The discrete number of time steps to consider.
        Defaults to 500.
        nmax_internal (int, optional): The internal max number of tickers to consider
        in the computationally intensive (O(n²)) covariance estimation. Defaults to 1000.

    Returns:
        Tuple[List[str], pd.Series, pd.DataFrame]: selection, mean, cov of log returns
    """

    # ########################### CACHE ############################

    # Calculate fingerprint of input args
    f = fingerprint(n, list(history.index), list(history.columns), horizon, nmax_internal)

    # If cached, return cache
    if f in SELECTION_CACHES:
        return SELECTION_CACHES[f]

    # ######################### / CACHE ############################

    # Cutoff at horizon start
    selection = history.iloc[-horizon:, :] if history.shape[0] > horizon else history

    # Keep only those that are currently not NA
    selection = selection.loc[:, ~selection.iloc[-1, :].isna()]

    # Keep only those that are currently nonzero
    selection = selection.loc[:, selection.iloc[-1, :] > 1e-8]

    # Filter out those with way too much NA
    selection = selection.loc[:, selection.isna().sum() / selection.shape[0] < 0.3]

    # Covert to log returns
    selection = logify(selection)

    # Keep only those with positive mean returns
    selection = selection.loc[:, selection.mean() > 1e-8]

    # Calculate mean
    selection_mu = selection.mean()

    # Keep those with the best value at risk
    if selection.shape[1] > nmax_internal:
        selection_var = selection.var()
        score = selection_mu - 1.96 * selection_var.apply(np.sqrt)
        selection_cols = heapq.nlargest(
            nmax_internal, selection.columns, key=lambda col: score[col]
        )
        selection = selection[selection_cols]

    # Calculate covariance
    cov = selection.cov()

    # Keep the most negatively correlated. Narrow filter iteratively.
    columns = list(selection.columns)

    for n_select in (n * 5, n * 2, n):
        min_cov = cov.min()
        columns = heapq.nsmallest(n_select, columns, key=lambda col: min_cov[col])
        cov = cov.loc[columns, columns]

    # Calculate mu
    mu = selection_mu[columns]

    # ########################### CACHE ############################

    # Wipe cache to save memory
    # SELECTION_CACHES.clear()

    # Cache with fingerprint
    SELECTION_CACHES[f] = columns, mu, cov

    # ######################### / CACHE ############################

    return columns, mu, cov
