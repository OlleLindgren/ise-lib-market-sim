from abc import ABC, abstractmethod
import heapq
import pandas as pd
import numpy as np

from ise_efficient_frontier import min_risk, max_sharpe
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from .market_selection import prep_select, select_from_history


class TraderBot(ABC):

    # A set of weights that must always sum to between 0 and 1.
    # If the weights do not sum to 1, the remainder represents
    # cash.
    weights: pd.Series
    __adj_weights: pd.Series

    # A constant that represents how much the portfolio has grown. Starts at 1.
    k: float
    score: float

    # The saved historical state of the market
    market_history: pd.DataFrame = None

    def __init__(self) -> None:
        self.k = 1.
        self.weights = pd.Series(data=[], index=[])
        self.__adj_weights = pd.Series(data=[], index=[])

    def alive(self) -> bool:
        """Whether the bot has anything left to invest."""
        return self.k + self.weights.sum() > 1e-16

    @abstractmethod
    def update_weights(self) -> None:
        """Update portfolio weights"""

    @classmethod
    def record_market(cls, market: pd.Series) -> None:
        """Record state of market"""
        if cls.market_history is None:
            cls.market_history = pd.DataFrame(
                columns=list(market.index),
                data=[market.values],
                index=[market.name]
            )
        elif market.name in cls.market_history.index:
            # If already recorded by other instance, return
            return
        else:
            cls.market_history.loc[market.name, market.index] = market.values

    def update(self, market: pd.Series):
        """Update bot with new market state, and perform bot actions"""

        # Record state of market
        self.record_market(market)

        # If there are tickers in old weights that are not present in market, we retain these.
        index_diff = self.weights.index.difference(market.index)

        if not index_diff.empty:
            retained_adj_weights = self.__adj_weights[index_diff]
            retained_weights = self.weights[index_diff]

        # Sell
        self.k += market.multiply(self.__adj_weights, fill_value=0.).sum()

        # Update score
        self.score = self.k

        # Get new weights
        self.update_weights()

        if self.weights.sum() > 1.:
            raise Exception("Weights too large")
        if self.weights.sum() < 0.:
            raise Exception("Weights too small")
        if self.weights.min() < 0.:
            raise Exception("Negative weights are not allowed")

        # Compute adjusted weights from absolute weights.
        # This is to compensate for the different magnitudes
        # of the prices of different securities.
        self.__adj_weights = self.k * self.weights.divide(market).fillna(0.).replace(np.inf, 0.)

        if not index_diff.empty:
            k_retained_weights = retained_weights.sum()
            self.__adj_weights = pd.concat([self.__adj_weights * (1. - k_retained_weights), retained_adj_weights])
            self.weights = pd.concat([self.weights * (1. - k_retained_weights), retained_weights])

        # Set k to the non-invested ratio.
        self.k *= 1. - self.weights.sum()


class MinRiskBot(TraderBot):
    """A bot that seeks to construct a portfolio on the efficient frontier,
    with the lowest possible statistical risk."""

    def update_weights(self) -> None:
        if len(self.market_history.index) > 50:

            tickers, mu, cov = select_from_history(n=30, history=self.market_history)

            self.weights = pd.Series(data=min_risk(mu, cov), index=tickers)

            if (total_weight := self.weights.sum()) > 1.:
                self.weights *= .99 / total_weight
            elif total_weight < 0.:
                self.weights *= 0.


class MaxSharpeBot(TraderBot):
    """A bot that seeks to construct a portfolio on the efficient frontier,
    with the highest possible sharpe ratio."""

    def update_weights(self) -> None:
        if len(self.market_history.index) > 50:

            tickers, mu, cov = select_from_history(n=30, history=self.market_history)

            self.weights = pd.Series(data=max_sharpe(mu, cov), index=tickers)

            if (total_weight := self.weights.sum()) > 1.:
                self.weights *= .99 / total_weight
            elif total_weight < 0.:
                self.weights *= 0.


class RandomBot(TraderBot):
    """A bot that generates uniform random weights"""

    def update_weights(self) -> None:
        if len(self.market_history.index) > 3:
            self.weights = pd.Series(
                data=np.random.dirichlet(np.ones(self.market_history.shape[1])),
                index=self.market_history.columns
            )
            if (total_weight := self.weights.sum()) > 1.:
                self.weights *= .99 / total_weight
            elif total_weight < 0.:
                self.weights *= 0.


class UniformBot(TraderBot):
    """A bot that generates uniform weights."""

    def update_weights(self) -> None:
        if len(self.market_history.index) > 3:
            self.weights = pd.Series(
                data=np.ones(self.market_history.shape[1]) / self.market_history.shape[1],
                index=self.market_history.columns
            )
            if (total_weight := self.weights.sum()) > 1.:
                self.weights *= .99 / total_weight
            elif total_weight < 0.:
                self.weights *= 0.
