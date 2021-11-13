from abc import ABC, abstractmethod
import datetime
import pandas as pd
import numpy as np

from ise_efficient_frontier import min_risk, max_sharpe
from .market_selection import prep_select


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

    @abstractmethod
    def update_weights(self) -> None:
        """Update portfolio weights"""

    def record_market(self, market: pd.Series) -> None:
        """Record state of market"""
        if self.market_history is None:
            self.market_history = pd.DataFrame(
                columns=list(market.index),
                data=[market.values],
                index=[market.name]
            )
        else:
            self.market_history.loc[market.name, market.index] = market.values

    def update(self, market: pd.Series):
        """Update bot with new market state, and perform bot actions"""

        # Record state of market
        self.record_market(market)

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

        self.weights[market < 1e-8] = 0.

        # Compute adjusted weights from absolute weights.
        # This is to compensate for the different magnitudes
        # of the prices of different securities.
        self.__adj_weights = self.k * self.weights.divide(market).fillna(0.).replace(np.inf, 0.)

        # Set k to the non-invested ratio.
        self.k *= 1. - self.weights.sum()


class MinRiskBot(TraderBot):
    """A bot that seeks to construct a portfolio on the efficient frontier,
    with the lowest possible statistical risk."""

    def update_weights(self) -> None:
        if len(self.market_history.index) > 50:
            data_selection = prep_select(self.market_history, n=100)
            mu = data_selection.mean()
            cov = data_selection.cov()
            self.weights = pd.Series(data=min_risk(mu, cov), index=data_selection.columns)
            if (total_weight := self.weights.sum()) > 1.:
                self.weights *= .99 / total_weight
            elif total_weight < 0.:
                self.weights *= 0.


class MaxSharpeBot(TraderBot):
    """A bot that seeks to construct a portfolio on the efficient frontier,
    with the highest possible sharpe ratio."""

    def update_weights(self) -> None:
        if len(self.market_history.index) > 50:
            data_selection = prep_select(self.market_history, n=100)
            mu = data_selection.mean()
            cov = data_selection.cov()
            self.weights = pd.Series(data=max_sharpe(mu, cov), index=data_selection.columns)
            if (total_weight := self.weights.sum()) > 1.:
                self.weights *= .99 / total_weight
            elif total_weight < 0.:
                self.weights *= 0.
