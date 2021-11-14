from abc import ABC, abstractmethod
from functools import reduce
from typing import List, Tuple
import pandas as pd
import numpy as np

from ise_efficient_frontier import min_risk, max_sharpe
from .market_selection import logify, select_from_history


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

    # Minimum historical data to start trading
    min_data_samples: int
    # Number of tickers to actively keep in portfolio
    n_trading_tickers: int
    # Length of historical data to consider for evaluating stocks to invest in
    historical_trading_horizon: int

    def __init__(self) -> None:
        self.k = 1.
        self.weights = pd.Series(data=[], index=[])
        self.__adj_weights = pd.Series(data=[], index=[])
        self.name = self.__class__.__name__

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

    def data_selection_function(self) -> Tuple[List[str], pd.Series, pd.DataFrame]:
        """Function for selecting data"""
        return select_from_history(n=self.n_trading_tickers, history=self.market_history,
                                   horizon=self.historical_trading_horizon)

    def weight_function(self, tickers, mu, cov) -> pd.Series:
        """Function for weighting tickers"""

    def update_weights(self) -> None:
        if len(self.market_history.index) > self.min_data_samples:

            tickers, mu, cov = self.data_selection_function()

            if len(tickers) == 0:
                return

            self.weights = self.weight_function(tickers, mu, cov)

            if (total_weight := self.weights.sum()) > 1.:
                self.weights *= .99 / total_weight
            elif total_weight < 0.:
                self.weights *= 0.


class MinRiskBot(TraderBot):
    """A bot that seeks to construct a portfolio on the efficient frontier,
    with the lowest possible statistical risk."""

    def __init__(self, *,
                 n_trading_tickers = 30,
                 historical_trading_horizon = 100,
                 min_data_samples = 30) -> None:
        super().__init__()

        self.min_data_samples = min_data_samples
        self.n_trading_tickers = n_trading_tickers
        self.historical_trading_horizon = historical_trading_horizon

        self.name = f"{self.__class__.__name__}({n_trading_tickers}, {historical_trading_horizon}, {min_data_samples})"

    def weight_function(self, tickers, mu, cov) -> np.array:
        """Function for weighting tickers"""
        return pd.Series(index=tickers, data=min_risk(mu, cov))


class MaxSharpeBot(TraderBot):
    """A bot that seeks to construct a portfolio on the efficient frontier,
    with the highest possible sharpe ratio."""

    def __init__(self, *,
                 n_trading_tickers = 30,
                 historical_trading_horizon = 100,
                 min_data_samples = 30) -> None:
        super().__init__()

        self.min_data_samples = min_data_samples
        self.n_trading_tickers = n_trading_tickers
        self.historical_trading_horizon = historical_trading_horizon

        self.name = f"{self.__class__.__name__}({n_trading_tickers}, {historical_trading_horizon}, {min_data_samples})"

    def weight_function(self, tickers, mu, cov) -> np.array:
        """Function for weighting tickers"""
        return pd.Series(index=tickers, data=max_sharpe(mu, cov))


class RandomBot(TraderBot):
    """A bot that generates uniform random weights"""

    def __init__(self, n_trading_tickers = 30) -> None:
        super().__init__()

        self.n_trading_tickers = n_trading_tickers

        self.name = f"{self.__class__.__name__}({n_trading_tickers})"

    # Both ideally 0, but now 3 so the math doesn't implode
    min_data_samples = 3
    historical_trading_horizon = 3

    def weight_function(self, tickers, mu, cov) -> np.array:
        """Function for weighting tickers"""
        return pd.Series(index=tickers, data=np.random.dirichlet(np.ones(len(tickers))))


class UniformBot(TraderBot):
    """A bot that generates uniform weights."""

    def __init__(self, n_trading_tickers = 30) -> None:
        super().__init__()

        self.n_trading_tickers = n_trading_tickers

        self.name = f"{self.__class__.__name__}({n_trading_tickers})"

    # Both ideally 0, but now 3 so the math doesn't implode
    min_data_samples = 3
    historical_trading_horizon = 3

    def weight_function(self, tickers, mu, cov) -> np.array:
        """Function for weighting tickers"""
        return pd.Series(index=tickers, data=np.ones(len(tickers)) / len(tickers))


class MasterBot(TraderBot):
    """A bot that manages other bots"""

    min_data_samples = 30
    historical_trading_horizon = 300

    _bots: List[TraderBot]

    @property
    def bots(self) -> List[TraderBot]:
        def bot_iterator():
            for bot in self._bots:
                yield bot
                if hasattr(bot, 'bots'):
                    yield from bot.bots
        return list(bot_iterator())

    def __init__(self, bots: List[TraderBot]) -> None:
        super().__init__()

        self._bots = bots

        if self in self.bots:
            raise ValueError("MasterBot cannot be sub-bot of itself")

        self.history = pd.DataFrame(columns=[f"bot_{i}" for i in range(len(self.bots))], data=[])

        self.name = f"{self.__class__.__name__}([{', '.join(bot.name for bot in self.bots)}])"

    def data_selection_function(self) -> Tuple[List[str], pd.Series, pd.DataFrame]:
        """Overload and in effect disable data_selection_function for this version"""
        return [None], [None], [None]

    def weight_function(self, tickers, mu, cov) -> np.array:

        if len(self.history.index) > 10:

            log_history = logify(self.history)
            mu = log_history.mean()
            cov = log_history.cov()
            bot_weights = max_sharpe(mu, cov)
            
            # Return weighted sum of slave weights that slave bots use
            return reduce(
                lambda a, b: a.add(b, fill_value=0.),
                (bot.weights * weight for bot, weight in zip(self.bots, bot_weights)))

        return pd.Series()

    def update(self, market: pd.Series):
        self.history.loc[market.name, :] = [bot.score for bot in self.bots]
        super().update(market)
