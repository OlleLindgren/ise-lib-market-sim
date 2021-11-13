import datetime
import os
from abc import ABC
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from ise_market_sim import market_selection


class Observer(ABC):
    def update(market: pd.Series) -> None:
        pass


class Simulator:
    _observers: List[Observer]
    _market_history: pd.DataFrame
    _t: datetime.datetime
    __timeline: Iterable
    verbose: bool
    _sparse: bool
    scores: List

    def __init__(self,
                 observers: List[Observer],
                 market_history: pd.DataFrame,
                 *,
                 t0: datetime.datetime = None,
                 verbose: bool = False
                 ) -> None:
        self._observers = observers
        self._market_history = market_history
        self.__timeline = iter(market_history.index)
        self._t = next(self.__timeline)
        if t0:
            while self._t < t0:
                self._t = next(self.__timeline)
        self.verbose = verbose

        # Whether the datatype in market_history is sparse
        self._sparse = pd.api.types.is_sparse(self._market_history.loc[self._t, :])
        self.scores = []

    def step(self):

        # Get new market state
        market = self._market_history.loc[self._t, :].dropna()
        market = market[market > 0.]

        if self._sparse:
            market = market.sparse.to_dense()

        # Update every observer
        for observer in self._observers:
            observer.update(market)

        # Record everyone's scores
        self.scores.append([obs.score for obs in self._observers])

        self._t = next(self.__timeline)

    def print_state(self):
        strings = (f"{p.__class__.__name__}: {p.score:.3E}" for p in self._observers)
        print(' | '.join(strings))

    def save_scores(self):
        score_df = pd.DataFrame(
            index=[ix for _, ix in zip(self.scores, self._market_history.index)],
            columns=[obs.__class__.__name__ for obs in self._observers],
            data=self.scores
        )
        filename = (Path('.') / 'market_sim_scores.csv').absolute()
        score_df.to_csv(filename)
        print(f'saved scores to {filename}')

    def run(self):
        while True:
            try:
                self.step()
                if self.verbose:
                    self.print_state()
            except (StopIteration, KeyboardInterrupt):
                self.save_scores()
                break
