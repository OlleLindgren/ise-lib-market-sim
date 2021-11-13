from abc import ABC
import datetime
from typing import Iterable, List
from ise_market_sim import market_selection

import pandas as pd

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

    def step(self):
        self._t = next(self.__timeline)

        # Get new market state
        market = self._market_history.loc[self._t, :].dropna()
        market = market[market > 0.]

        if self._sparse:
            market = market.sparse.to_dense()

        # Update every observer
        for observer in self._observers:
            observer.update(market)

    def print_state(self):
        strings = (f"{p.__class__.__name__}: {p.score:.3E}" for p in self._observers)
        print(' | '.join(strings))

    def run(self):
        while True:
            try:
                self.step()
                if self.verbose:
                    self.print_state()
            except StopIteration:
                break
