"""Common interface: every model predicts cumulative incidence on a monthly grid."""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

GRID = np.arange(1, 97)  # months 1..96 (8 years)


class CompetingRisksModel(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, X: pd.DataFrame, time: np.ndarray, event: np.ndarray) -> "CompetingRisksModel":
        """time: month index (>= 1); event: 0 censored, 1 operational, 2 withdrawn."""

    @abstractmethod
    def predict_cif(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        """{'operational': (n, len(GRID)), 'withdrawn': (n, len(GRID))}; CIF at the END of
        each month in GRID (column j = month GRID[j])."""

    def params(self) -> dict:
        return {}
