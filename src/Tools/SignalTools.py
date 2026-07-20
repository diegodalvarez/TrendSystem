# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 22:04:37 2026

@author: Diego
"""

import numpy as np
import pandas as pd

class SignalAnalysis:
    
    def autocovariance(self,
                       r: pd.Series,
                       max_lag: int = 200) -> pd.Series:
        """
        Sample autocovariance function.

        Parameters
        ----------
        r : pd.Series
            Return (or price change) series.
        max_lag : int
            Maximum lag.

        Returns
        -------
        pd.Series
            Autocovariance indexed by lag.
        """

        r = r.dropna().astype(float)
        r = r - r.mean()

        C = {}

        x = r.to_numpy()

        for lag in range(max_lag + 1):

            if lag == 0:

                C[lag] = np.mean(x**2)

            else:

                C[lag] = np.mean(
                    x[:-lag] * x[lag:]
                )

        return pd.Series(C)


    def signature_volatility(self,
                             r: pd.Series,
                             max_tau: int = 500) -> pd.DataFrame:
        """
        Compute the signature plot from Dao et al. (2016).

        Returns:

            sigma²(tau) / sigma²(1)

        where:

            sigma²(tau)
            =
            sigma²(1)
            +
            (2/tau)
            * Σ_{u=1}^{tau-1} (tau-u) C(u)

        Parameters
        ----------
        r : pd.Series
            Return series.

        max_tau : int
            Maximum aggregation horizon.

        Returns
        -------
        pd.DataFrame
            Columns:
                tau
                variance
                signature
        """

        C = self.autocovariance(
            r=r,
            max_lag=max_tau
        )

        variance_1 = C.iloc[0]

        out = []

        for tau in range(1, max_tau + 1):

            if tau == 1:

                variance_tau = variance_1

            else:

                u = np.arange(1, tau)

                cov_sum = np.sum(
                    (tau - u) * C.iloc[u].values
                )

                variance_tau = (
                    variance_1
                    + (2 / tau) * cov_sum
                )

            out.append(
                {
                    "tau": tau,
                    "variance": variance_tau,
                    "signature": variance_tau / variance_1
                }
            )

        return pd.DataFrame(out)