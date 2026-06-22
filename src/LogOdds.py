# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 10:26:35 2026

@author: Diego
"""

import warnings
import numpy as np
import pandas as pd
from   typing import Tuple

from scipy.stats import (
    fisher_exact,
    chi2_contingency,
    norm)

class LogOddsTesting:
    
    def __init__(self) -> None: 
        pass

    def odds_ratio_tests(
        self,
        df           : pd.DataFrame,
        signal_col   : str,
        return_col   : str,
        signal_thresh: float = 0,
        return_thresh: float = 0,
        show_warning : bool = True) -> Tuple[pd.DataFrame, pd.Series]:
    
        if show_warning:
            warnings.warn(
                "This function assumes that the signal is already lagged "
                "relative to the realized return. Results may be biased if "
                "the signal contains look-ahead information.",
                UserWarning,
                stacklevel=2,
            )
        
        df_tmp = (
            df[[signal_col, return_col]]
            .dropna()
            .copy()
        )
    
    
        signal_up = (
            df_tmp[signal_col]
            > signal_thresh).astype(int)
    
        rtn_up = (
            df_tmp[return_col]
            > return_thresh).astype(int)
    
        # ---------------------------
        # contingency table
        #
        #           rtn up   rtn down
        # sig up      a         b
        # sig down    c         d
        # ---------------------------
    
        table = (
            pd.crosstab(
                signal_up,
                rtn_up
            )
            .reindex(
                index=[1,0],
                columns=[1,0],
                fill_value=0
            )
        )
    
        a,b,c,d = table.to_numpy().ravel()
    
        # ---------------------------
        # Odds ratio
        # ---------------------------
    
        if min(a,b,c,d) == 0:
    
            odds_ratio = np.nan
            log_or     = np.nan
            se         = np.nan
            z          = np.nan
            wald_p     = np.nan
    
        else:
    
            odds_ratio = (a*d)/(b*c)
    
            log_or = np.log(odds_ratio)
    
            se = np.sqrt(
                1/a +
                1/b +
                1/c +
                1/d
            )
    
            z = log_or / se
    
            wald_p = 2 * (
                1 - norm.cdf(np.abs(z))
            )
    
        # ---------------------------
        # Fisher
        # ---------------------------
    
        fisher_or, fisher_p = fisher_exact(table)
    
        # ---------------------------
        # Chi square
        # ---------------------------
    
        chi2, chi2_p, dof, expected = (
            chi2_contingency(table)
        )
    
        # ---------------------------
        # Results
        # ---------------------------
    
        results = pd.Series({
    
            "n_obs" : len(df_tmp),
    
            "TP" : a,
            "FP" : b,
            "FN" : c,
            "TN" : d,
    
            "odds_ratio" : odds_ratio,
            "log_odds_ratio" : log_or,
    
            "fisher_p" : fisher_p,
    
            "chi2" : chi2,
            "chi2_p" : chi2_p,
    
            "wald_z" : z,
            "wald_p" : wald_p,
    
        })
    
        return table, results
    
 
# test
import os
    
src_path    = os.getcwd()
root_path   = os.path.abspath(os.path.join(src_path, ".."))
data_path   = os.path.join(root_path, "data")
price_path  = os.path.join(data_path, "FuturesData", "PrepFuturesPX.parquet")
signal_path = os.path.join(data_path, "GenericSignal", "GenericSignal.parquet")   

ticker = "es1_index"

df_rtn = (pd.read_parquet(
    path = price_path, engine = "pyarrow").
    pivot(index = "date", columns = "ticker", values = "px_val").
    pct_change().
    reset_index().
    melt(id_vars = "date", value_name = "rtn").
    dropna().
    query("ticker == @ticker"))

df_signal = (pd.read_parquet(
    path = signal_path, engine = "pyarrow").
    query("ticker == @ticker").
    set_index("date").
    sort_index().
    assign(signal = lambda x: x.signal.shift()).
    dropna())

df_input = (df_rtn.merge(
    right = df_signal, how = "inner", on = ["date", "ticker"]))

table, results = LogOddsTesting().odds_ratio_tests(df_input, "signal", "rtn")

print(table, results)