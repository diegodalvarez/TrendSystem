# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 20:14:28 2026

@author: Diego
"""

import os
import numpy as np
import pandas as pd

class BacktestingTools:
    
    def __init__(self) -> None: 
        pass
    
    def apply_signal(
            self, 
            df_rtn_longer    : pd.DataFrame, 
            df_signal_longer : pd.DataFrame, 
            port_name        : str   = "port",
            vol_lookback     : int   = 100, 
            vol_target       : float = 0.2,
            vol_shifter      : int   = 0,
            signal_shifter   : int   = 1,
            outlier_threshold: float = 2) -> pd.DataFrame:
        
        df_lag_signal = (df_signal.pivot(
            index = "date", columns = "ticker", values = "signal").
            shift(signal_shifter).
            reset_index().
            melt(id_vars = "date", value_name = "lag_signal").
            dropna())
        
        df_out = (df_rtn.merge(
            right = df_lag_signal, how = "inner", on = ["date", "ticker"]).
            assign(signal_rtn = lambda x: np.sign(x.lag_signal) * x.rtn).
            pivot(index = "date", columns = "ticker", values = "signal_rtn").
            apply(lambda x: (
                x * (
                    vol_target / 
                    (x.ewm(span = vol_lookback).std().shift(vol_shifter) * 
                     np.sqrt(252))))).
            apply(
                lambda x: np.where(np.abs(x) > outlier_threshold, np.nan, x)).
            mean(axis = 1).
            to_frame(name = port_name).
            dropna())
        
        return df_out
        
# Testing functionality 

src_path    = os.getcwd()
root_path   = os.path.abspath(os.path.join(src_path, ".."))
data_path   = os.path.join(root_path, "data")
price_path  = os.path.join(data_path, "FuturesData", "PrepFuturesPX.parquet")
signal_path = os.path.join(data_path, "GenericSignal", "GenericSignal.parquet")   

df_rtn = (pd.read_parquet(
    path = price_path, engine = "pyarrow").
    pivot(index = "date", columns = "ticker", values = "px_val").
    pct_change().
    reset_index().
    melt(id_vars = "date", value_name = "rtn").
    dropna())

df_signal = (pd.read_parquet(
    path = signal_path, engine = "pyarrow"))

df_backtest = BacktestingTools().apply_signal(df_rtn, df_signal)