# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 11:17:38 2026

@author: Diego
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from   sklearn.decomposition import PCA

from tqdm import tqdm
tqdm.pandas()

class OLSFit:
    
    def __init__(self) -> None: 
        
        self.parent_path = os.getcwd()
        self.root_path   = os.path.abspath(os.path.join(self.parent_path,".."))
        self.data_path   = os.path.join(self.root_path, "data")
        self.fut_path    = os.path.join(self.data_path, "FuturesData")
        self.gen_path    = os.path.join(self.data_path, "GenericSignal")
        self.ols_path    = os.path.join(self.data_path, "OLS")
        
        if os.path.exists(self.ols_path) == False:
            os.makedirs(self.ols_path)
        
    def _risk_adj_rtn(
            self, 
            vol_target: float = 0.1,
            vol_window: int   = 100,
            verbose   : bool  = True) -> None:
        
        if verbose:
            print("Getting Risk Adjusted Returns")
        
        out_path = os.path.join(self.ols_path, "RiskAdjReturns.parquet")
        if os.path.exists(out_path) == True:
            if verbose: 
                print("Already have data\n")
                
            return None
        
        if verbose: 
            print("Calculating Risk Adjusted Returns")
        
        fut_path = os.path.join(self.fut_path, "PrepFuturesPX.parquet")
        df_out   = (pd.read_parquet(
            path = fut_path, engine = "pyarrow").
            pivot(index = "date", columns = "ticker", values = "px_val").
            pct_change().
            apply(lambda x: 
                  x * (
                      vol_target / 
                      (x.ewm(span = vol_window, adjust = False).std() * 
                       np.sqrt(252)))).
            reset_index().
            melt(id_vars = "date", value_name = "rtn").
            dropna())
            
        if verbose: 
            print("Saving data\n")
            
        df_out.to_parquet(path = out_path, engine = "pyarrow")
        
    def _is_ols(self, df: pd.DataFrame) -> pd.DataFrame: 
        
        df_out = (sm.OLS(
            endog = df.vol_rtn,
            exog  = sm.add_constant(df.lag_signal)).
            fit().
            fittedvalues.
            to_frame(name = "predicted").
            merge(right = df, how = "inner", on = ["date"]))
        
        return df_out
        
    def in_sample_single_ols_fit(
            self):
        
        rtn_path    = os.path.join(self.ols_path, "RiskAdjReturns.parquet")
        signal_path = os.path.join(self.gen_path, "GenericSignal.parquet")
        out_path    = os.path.join(self.ols_path, "OLSSignal.parquet")
        
        df_rtn = (pd.read_parquet(
            path = rtn_path, engine = "pyarrow").
            rename(columns = {"rtn": "vol_rtn"}))
        
        df_lag_signal = (pd.read_parquet(
            path = signal_path, engine = "pyarrow").
            pivot(index = "date", columns = "ticker", values = "signal").
            shift().
            reset_index().
            melt(id_vars = "date", value_name = "lag_signal").
            dropna())
        
        display(df_lag_signal.merge(
            right = df_rtn, how = "inner", on = ["date", "ticker"]).
            set_index("date").
            groupby("ticker").
            progress_apply(lambda group: self._is_ols(group)).
            reset_index())
        
        
ols_fit = OLSFit()
#ols_fit._risk_adj_rtn()
ols_fit.in_sample_single_ols_fit()