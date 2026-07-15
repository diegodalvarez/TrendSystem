# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 16:46:53 2026

@author: Diego
"""


import os
import numpy as np
import pandas as pd

class EWMASignal:
    
    def __init__(self) -> None: 
        
        self.signal_path = os.getcwd()
        self.src_path    = os.path.abspath(os.path.join(self.signal_path, ".."))
        self.repo_path   = os.path.abspath(os.path.join(self.src_path, ".."))
        self.data_path   = os.path.join(self.repo_path, "data")
        self.signal_path = os.path.join(self.data_path, "Signals")
        
        self.trend_window  = 180
        self.hetero_window = 10
        self.vol_window    = 100
        self.vol_target    = 0.01
        
        self.products = ["future", "EquityIndex"]
       
    def _get_trend(
        self,
        df           : pd.DataFrame,
        trend_window : int,
        hetero_window: int,
        vol_window   : int   = 100,
        vol_target   : float = 0.01,
    ) -> pd.DataFrame:
    
    
        return (
            df.set_index("date")
              .sort_index()
              .assign(
                  px_rtn =lambda x: x.px.pct_change(),
                  px_diff=lambda x: x.px.diff(),
    
                  sigma=lambda x: (
                      x.px_diff
                       .ewm(span=hetero_window, adjust=False)
                       .std()
                       .replace(0, np.nan)
                  ),
    
                  R=lambda x: x.px_diff / x.sigma,
    
                  signal=lambda x: (
                      x.R
                       .ewm(span=trend_window, adjust=False)
                       .mean()
                  ),
                  lag_signal=lambda x: x.signal.shift(),
                  signal_rtn=lambda x: x.lag_signal * x.px_rtn,
                  weight    =lambda x: vol_target / (x.signal_rtn.ewm(span = vol_window, adjust = False).std()),
                  lag_weight=lambda x: x.weight.shift()))
        
    def get_signal(self, verbose: bool = True) -> None: 

        '''
        For now using Futures Data
        '''

        in_path  = os.path.join(self.data_path, "FuturesData", "PrepFuturesPX.parquet")
        out_path = os.path.join(self.signal_path, "DaoNguyenDerembleLemperiereBouchaudPotters.parquet")
        
        if verbose: 
            print("Getting EWMA Signal")
            
        if os.path.exists(out_path) == True: 
            if verbose: 
                print("Already have EWMA Signal")
                
            return None
        
        df_prep = (
            pd.read_parquet(path=in_path, engine="pyarrow")
              [["date", "ticker", "adj_val"]]
              .rename(columns={"adj_val": "px"})
              .dropna())
        
        df_out = (df_prep
              .groupby("ticker")
              .apply(
                  self._get_trend,
                  self.trend_window,
                  self.hetero_window,
                  self.vol_window,
                  self.vol_target,
              )
              .reset_index()
        )

        if verbose: 
            print("Saving data\n")
            
        df_out.to_parquet(path = out_path, engine = "pyarrow")
        
def main() -> None: 
            
    ewma_signal = EWMASignal()
    ewma_signal.get_signal()
    
if __name__ == "__main__": main()