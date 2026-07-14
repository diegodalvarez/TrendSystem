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
        
        self.trend_window  = 100
        self.hetero_window = 10
        self.vol_window    = 100
        self.vol_target    = 0.1
        
        self.products = ["future", "EquityIndex"]
        
    def _get_trend(
            self, 
            df           : pd.DataFrame, 
            trend_window : int, 
            hetero_window: int, 
            vol_window   : int, 
            vol_target   : float) -> pd.DataFrame: 
        
        df_out = (df.set_index(
            "date").
            sort_index().
            assign(
                px_diff    = lambda x: x.px.diff(),
                px_dt      = lambda x: x.px_diff / x.px_diff.ewm(span = hetero_window, adjust = False).std(),
                signal     = lambda x: x.px_dt.ewm(span = trend_window, adjust = False).mean(),
                lag_signal = lambda x: x.signal.shift(),
                px_rtn     = lambda x: x.px.pct_change(),
                signal_rtn = lambda x: np.sign(x.lag_signal) * x.px_rtn,
                weight     = lambda x: x.signal_rtn * (vol_target / (x.signal_rtn.ewm(span = vol_window, adjust = False).std() * np.sqrt(252))),
                lag_weight = lambda x: x.weight.shift()))
        
        return df_out
        
    def get_signal(self, verbose: bool = True) -> None: 
        
        in_path  = os.path.join(self.data_path, "AllPx", "AllProducts.parquet")
        out_path = os.path.join(self.signal_path, "EWMASignal.parquet")
        
        if verbose: 
            print("Getting EWMA Signal")
            
        if os.path.exists(out_path) == True: 
            if verbose: 
                print("Calculating EWMA Signal")
                
            return None
        
        df_out = (pd.read_parquet(
            path = in_path, engine = "pyarrow").
            query("product == @self.products").
            query("variable == 'fx_adj_px'").
            drop(columns = ["variable"]).
            assign(group_var = lambda x: x.ticker + " " + x["product"]).
            groupby("group_var").
            apply(self._get_trend, self.trend_window, self.hetero_window, self.vol_window, self.vol_target))
        
        if verbose: 
            print("Saving data\n")
            
        df_out.to_parquet(path = out_path, engine = "pyarrow")
        
def main() -> None: 
            
    ewma_signal = EWMASignal()
    ewma_signal.get_signal()
    
if __name__ == "__main__": main()