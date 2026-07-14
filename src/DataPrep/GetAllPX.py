# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 23:19:50 2026

@author: Diego
"""

import os
import numpy as np
import pandas as pd
import datetime as dt

class GetAllProducts: 
    
    def __init__(self) -> None: 
        
        self.data_prep_path = os.getcwd()
        self.src_path       = os.path.abspath(os.path.join(self.data_prep_path, ".."))
        self.repo_path      = os.path.abspath(os.path.join(self.src_path, ".."))
        self.data_path      = os.path.join(self.repo_path, "data")
        self.all_path       = os.path.join(self.data_path, "AllPx")
        
    def _get_fut(self) -> pd.DataFrame: 
        
        path   = os.path.join(self.data_path, "FuturesData", "PrepFuturesPX.parquet")
        df_out = (pd.read_parquet(
            path = path, engine = "pyarrow")
            [["date", "ticker", "adj_val", "CLEAN_PX"]].
            rename(columns = {
                "adj_val" : "fx_adj_px",
                "CLEAN_PX": "adj_pxs"}).
            melt(id_vars = ["date", "ticker"], value_name = "px").
            assign(product = "future"))
        
        return df_out
    
    def _prep_eq_idx(self, df: pd.DataFrame) -> pd.DataFrame: 
        
        df_raw   = df.sort_index()
        start_px = df_raw.query("date == date.min()").CLEAN_PX.item()
        df_out   = (df_raw.rename(
            columns = {"CLEAN_PX": "unadj_px"}).
            assign(
                adj_px      = lambda x: np.cumprod(1 + x.TotRtn) * start_px,
                fx_adj_px   = lambda x: x.adj_px * x.fx_val,
                fx_unadj_px = lambda x: x.unadj_px * x.fx_val).
        drop(columns = ["fx_ticker", "TotRtn", "fx_val"]))
        
        return df_out
        
    
    def _get_eq_idx(self, min_date: dt.date, max_date: dt.date) -> pd.DataFrame: 
        
        path = os.path.join(self.data_path, "EquityBenchmarks", "PrepEquityBenchmarkPXRtn.parquet")
        df_out = (pd.read_parquet(
            path = path, engine = "pyarrow").
            query("@min_date <= date <= @max_date").
            drop(columns = ["PX_LAST"]).
            set_index("date").
            query("ticker == ticker.min()").
            groupby("ticker").
            apply(self._prep_eq_idx).
            reset_index().
            melt(id_vars = ["date", "ticker"], value_name = "px").
            assign(product = "EquityIndex"))
        
        return df_out
        
    def get_all(self, verbose: bool = True) -> None:
        
        out_path = os.path.join(self.all_path, "AllProducts.parquet")
        
        if verbose: 
            print("Getting All Products Prices")
        
        if os.path.exists(out_path) == True: 
            if verbose: 
                print("Already have data\n")
                
            return None
        
        if verbose: 
            print("Getting Futures and Equity Indices Data")
        
        df_fut    = self._get_fut()
        max_date  = df_fut.date.max()
        min_date  = df_fut.date.min()
        df_eq_idx = self._get_eq_idx(min_date, max_date)
        
        df_out = pd.concat([df_fut, df_eq_idx])
        
        if verbose: 
            print("Saving data\n")
            
        df_out.to_parquet(path = out_path, engine = "pyarrow")
        
def main() -> None:
            
    all_products = GetAllProducts()
    all_products.get_all()
    
if __name__ == "__main__": main()