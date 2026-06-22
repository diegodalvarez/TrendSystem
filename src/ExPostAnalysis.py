# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 12:41:36 2026

@author: Diego
"""

import os
import numpy as np
import pandas as pd

class BenchmarkGenerator:
    
    def __init__(self) -> None: 
        
        self.src_path  = os.getcwd()
        self.root_path = os.path.abspath(os.path.join(self.src_path, ".."))
        self.data_path = os.path.join(self.root_path, "data")
        self.fut_path  = os.path.join(self.data_path, "FuturesData")
        
        self.benchmark_path = os.path.join(self.data_path, "Benchmarks")
        
    def _get_vol(self, df: pd.DataFrame, vol_lookback: int) -> pd.DataFrame: 
        
        df_out = (df.assign(
            vol = lambda x: (x.rtn.ewm(
                span = vol_lookback, adjust = False).std()) * np.sqrt(252),
            lag_vol = lambda x: x.vol.shift()))
        
        return df_out
        
    def get_long_benchmark(
            self, 
            vol_lookback: int  = 100,
            verbose     : bool = True) -> None: 
        
        '''
        A function for generating a long only benchmark
        '''
        
        out_path = os.path.join(
            self.benchmark_path, "LongOnlyIndividualReturns.parquet")
        
        if verbose: 
            print("Getting individual long-only returns")
        
        if os.path.exists(out_path) == True: 
            if verbose: 
                print("Already have data\n")
                
            return None
        
        if verbose: 
            print("Generating Long only vol targeted weights")
        
        in_path = os.path.join(self.fut_path, "PrepFuturesPX.parquet")
        df_out  = (pd.read_parquet(
            path = in_path, engine = "pyarrow").
            pivot(index = "date", columns = "ticker", values = "px_val").
            pct_change().
            reset_index().
            melt(id_vars = "date", value_name = "rtn").
            dropna().
            set_index("date").
            query("ticker == ticker.min()").
            groupby("ticker").
            apply(self._get_vol, vol_lookback).
            reset_index().
            dropna())
        
        if verbose: 
            print("Saving data\n")
            
        df_out.to_parquet(path = out_path, engine = "pyarrow")
        
def main() -> None: 
        
    benchmark_generator = BenchmarkGenerator()
    benchmark_generator.get_long_benchmark()
    
if __name__ == "__main__": main()