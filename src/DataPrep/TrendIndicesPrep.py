# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 20:28:13 2026

@author: Diego
"""

import os
import numpy as np
import pandas as pd

class TrendIndicesDataPrep:
    
    def __init__(self, db_path: str) -> None: 
        
        self.dsrc_path = os.getcwd()
        self.src_path  = os.path.abspath(os.path.join(self.dsrc_path, ".."))
        self.repo_path = os.path.abspath(os.path.join(self.src_path, ".."))
        self.data_path = os.path.join(self.repo_path, "data")
        self.idx_path  = os.path.join(self.data_path, "TrendIndices")
        
        self.db_path = db_path
    
    def _get_barclays(self, df_tickers: pd.DataFrame) -> pd.DataFrame: 
        
        tickers = (df_tickers.query(
            "File == 'Barclays'").
            Ticker.
            drop_duplicates().
            sort_values().
            to_list())
        
        path   = os.path.join(self.db_path, "PX", "Barclays.parquet")
        df_out = (pd.read_parquet(
            path = path, engine = "pyarrow").
            assign(
                tmp      = lambda x: x.variable.str.split(" ").str.len(),
                security = lambda x: np.where(x.tmp == 1, x.variable + " Index", x.variable)).
            drop(columns = ["variable", "tmp"]).
            drop_duplicates().
            rename(columns = {"value": "PX_LAST"}).
            query("security == @tickers").
            rename(columns = {"Date": "date"}))
        
        return df_out
    
    def _get_px(self, df_tickers: pd.DataFrame, file_name: str) -> pd.DataFrame: 
        
        tickers = (df_tickers.query(
            "File == @file_name").
            Ticker.
            drop_duplicates().
            sort_values().
            to_list())
        
        path   = os.path.join(self.db_path, "PX", "{}.parquet".format(file_name))
        df_out = (pd.read_parquet(
            path = path, engine = "pyarrow").
            query("security == @tickers"))
        
        return df_out
    
    def get_raw_trend_indices(
            self, 
            verbose : bool = True) -> None: 
        
        out_path     = os.path.join(self.idx_path, "RawTrendIndices.parquet")
        tickers_path = os.path.join(self.idx_path, "TrendIndicesGuide.xlsx")
        df_tickers   = pd.read_excel(io = tickers_path, sheet_name = "TrendIndices")
        
        if os.path.exists(out_path) == True: 
            if verbose: 
                print("Already have Trend Indices data\n")
                
            return None
        
        if verbose: 
            print("Getting Trend Indices")
        
        df_barclays      = self._get_barclays(df_tickers)
        df_bloomberg     = self._get_px(df_tickers, "BloombergTrend")
        df_bnp           = self._get_px(df_tickers, "BNP")
        df_bofa          = self._get_px(df_tickers, "BofA")
        df_citi          = self._get_px(df_tickers, "CITI1")
        df_commod        = self._get_px(df_tickers, "CommodTrend")
        df_db_trend      = self._get_px(df_tickers, "DBTrend")
        df_trend_indices = self._get_px(df_tickers, "TrendIndices")
        
        df_out = (pd.concat([
            df_barclays, df_bloomberg, df_bnp, df_bofa, df_citi, df_commod,
            df_db_trend, df_trend_indices]))
        
        if verbose: 
            print("Saving Trend Indices")
            
        df_out.to_parquet(path = out_path, engine = "pyarrow")

    
def main() -> None: 

    db_path       = r"A:\2026BlpAdHocData\Combined"
    trend_indices = TrendIndicesDataPrep(db_path)
    trend_indices.get_raw_trend_indices()
    
if __name__ == "__main__": main()