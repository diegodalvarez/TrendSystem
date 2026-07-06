# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 20:28:13 2026

@author: Diego
"""

import os
import pandas as pd

class TrendIndicesDataPrep:
    
    def __init__(self) -> None: 
        
        self.dsrc_path = os.getcwd()
        self.src_path  = os.path.abspath(os.path.join(self.dsrc_path, ".."))
        self.repo_path = os.path.abspath(os.path.join(self.src_path, ".."))
        self.data_path = os.path.join(self.repo_path, "data")
        self.idx_path  = os.path.join(self.data_path, "TrendIndices")
        
    def get_trend_indices(self, tpath: str, cta_path: str, verbose: bool = True) -> None: 
        
        out_path   = os.path.join(self.idx_path, "TrendIndices.parquet")
        trend_path = os.path.join(tpath, "TrendIndices.parquet")
        cta_path   = os.path.join(cta_path, "NEIXCTA.xlsx")
        
        if os.path.exists(out_path) == True: 
            if verbose: 
                print("Already have Trend Indices data\n")
                
            return None
        
        if verbose: 
            print("Getting Trend Indices")
        
        df_trend = (pd.read_parquet(
            path = trend_path, engine = "pyarrow"))
        
        df_cta = (pd.read_excel(
            io = cta_path).
            rename(columns = {
                "Date"      : "date",
                "Last Price": "PX_LAST"}).
            assign(security = "NEIXCTA Index"))

        df_out = (pd.concat([
            df_trend, df_cta]))
        
        if verbose: 
            print("Saving Trend Indices")
            
        df_out.to_parquet(path = out_path, engine = "pyarrow")


trend_indices = TrendIndicesDataPrep()

tpath    = r"C:\Users\Diego\Desktop\WeekyNotebooks\20260422AprilDataCollect (passed)"
cta_path = r"C:\Users\Diego\Desktop\WeekyNotebooks\20260705TmpFutures\SGCTA"
trend_indices.get_trend_indices(tpath, cta_path)