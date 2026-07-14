# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 16:52:17 2026

@author: Diego
"""

import os
import numpy as np
import pandas as pd

class PrepareImpliedCorrelationData:
    
    def __init__(self, db_path: str) -> None: 
        
        self.dataprep_path = os.getcwd()
        self.src_path      = os.path.abspath(os.path.join(self.dataprep_path, ".."))
        self.repo_path     = os.path.abspath(os.path.join(self.src_path, ".."))
        self.data_path     = os.path.join(self.repo_path, "data")
        self.fut_path      = os.path.join(self.data_path, "FuturesData")
        self.misc_path     = os.path.join(self.data_path, "MiscData")
        self.db_path       = db_path
        
    def get_raw_cboe_corr(self, verbose: bool = True) -> None: 
        
        out_path    = os.path.join(self.misc_path, "RawCBOEImpliedCorrelation.parquet")
        ticker_path = os.path.join(self.misc_path, "ImpliedCorrelationTickerGuide.xlsx")
        
        if verbose: 
            print("Getting CBOE Implied Correlation Data")
            
        if os.path.exists(out_path) == True: 
            if verbose: 
                print("Already have data\n")
                
            return None
        
        if verbose: 
            print("Preparing data")
    
        file = (pd.read_excel(
            io = ticker_path, sheet_name = "CBOEGuide").
            File.
            drop_duplicates().
            item())
        
        in_path = os.path.join(self.db_path, "PX", file + ".parquet")
        df_out  = (pd.read_parquet(
            path = in_path, engine = "pyarrow").
            assign(value = lambda x: x.value / 100))
        
        if verbose: 
            print("Saving data\n")
            
        df_out.to_parquet(path = out_path, engine = "pyarrow")
        
    def _prep_bloomberg(self, path: str) -> pd.DataFrame:
        
        df_raw     = pd.read_excel(path, sheet_name = None)
        raw_keys   = df_raw.keys()
        sheet_list = []
        
        for raw_key in raw_keys: 
            
            df_add = (df_raw[raw_key].melt(
                id_vars = "Date").
                assign(window = raw_key))
            
            sheet_list.append(df_add)
            
        df_out = pd.concat(sheet_list)
        return df_out
        
    def get_raw_bloomberg_corr(self, verbose: bool = True) -> None:
        
        out_path = os.path.join(self.misc_path, "RawImpliedRealCorrelation.parquet")
        
        if verbose: 
            print("Gettting Bloomberg Implied Correlations")
            
        if os.path.exists(out_path) == True:
            if verbose: 
                print("File Already exists\n")
                
            return None
        
        if verbose: 
            print("Getting Data")
        
        
        file_dict = {
            "EquityImpliedCorrelation" : "ImpliedCorr", 
            "EquityRealizedCorrelation": "RealizedCorr"}
        
        paths = [
            os.path.join(self.db_path, "PX", file + ".parquet")
            for file in file_dict.keys()]
        
        df_out = (pd.concat([
            pd.read_parquet(path = file, engine = "pyarrow").assign(file = file)
            for file in paths]).
            assign(
                file = lambda x: (
                    x
                    .file.str.split("\\")
                    .str[-1]
                    .str.split(".")
                    .str[0]
                    .map(file_dict))))
        
        if verbose: 
            print("Saving data\n")
            
        df_out.to_parquet(path = out_path, engine = "pyarrow")
        
    def _clean_val(self, df: pd.DataFrame) -> pd.DataFrame: 
        
        df_out = (df.assign(
            replace_val = lambda x: x.clean_val.ewm(span = 2, adjust = False).mean(),
            clean_val   = lambda x: np.where(x.clean_val != x.clean_val, x.replace_val, x.clean_val)).
            drop(columns = ["replace_val"]))
        
        return df_out
        
    def clean_bbg_corr(self, verbose: bool = True) -> None: 
        
        in_path  = os.path.join(self.misc_path, "RawImpliedRealCorrelation.parquet")
        out_path = os.path.join(self.misc_path, "CleanedImpliedRealCorrelation.parquet")
        
        if verbose: 
            print("Getting Cleaned Correlation Data")
        
        if os.path.exists(out_path) == True:
            if verbose: 
                print("Already have data\n")
                
            return None
        
        if verbose: 
            print("Cleaning Bloomberg Data")
            
        df_out = (pd.read_parquet(
            path = in_path, engine = "pyarrow").
            assign(
                group_var = lambda x: x.variable + " " + x.window + " " + x.file,
                clean_val = lambda x: np.where(x.value >= 1, np.nan, x.value)).
            set_index("Date").
            groupby("group_var").
            apply(self._clean_val).
            reset_index().
            drop(columns = ["group_var"]))
        
        if verbose: 
            print("Saving data\n")
            
        df_out.to_parquet(path = out_path, engine = "pyarrow")
        
    def _clean_cboe_corr(self, df: pd.DataFrame) -> pd.DataFrame: 
        
        df_out = (df.assign(
            tmp1      = lambda x: np.where(x.value > 1, np.nan, x.value),
            tmp2      = lambda x: x.tmp1.ewm(span = 2, adjust = False).mean(),
            clean_val = lambda x: np.where(x.tmp1 != x.tmp1, x.tmp2, x.value)).
            drop(columns = ["tmp1", "tmp2"]))
        
        return df_out
        
    def clean_cboe_corr(self, verbose: bool = True) -> None: 
        
        in_path  = os.path.join(self.misc_path, "RawCBOEImpliedCorrelation.parquet")
        out_path = os.path.join(self.misc_path, "CleanedCBOEImpliedCorrelation.parquet")
        
        if verbose: 
            print("Getting Cleaning CBOE Data")
            
        if os.path.exists(out_path) == True: 
            if verbose: 
                print("Already have data\n")
                
            return None
        
        if verbose:
            print("Getting Cleaned CBOE data")
        
        df_out = (pd.read_parquet(
            path = in_path, engine = "pyarrow").
            set_index("Date").
            groupby("variable").
            apply(self._clean_cboe_corr).
            reset_index())
        
        if verbose: 
            print("Saving data\n")
            
        df_out.to_parquet(path = out_path, engine = "pyarrow")

def main() -> None: 
    
    #path      = r"C:\Users\Diego\Desktop\WeekyNotebooks\20260629TemporaryImpliedCorrelation"
    db_path = r"A:\2026BlpAdHocData\Combined"
    corr_prep = PrepareImpliedCorrelationData(db_path)
    corr_prep.get_raw_cboe_corr()
    corr_prep.clean_cboe_corr()
    corr_prep.get_raw_bloomberg_corr()
    corr_prep.clean_bbg_corr()
    
if __name__ == "__main__": main()