# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 21:52:03 2026

@author: Diego
"""

import os
import numpy as np
import pandas as pd

class EquityIndexPrep:
    
    def __init__(self) -> None: 
        
        self.data_prep_path = os.getcwd()
        self.src_path       = os.path.abspath(os.path.join(self.data_prep_path, ".."))
        self.repo_path      = os.path.abspath(os.path.join(self.src_path, ".."))
        self.data_path      = os.path.join(self.repo_path, "data")
        self.eq_data_path   = os.path.join(self.data_path, "EquityBenchmarks")
        
        self.bad_tickers = [
            "AS51", "CAC", "DAX", "HSI", "INDU", "NDX", "NKY", "OMX", 
            "SPTSX60", "SPX", "SX5E", "UKX"]
        
    def _get_eq1(self, path: str) -> pd.DataFrame: 
        
        files = [
            os.path.join(path, file) 
            for file in os.listdir(path)
            if file[0:3] == "Spo"]
        
        df_lists = []
        
        for file in files: 
            
            df_px = (pd.read_excel(
                io = file, sheet_name = "PX_LAST").
                rename(columns = {"Date": "date"}).
                assign(date = lambda x: pd.to_datetime(x.date).dt.date).
                melt(id_vars = "date", value_name = "px").
                dropna().
                assign(variable = lambda x: x.variable.str.split("(").str[0].str.strip().str.lstrip()))
            
            df_rtn = (pd.read_excel(
                io = file, sheet_name = "RT112").
                rename(columns = {"Date": "date"}).
                assign(date = lambda x: pd.to_datetime(x.date).dt.date).
                melt(id_vars = "date", value_name = "rtn").
                dropna().
                assign(
                    variable = lambda x: x.variable.str.split("(").str[0].str.strip().str.lstrip(),
                    rtn      = lambda x: x.rtn / 100))
            
            df_add = (df_px.merge(
                right = df_rtn, how = "inner", on = ["date", "variable"]))
            
            df_lists.append(df_add)
            
        df_out = (pd.concat(
            df_lists).
            rename(columns = {"variable": "ticker"}))
        
        return df_out
    
    def _get_eq2(self, path: str) -> pd.DataFrame: 
        
        files    = os.listdir(path)
        df_lists = []
        
        for file in files: 
            
            tmp_path = os.path.join(path, file)
            
            df_px = (pd.read_excel(
                io = tmp_path, sheet_name = "PX_LAST").
                rename(columns = {"Date": "date"}).
                assign(date = lambda x: pd.to_datetime(x.date).dt.date).
                melt(id_vars = "date", value_name = "px").
                dropna())
            
            df_rtn = (pd.read_excel(
                io = tmp_path, sheet_name = "RT112").
                rename(columns = {"Date": "date"}).
                assign(date = lambda x: pd.to_datetime(x.date).dt.date).
                melt(id_vars = "date", value_name = "rtn").
                assign(rtn = lambda x: x.rtn / 100))
            
            df_add = (df_px.merge(
                right = df_rtn, how = "inner", on = ["date", "variable"]).
                assign(ticker = lambda x: x.variable + " Index").
                drop(columns = ["variable"]))
            
            df_lists.append(df_add)
            
        df_out = pd.concat(df_lists)
        return df_out
        
    def get_raw_eq(self, path1: str, path2: str, verbose: bool = True) -> None:
        
        out_path = os.path.join(self.eq_data_path, "RawEquityBenchmarkPXRtn.parquet")
        
        if verbose: 
            print("Getting Equity Indices")
            
        if os.path.exists(out_path) == True: 
            if verbose: 
                print("Saving data\n")
                
            return None
        
        if verbose: 
            print("Preparing data")
        
        df_eq1 = self._get_eq1(path1)
        df_eq2 = self._get_eq2(path2)
        
        df_out = (pd.concat([
            df_eq1, df_eq2]).
            dropna())
        
        if verbose: 
            print("Saving data\n")
            
        df_out.to_parquet(path = out_path, engine = "pyarrow")
        
    def _clean_eq(self, df: pd.DataFrame) -> pd.DataFrame:
        
        tickers  = df.ticker.drop_duplicates().sort_values().to_list()
        df_lists = [] 
        
        for ticker in tickers: 
            
            df_tmp   = df.query("ticker == @ticker")
            start_px = df_tmp.query("date == date.min()").drop_duplicates().px.item()
            
            df_add = (df_tmp.set_index(
                "date").
                sort_index().
                assign(
                    px_rtn   = lambda x: x.px.pct_change(),
                    repl_rtn = lambda x: np.where(np.abs(x.px_rtn) > np.abs(x.rtn), x.rtn, x.px_rtn),
                    px_repl  = lambda x: np.cumprod(1 + x.repl_rtn) * start_px))
            
            df_lists.append(df_add)
            
        df_out = (pd.concat(
            df_lists).
            drop(columns = ["repl_rtn", "px_rtn"]).
            reset_index())
        
        return df_out
        
    def get_clean_eq(self, verbose: bool = True) -> pd.DataFrame: 
        
        in_path     = os.path.join(self.eq_data_path, "RawEquityBenchmarkPXRtn.parquet")
        out_path    = os.path.join(self.eq_data_path, "CleanEquityBenchmarkPXRtn.parquet")
        bad_tickers = ["{} Index".format(ticker) for ticker in self.bad_tickers]
        
        if verbose: 
            print("Getting Clean Equity Data")
            
        if os.path.exists(out_path) == True: 
            if verbose: 
                print("Already have data\n")
                
            return None
        
        if verbose: 
            print("Cleaning data")
        
        df_input = pd.read_parquet(path = in_path, engine = "pyarrow")
        df_bad   = df_input.query("ticker == @bad_tickers")
        df_clean = self._clean_eq(df_bad)
        
        df_good  = (df_input.query(
            "ticker != @bad_tickers").
            assign(px_repl = lambda x: x.px))
        
        df_out = (pd.concat([
            df_good, df_clean]).
            rename(columns = {
                "px"     : "PX_LAST",
                "rtn"    : "TotRtn",
                "px_repl": "CLEAN_PX"}))
        
        if verbose: 
            print("Saving data\n")
            
        df_out.to_parquet(path = out_path, engine = "pyarrow")
        
    def _get_fx1(self, path: str) -> pd.DataFrame: 
        
        in_path = os.path.join(path, "FX.xlsx")
        df_raw  = (pd.read_excel(
            io = in_path).
            rename(columns = {"Date": "date"}).
            set_index("date"))
        
        df_raw.columns = [
            col.replace("BGN", "").split("(")[0].strip().replace("  "," ") 
            for col in df_raw.columns]
        
        df_out = (df_raw.reset_index().melt(
            id_vars = "date", var_name = "ticker").
            dropna().
            assign(
                field = "PX_LAST",
                date  = lambda x: pd.to_datetime(x.date).dt.date))
        
        return df_out
        
    def get_prep_eq(self, fx_path: str, fx_path1: str, verbose: bool = True) -> None: 
        
        in_path     = os.path.join(self.eq_data_path, "CleanEquityBenchmarkPXRtn.parquet")
        ticker_path = os.path.join(self.data_path, "TickerGuide.xlsx")
        out_path    = os.path.join(self.eq_data_path, "PrepEquityBenchmarkPXRtn.parquet")
        
        if verbose: 
            print("Getting Prepped Prices")
            
        if os.path.exists(out_path) == True: 
            if verbose: 
                print("Already have data\n")
                
            return None
        
        if verbose: 
            print("Adding FX to data to equity indices")
        
        df_ticker_guide = (pd.read_excel(
            io = ticker_path, sheet_name = "fut_guide")
            [["bloomberg_underlying", "currency"]].
            dropna().
            rename(columns = {
                "bloomberg_underlying": "ticker",
                "currency"            : "fx_ticker"}))
        
        df_tmp = (pd.read_parquet(
            path = in_path, engine = "pyarrow").
            merge(right = df_ticker_guide, how = "left", on = ["ticker"]))
        
        fx_names = ([
            fx + "USD Curncy" for fx in 
            df_ticker_guide.fx_ticker.drop_duplicates().sort_values().to_list()
            if fx != "USD"])
        
        fx_read_path = os.path.join(fx_path, "FX.parquet")
        
        df_fx_orig = pd.read_parquet(path = fx_read_path, engine = "pyarrow")
        df_fx1     = self._get_fx1(fx_path1)
        df_fx_raw  = pd.concat([df_fx_orig, df_fx1])
    
        df_fx = (df_fx_raw.query(
            "ticker == @fx_names").
            assign(fx_ticker = lambda x: x.ticker.str[0:3])
            [["date", "fx_ticker", "value"]].
            rename(columns = {"value": "fx_val"}))
        
        df_out = (df_tmp.merge(
            right = df_fx, how = "left", on = ["date", "fx_ticker"]).
            assign(
                ticker = lambda x: x.ticker.str.lower().str.replace(" ", "_"),
                fx_val = lambda x: np.where(x.fx_ticker == "USD", 1, x.fx_val)).
            assign(fx_val = lambda x: x.fx_val.astype(float)))
        
        if verbose: 
            print("Saving data\n")
            
        df_out.to_parquet(path = out_path, engine = "pyarrow")
        
def main() -> None: 
        
    path1    = r"C:\Users\Diego\Desktop\WeekyNotebooks\20260705TmpFutures\EquityFutures"
    path2    = r"C:\Users\Diego\Desktop\WeekyNotebooks\20260607EquityIndicesTmp"     
    fx_path  = r"C:\Users\Diego\Desktop\WeekyNotebooks\20260609LastBBG"
    fx1_path = r"C:\Users\Diego\Desktop\WeekyNotebooks\20260705TmpFutures\FX"
    
    eq_index = EquityIndexPrep()
    eq_index.get_raw_eq(path1, path2)
    eq_index.get_clean_eq()
    eq_index.get_prep_eq(fx_path, fx1_path)
    
if __name__ == "__main__": main()

