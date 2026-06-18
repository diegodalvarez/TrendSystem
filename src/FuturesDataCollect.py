# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 23:05:34 2026

@author: Diego
"""

import os
import numpy as np
import pandas as pd

class FuturesDataCollect:
    
    def __init__(self) -> None: 
        
        self.src_path  = os.getcwd()
        self.root_path = os.path.abspath(os.path.join(self.src_path, ".."))
        self.data_path = os.path.join(self.root_path, "data")
    
    def get_fut_px(
            self, 
            fut_path: str, 
            fx_path : str, 
            verbose : bool = True) -> None: 
        
        if verbose: 
            print("Collecting Futures Data")
            
        out_path = os.path.join(self.data_path, "FuturesPX.parquet")
        
        if os.path.exists(out_path) == True: 
            if verbose: 
                print("Already have futures data")
        
            return None
        
        if verbose: 
            print("Getting Futures Data")
        
        ticker_path     = os.path.join(self.data_path, "TickerGuide.xlsx")
        df_ticker_guide = (pd.read_excel(
            io = ticker_path, sheet_name = "fut_guide")
            [["Front", "currency"]].
            rename(columns = {"Front": "ticker"}))
        
        fx_names = ([
            fx + "USD Curncy" for fx in 
            df_ticker_guide.currency.drop_duplicates().sort_values().to_list()
            if fx != "USD"])
        
        fx_path = os.path.join(fx_path, "FX.parquet")
        df_fx   = (pd.read_parquet(
            path = fx_path, engine = "pyarrow").
            query("ticker == @fx_names").
            assign(fx_ticker = lambda x: x.ticker.str[0:3])
            [["date", "fx_ticker", "value"]].
            rename(columns = {"value": "fx_val"}))
        
        vix_files = ["1CboeFuturesExchangePx.parquet"]
        tmp_files = [
            "1NewYorkMercantileExchangePx.parquet", "1OsakaExchangePx.parquet"]
        
        all_paths = [
            os.path.join(fut_path, file) for file in os.listdir(fut_path)]
        
        vix_paths   = [os.path.join(fut_path, file) for file in vix_files]
        tmp_paths   = [os.path.join(fut_path, file) for file in tmp_files]
        other_paths = list(set(all_paths) - set(vix_paths) - set(tmp_paths))
        
        df_vix = (pd.read_parquet(
            path = vix_paths, engine = "pyarrow")
            [["date", "security", "PX_LAST"]].
            rename(columns = {"security": "ticker"}))
        
        df_tmp = (pd.read_parquet(
            path = tmp_paths, engine = "pyarrow").
            pivot(
                index   = ["date", "ticker"], 
                columns = "field", 
                values  = "value").
            reset_index())
        
        df_other = (pd.read_parquet(
            path = other_paths, engine = "pyarrow").
            rename(columns = {"security": "ticker"}))
        
        df_raw_px = (pd.concat([
            df_vix, df_tmp, df_other]).
            dropna())
        
        fx_map       = df_ticker_guide.set_index("ticker").currency.to_dict()
        good_tickers = list(fx_map.keys())
        
        df_out = (df_raw_px.query(
            "ticker == @good_tickers").
            assign(
                PX_LAST   = lambda x: x.PX_LAST.astype(float),
                date      = lambda x: pd.to_datetime(x.date).dt.date,
                fx_ticker = lambda x: x.ticker.map(fx_map)).
            merge(right = df_fx, how = "left", on = ["date", "fx_ticker"]).
            assign(fx_val = lambda x: np.where(
                x.fx_ticker == "USD", 1, x.fx_val).astype(float)).
            assign(adj_val = lambda x: x.fx_val * x.PX_LAST))
        
        if verbose: 
            print("Saving data\n")
            
        df_out.to_parquet(path = out_path, engine = "pyarrow")
        
fx_path     = r"C:\Users\Diego\Desktop\WeekyNotebooks\20260609LastBBG"
fut_path    = r"C:\Users\Diego\Desktop\WeekyNotebooks\20260609LastBBG\Prices"

fut_collect = FuturesDataCollect()
fut_collect.get_fut_px(fut_path, fx_path)