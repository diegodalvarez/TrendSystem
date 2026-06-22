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
        self.px_path   = os.path.join(self.data_path, "FuturesData")
        
        if os.path.exists(self.px_path) == False: os.makedirs(self.px_path)
        
    def _get_left_over_futures(
            self, 
            path: str,
            df  : pd.DataFrame) -> pd.DataFrame: 
        
        read_path    = os.path.join(path, "LeftOverFutures.xlsx")
        sheet_names  = pd.ExcelFile(path_or_buffer = read_path).sheet_names
        df_tmp_guide = (df.assign(
            tmp = lambda x: x.ticker.str.split("1").str[0].str.strip()).
            query("tmp == @sheet_names").
            sort_values("ticker").
            query("ticker != ['SM1 Comdty', 'QC1 Comdty']").
            rename(columns = {"tmp": "root_ticker"}))
    
        df_out = pd.DataFrame()
    
        for sheet_name in sheet_names: 
            
            df_tmp = (pd.read_excel(
                io = read_path, sheet_name = sheet_name)
                [["Date", "Last Px"]].
                rename(columns = {"Last Px": "px"}).
                assign(root_ticker = sheet_name).
                merge(
                    right = df_tmp_guide, 
                    how   = "inner", 
                    on    = ["root_ticker"]).
                assign(px = lambda x: x.px.astype(float)))
            
            df_out = pd.concat([df_out, df_tmp])

        return df_out
    
    def get_raw_fut_px(
            self, 
            fut_path     : str, 
            fx_path      : str, 
            leftover_path: str,
            verbose      : bool = True) -> None: 
        
        '''
        Gets the raw futures data from all the data sources, combines them,
        then adjusts each future by the currency rate to get them in USD
        '''
        
        if verbose: 
            print("Collecting Futures Data")
            
        out_path = os.path.join(self.px_path, "RawFuturesPX.parquet")
        
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
        
        keep_tickers = (df_ticker_guide.
                        ticker.
                        drop_duplicates().
                        sort_values().
                        to_list())
        
        df_extra = (self._get_left_over_futures(
            leftover_path, df_ticker_guide).
            rename(columns = {
                "Date": "date",
                "px"  : "PX_LAST"}).
            assign(date = lambda x: pd.to_datetime(x.date)).
            drop(columns = ["root_ticker", "currency"]))
        
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
            df_vix, df_tmp, df_other, df_extra]).
            dropna().
            query("ticker == @keep_tickers"))
        
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
        
    def prep_fut_px(self, verbose: bool = True) -> None: 
        
        '''
        Prepares futures data in USD terms as well small formatting adjustments
        '''
        
        out_path = os.path.join(self.px_path, "PrepFuturesPX.parquet")
        if os.path.exists(out_path) == True: 
            if verbose: 
                print("Already have data\n")
                
            return None
        
        if verbose: 
            print("Cleaning futures data")
        
        in_path  = os.path.join(self.px_path, "RawFuturesPX.parquet")
        end_date = (pd.read_parquet(
            path = in_path, engine = "pyarrow")
            [["ticker", "date"]].
            groupby(["ticker"]).
            agg("max").
            date.
            min())
        
        df_out = (pd.read_parquet(
            path = in_path, engine = "pyarrow").
            assign(ticker = lambda x: (
                x.ticker
                .str.lower()
                .str.replace(" 1", "1")
                .str.strip()
                .str.replace(" ", "_"))).
            query("date < @end_date").
            pivot(index = "date", columns = "ticker", values = "adj_val").
            reset_index().
            melt(id_vars = "date", var_name = "ticker", value_name = "px_val").
            dropna())
        
        if verbose: 
            print("Saving data\n")
        
        df_out.to_parquet(path = out_path, engine = "pyarrow")
        
def main() -> None: 
        
    fx_path    = r"C:\Users\Diego\Desktop\WeekyNotebooks\20260609LastBBG"
    fut_path   = r"C:\Users\Diego\Desktop\WeekyNotebooks\20260609LastBBG\Prices"
    extra_path = r"C:\Users\Diego\Desktop\WeekyNotebooks\20260618FuturesVolsTmp"
    
    fut_collect = FuturesDataCollect()
    fut_collect.get_raw_fut_px(fut_path, fx_path, extra_path)
    fut_collect.prep_fut_px()
    
if __name__ == "__main__": main()