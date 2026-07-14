# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 23:05:34 2026

@author: Diego
"""

import os
import numpy as np
import pandas as pd
import datetime as dt

class FuturesDataCollect:
    
    def __init__(self) -> None: 
        
        self.dsrc_path = os.getcwd()
        self.src_path  = os.path.abspath(os.path.join(self.dsrc_path, ".."))
        self.root_path = os.path.abspath(os.path.join(self.src_path, ".."))
        self.data_path = os.path.join(self.root_path, "data")
        self.px_path   = os.path.join(self.data_path, "FuturesData")
        
        if os.path.exists(self.px_path) == False: os.makedirs(self.px_path)
        
        self.eq_bad_tickers1 = [
            "HI1 Index", "IB1 Index", "PT1 Index", "QC1 Index", "SM1 Index"]
        
        self.eq1_max_px_dict = {
            "HI1 Index": 100_000,
            "IB1 Index": 30_000,
            "PT1 Index": 20_000,
            "QC1 Index": 100_000,
            "SM1 Index": 100_000}
        
        self.commod1_slice_px_dict = {
            "CN1 Comdty" : 2_000,
            "COR1 Comdty": 98,
            "JB1 Comdty" : 500,
            "JOA1 Comdty": 100,
            "KAA1 Comdty": 1_000,
            "KE1 Comdty" : 1_000,
            "XM1 Comdty" : 200,
            "YM1 Comdty" : 250,
            "ZB1 Comdty" : 140}
        
        self.commod2_slice_px_dict = {
            "XM1 Comdty": 80,
            "ZB1 Comdty": 83}
        
        self.start_date = dt.date(year = 2000, month = 1, day = 1)
        self.end_date   = dt.date(year = 2026, month = 6, day = 1)
        
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
    
    def _get_extra_equity1(
            self,
            path: str) -> pd.DataFrame: 
        
        files    = [file for file in os.listdir(path) if file[0:3] == "Fut"]
        df_lists = []
        
        for file in files: 
        
            tmp_path = os.path.join(path, file)
            df_raw   = (pd.read_excel(
                io         = tmp_path, 
                sheet_name = "PX_LAST").
                rename(columns = {"Date": "date"}).
                set_index("date"))
        
            df_raw.columns = [
                col.split("(")[0].replace("COMB", "").replace("  "," ").strip() 
                for col in df_raw.columns]
            
            df_tmp = (df_raw.reset_index().melt(
                id_vars = "date", var_name = "ticker", value_name = "PX_LAST").
                dropna())
        
            df_lists.append(df_tmp)
        
        df_out = pd.concat(df_lists)
        return df_out
    
    def _get_extra_fx_fut(self, fx_path: str) -> pd.DataFrame:    

        extra_fx_fut_path = os.path.join(fx_path, "FXFutures.xlsx")
        
        df_out = (pd.read_excel(
            io = extra_fx_fut_path, sheet_name = "PX_LAST").
            rename(columns = {"Date": "date"}).
            assign(date = lambda x: pd.to_datetime(x.date).dt.date).
            melt(id_vars = "date").
            assign(variable = lambda x: (
                x.variable
                .str.replace(" COMB", "")
                .str.split("(")
                .str[0]
                .str.strip())).
            dropna().
            rename(columns = {
                "variable": "ticker",
                "value"   : "PX_LAST"}))
        
        return df_out
    
    def get_raw_fut_px(
            self, 
            fut_path     : str, 
            leftover_path: str,
            eq_path      : str,
            fx1_path     : str,
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
                print("Already have futures data\n")
        
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
                        dropna().
                        sort_values().
                        to_list())
        
        df_fx_fut1 = self._get_extra_fx_fut(fx1_path)
            
        df_extra = (self._get_left_over_futures(
            leftover_path, df_ticker_guide).
            rename(columns = {
                "Date": "date",
                "px"  : "PX_LAST"}).
            assign(date = lambda x: pd.to_datetime(x.date)).
            drop(columns = ["root_ticker", "currency"]))
        
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
            rename(columns = {"security": "ticker"}).
            query("ticker != 'EO1 Index'"))
        
        df_extra_equity1 = self._get_extra_equity1(eq_path)
        
        df_raw_px = (pd.concat([
            df_vix, df_tmp, df_other, df_extra, df_extra_equity1,
            df_fx_fut1]).
            dropna().
            query("ticker == @keep_tickers").
            assign(
                date    = lambda x: pd.to_datetime(x.date).dt.date,
                PX_LAST = lambda x: x.PX_LAST.astype(float)))
        
        if verbose: 
            print("Saving Raw Data")
            
        df_raw_px.to_parquet(path = out_path, engine = "pyarrow")
        
    def _get_equity_spot_data(self) -> pd.DataFrame: 
        
        '''
        Gets equity spot data to clean equity futures data, uses Yahoo Finance
        for all of the data except Canadian equities which is pulled from
        BBG
        '''
        
        yf_path  = os.path.join(self.px_path, "YahooFinanceEquitySpot.parquet")
        bbg_path = os.path.join(self.px_path, "EquityIndexSpotPrices.xlsx")
        
        df_yf_spot = (pd.read_parquet(
            path = yf_path, engine = "pyarrow").
            reset_index().
            assign(date = lambda x: pd.to_datetime(x.Date, utc = True).dt.date)
            [["date", "ticker", "Close"]].
            rename(columns = {"Close": "spot_px"}))
        
        df_bbg_spot = pd.read_excel(io = bbg_path)
        
        df_bbg_spot_tmp = (df_bbg_spot.set_index(
            "Date")
            [["SPTSX60"]].
            apply(lambda x: np.where(x < 3_000, x, np.nan)).
            ffill().
            reset_index().
            rename(columns = {"Date": "date"}).
            melt(id_vars = "date", var_name = "ticker", value_name = "spot_px").
            assign(date = lambda x: pd.to_datetime(x.date).dt.date))
        
        df_spot = (pd.concat([
            df_yf_spot, df_bbg_spot_tmp]))
        
        return df_spot
        
    def _clean_equity_futures1(self, df: pd.DataFrame, df_ticker: pd.DataFrame) -> pd.DataFrame: 
        
        ticker_dict = (df_ticker.set_index(
            "Front").
            yahoo_underlying.
            to_dict())
        
        df_spot    = self._get_equity_spot_data()
        clean1_dfs = []
        
        df_good = (df.query(
            "ticker != @self.eq_bad_tickers1").
            assign(
                fut_px  = lambda x: x.PX_LAST,
                tmp_px  = lambda x: x.PX_LAST,
                repl_px = lambda x: x.PX_LAST,
                new_px  = lambda x: x.PX_LAST).
            drop(columns = ["PX_LAST"]))
        
        ending_dict = (df[
            ["ticker", "ending"]].
            drop_duplicates().
            set_index("ticker").
            ending.
            to_dict())
        
        for bad_ticker in self.eq_bad_tickers1: 
            
            df_fut_tmp = (df.query(
                "ticker == @bad_ticker")
                [["date", "ticker", "PX_LAST"]].
                rename(columns = {"PX_LAST": "fut_px"}))
        
            spot_ticker = ticker_dict[bad_ticker]
            df_spot_tmp = (df_spot.query(
                "ticker == @spot_ticker").
                drop(columns = ["ticker"]))
        
            max_px = self.eq1_max_px_dict[bad_ticker]
        
            df_clean1_tmp = (df_fut_tmp.merge(
                right = df_spot_tmp, how = "inner", on = ["date"]).
                set_index("date").
                sort_index().
                assign(
                    fut_rtn  = lambda x: x.fut_px.pct_change(),
                    spot_rtn = lambda x: x.spot_px.pct_change(),
                    diff_rtn = lambda x: (x.fut_rtn - x.spot_rtn) ** 2,
                    z_score  = lambda x: (x.diff_rtn - x.diff_rtn.mean()) / x.diff_rtn.std(),
                    z_px     = lambda x: np.where(np.abs(x.z_score) < 4, x.fut_px, np.nan),
                    tmp_px   = lambda x: np.where(x.z_px < max_px, x.z_px, np.nan),
                    repl_px  = lambda x: x.tmp_px.ewm(span = 5, adjust = False).mean(),
                    new_px   = lambda x: np.where(x.tmp_px != x.tmp_px, x.repl_px, x.tmp_px))
                [["fut_px", "tmp_px", "repl_px", "new_px"]].
                assign(ticker = bad_ticker))
        
            clean1_dfs.append(df_clean1_tmp)
            
        df_cleaned1 = (pd.concat(
            clean1_dfs).
            reset_index().
            assign(ending = lambda x: x.ticker.map(ending_dict)))
        
        df_out = pd.concat([df_good, df_cleaned1])
        
        return df_out
    
    def _clean_equity_futures2(self, df: pd.DataFrame) -> pd.DataFrame: 
        
        '''
        Just a small fix for IB Futures that need to be fixed
        '''
        
        df_ib = (df.query(
            "ticker == 'IB1 Index'").
            assign(
                year = lambda x: pd.to_datetime(x.date).dt.year,
                new_px = lambda x: np.where((x.year == 2000) & (x.new_px > 8_000), np.nan, x.new_px)))
        
        df_out = (pd.concat([
            df.query("ticker != 'IB1 Index'"),
            df_ib]).
            drop(columns = ["year", "fut_px", "tmp_px", "repl_px"]).
            rename(columns = {"new_px": "PX_LAST"}))
        
        return df_out
    
    def _clean_commod_futures1(self, df: pd.DataFrame) -> pd.DataFrame: 
        
        bad_commods = list(self.commod1_slice_px_dict.keys())
        
        df_good = (df.query(
            "ticker != @bad_commods").
            assign(tmp_px = lambda x: x.PX_LAST))
        
        df_add = pd.DataFrame()
        for bad_ticker in self.commod1_slice_px_dict.keys():
            
            slice_px = self.commod1_slice_px_dict[bad_ticker]
            
            df_tmp = (df.query(
                "ticker == @bad_ticker").
                assign(tmp_px = lambda x: np.where(x.PX_LAST < slice_px, x.PX_LAST, np.nan)).
                assign(tmp_px = lambda x: np.where(x.tmp_px != 0, x.tmp_px, np.nan)))
            
            df_add = pd.concat([df_tmp, df_add])
        
        df_out = pd.concat([df_add, df_good])
        return df_out
    
    def _clean_commod_futures2(self, df: pd.DataFrame) -> pd.DataFrame: 
        
        bad_tickers = list(self.commod2_slice_px_dict.keys())
        df_good     = df.query("ticker != @bad_tickers")
        
        for ticker in bad_tickers: 
            
            slice_px = self.commod2_slice_px_dict[ticker]
            df_add   = (df.query(
                "ticker == @ticker").
                assign(tmp_px = lambda x: np.where(x.tmp_px < slice_px, np.nan, x.tmp_px)))
            
            df_good = pd.concat([df_good, df_add])
        
        tmp_ticker = "KE1 Comdty"
        df_non_ke  = df_good.query("ticker != @tmp_ticker")
        
        df_ke = (df_good.query(
            "ticker == @tmp_ticker").
            assign(year = lambda x: pd.to_datetime(x.date).dt.year))
        
        df_ke_ex_2000 = df_ke.query("year != 2000")
        
        df_ke_2000 = (df_ke.query(
            "year == 2000").
            assign(tmp_px = lambda x: np.where(x.tmp_px <= 85, x.tmp_px, np.nan)))
        
        df_new_ke = (pd.concat([
            df_ke_ex_2000,
            df_ke_2000]).
            drop(columns = ["year"]))
        
        df_new_commod = pd.concat([df_new_ke, df_non_ke])
        return df_new_commod
                
    def clean_futures(self, verbose: bool = True) -> None:
        
        names   = ["HI", "IB", "PT", "QC", "SM"]
        tickers = ["{}1 Index".format(name) for name in names] 
        
        '''
        Cleans bad data within futures  data
        need to implement a commodities cleaner
        '''
        
        raw_path    = os.path.join(self.px_path, "RawFuturesPX.parquet")
        ticker_path = os.path.join(self.data_path, "TickerGuide.xlsx")
        out_path    = os.path.join(self.px_path, "CleanedRawFuturesPX.parquet")
        
        if verbose: 
            print("Getting cleaned data")
            
        if os.path.exists(out_path) == True:
            if verbose: 
                print("Already have data\n")
        
            return None
        
        if verbose: 
            print("Cleaning Futures Data")
        
        df_tmp = (pd.read_parquet(
            path = raw_path, engine = "pyarrow").
            assign(ending = lambda x: x.ticker.str.replace(" 1", "1").str.split(" ").str[-1]))
        
        df_eq_ticker = (pd.read_excel(
            io = ticker_path)
            [["Front", "yahoo_underlying"]].
            dropna().
            assign(yahoo_underlying = lambda x: x.yahoo_underlying.replace("TX60.TS", "SPTSX60")))
        
        df_eq        = df_tmp.query("ending == 'Index'")
        df_noneq     = df_tmp.query("ending != 'Index'")
        
        df_clean_eq1 = self._clean_equity_futures1(df_eq, df_eq_ticker)
        df_clean_eq2 = self._clean_equity_futures2(df_clean_eq1)
        
        df_clean1 = (pd.concat([
            df_clean_eq2, df_noneq]).
            query("@self.start_date <= date <= @self.end_date"))
        
        df_commod = df_clean1.query("ending == 'Comdty'")
        
        df_noncommod = (df_clean1.query(
            "ending != 'Comdty'").
            assign(tmp_px = lambda x: x.PX_LAST))
        
        df_clean_commod1 = self._clean_commod_futures1(df_commod)
        df_clean_commod2 = self._clean_commod_futures2(df_clean_commod1)
        
        df_out = (pd.concat([
            df_clean_commod2, df_noncommod]).
            rename(columns = {"tmp_px": "CLEAN_PX"}))
        
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
        
    def prep_fut_px(self, fx_path: str, fx_path1: str, verbose: bool = True) -> None: 
        
        if verbose:
            print("Getting FX Adjusted Data")
        
        out_path = os.path.join(self.px_path, "PrepFuturesPX.parquet")
        
        if os.path.exists(out_path) == True: 
            if verbose:
                print("Already have data\n")
                
            return None
        
        in_path  = os.path.join(self.px_path, "RawFuturesPX.parquet")
        
        end_date = (pd.read_parquet(
            path = in_path, engine = "pyarrow")
            [["ticker", "date"]].
            groupby(["ticker"]).
            agg("max").
            date.
            min())
        
        px_path = os.path.join(self.px_path, "CleanedRawFuturesPX.parquet")
        df_px   = pd.read_parquet(path = px_path, engine = "pyarrow")
        
        ticker_path     = os.path.join(self.data_path, "TickerGuide.xlsx")
        df_ticker_guide = (pd.read_excel(
            io = ticker_path, sheet_name = "fut_guide")
            [["Front", "currency"]].
            rename(columns = {"Front": "ticker"}))
        
        fx_names = ([
            fx + "USD Curncy" for fx in 
            df_ticker_guide.currency.drop_duplicates().sort_values().to_list()
            if fx != "USD"])
        
        fx_read_path = os.path.join(fx_path, "FX.parquet")
        
        df_fx_orig = pd.read_parquet(path = fx_read_path, engine = "pyarrow")
        df_fx1     = self._get_fx1(fx_path1)
        df_fx_raw  = pd.concat([df_fx_orig, df_fx1])
    
        df_fx = (df_fx_raw.query(
            "ticker == @fx_names").
            assign(fx_ticker = lambda x: x.ticker.str[0:3])
            [["date", "fx_ticker", "value"]].
            rename(columns = {"value": "fx_val"}).
            assign(fx_val = lambda x: x.fx_val.astype(float)).
            drop_duplicates())

        fx_map = (df_ticker_guide[
            ["ticker", "currency"]].
            drop_duplicates().
            set_index("ticker").
            currency.
            to_dict())
        
        df_out = (df_px.assign(
            fx_ticker = lambda x: x.ticker.map(fx_map)).
            merge(right = df_fx, how = "left", on = ["date", "fx_ticker"]).
            assign(
                fx_val  = lambda x: np.where(x.fx_ticker == "USD", 1, x.fx_val).astype(float),
                adj_val = lambda x: x.fx_val * x.PX_LAST,
                ticker  = lambda x: x.ticker.str.lower().str.replace(" 1", "1").str.strip().str.replace(" ", "_")).
            query("date <= @end_date").
            assign(ticker = lambda x: x.ticker.str.strip()))
        
        if verbose: 
            print("Saving data\n")
            
        df_out.to_parquet(path = out_path, engine = "pyarrow")
        
def main() -> None: 
        
    fx_path    = r"C:\Users\Diego\Desktop\WeekyNotebooks\20260609LastBBG"
    fut_path   = r"C:\Users\Diego\Desktop\WeekyNotebooks\20260609LastBBG\Prices"
    extra_path = r"C:\Users\Diego\Desktop\WeekyNotebooks\20260618FuturesVolsTmp\tmp"
    eq_path    = r"C:\Users\Diego\Desktop\WeekyNotebooks\20260705TmpFutures\EquityFutures"
    fx1_path   = r"C:\Users\Diego\Desktop\WeekyNotebooks\20260705TmpFutures\FX"
    
    fut_collect = FuturesDataCollect()
    fut_collect.get_raw_fut_px(fut_path, extra_path, eq_path, fx1_path)
    fut_collect.clean_futures()
    fut_collect.prep_fut_px(fx_path, fx1_path)
    
if __name__ == "__main__": main()