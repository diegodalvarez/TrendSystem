# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 08:46:01 2026

@author: Diego
"""

import os
import pandas as pd
import yfinance as yf
import datetime as dt

from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler



class CreditData:
    
    def __init__(self) -> None: 
        
        self.source1 = r"A:\BBGData\data"
        self.source2 = r"A:\2026BlpAdHocData\Combined\PX"
        self.source3 = r"A:\BBGData\ETFIndices\BondPricing"
        self.source4 = r"C:\Users\Diego\Desktop\WeekyNotebooks\20260422AprilDataCollect (passed)"
        
        self.dprep_path = os.getcwd()
        self.src_path   = os.path.abspath(os.path.join(self.dprep_path, ".."))
        self.repo_path  = os.path.abspath(os.path.join(self.src_path, ".."))
        self.data_path  = os.path.join(self.repo_path, "data")
        
        self.credit_path = os.path.join(self.data_path, "CreditData")
        if not os.path.exists(self.credit_path):
            os.makedirs(self.credit_path)
            
        self.n_clusters   = 5
        self.random_state = 42
        
    def collect_raw_cds_data(self, verbose: bool = True) -> None: 
        
        if verbose:
            print("Getting Raw CDS Data")
        
        out_path = os.path.join(self.credit_path, "RawCDS.parquet")
        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None
        
        ticker_path = os.path.join(self.data_path, "TickerGuide.xlsx")
        df_ticker   = (pd
                .read_excel(io = ticker_path, sheet_name = "credit")
                .loc[lambda x: x.AssetClass == "CreditDefaultSwap"])
        
        files1 = (df_ticker
                .loc[lambda x: x.Source == "source1"]
                .File
                .to_list())
        
        paths1    = [os.path.join(self.source1, file + ".parquet") for file in files1]
        df_group1 = pd.read_parquet(path = paths1, engine = "pyarrow")
        
        files2 = (df_ticker
                .loc[lambda x: x.Source == "source2"]
                .File
                .drop_duplicates()
                .to_list())
        
        paths2    = [os.path.join(self.source2, file + ".parquet") for file in files2]
        df_group2 = (pd
                .read_parquet(path = paths2, engine = "pyarrow")
                .melt(id_vars = ["date", "security"])
                .dropna())
        
        df_combined = pd.concat([df_group1, df_group2])
        if verbose: print("Saving data\n")
        df_combined.to_parquet(path = out_path, engine = "pyarrow")
        
    def collect_raw_credit_etf(self, verbose: bool = True) -> None:
        
        if verbose: 
            print("Collecting ETF Data")
            
        out_path = os.path.join(self.credit_path, "RawCreditETF.parquet")
        
        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None
        
        ticker_path = os.path.join(self.data_path, "TickerGuide.xlsx")
        tickers     = (pd
                .read_excel(io = ticker_path, sheet_name = "credit")
                .loc[lambda x: x.AssetClass == "CreditETF"]
                .File
                .to_list())
        
        paths          = [os.path.join(self.source3, file + ".parquet") for file in tickers]
        df_fundamental = (pd
                          .read_parquet(path = paths, engine = "pyarrow")
                          .assign(security = lambda x: x.security.str.split(" ").str[0]))
    
        start_date = dt.date(year = 2000, month = 1, day = 1)
        end_date   = dt.date(year = 2026, month = 8, day = 1)
        
        df_px = (yf
                .download(
                    tickers     = tickers,
                    start       = start_date,
                    end         = end_date,
                    auto_adjust = False)
                .reset_index()
                .melt(id_vars = [("Date", "")])
                .rename(columns = {
                    ("Date", ""): "date",
                    "Price"     : "variable",
                    "Ticker"    : "security"})
                .dropna())
        
        df_out = pd.concat([df_fundamental, df_px])
        if verbose: print("Saving data\n")
        df_out.to_parquet(path = out_path, engine = "pyarrow")

    def collect_raw_credit_indices(self, verbose: bool = True) -> None:
        
        if verbose:
            print("Getting Credit Index Data")
        
        out_path = os.path.join(self.credit_path, "RawCreditIndex.parquet")
        
        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None
        
        files = [
            "DAY_TO_DAY_TOT_RETURN_GROSS_DVDS", "INDEX_OAC_TSY", "INDEX_OAS_SWAP_BP", 
            "INDEX_OAS_TSY_BP", "INDEX_OAC_TSY", "INDEX_OAS_SWAP_BP", "INDEX_OAS_TSY_BP",
            "INDEX_SPREAD_BENCHMARK", "INDEX_YIELD_TO_MATURITY", "INDEX_YIELD_TO_WORST",
            "INDEX_Z_SPREAD_BP"]
        
        paths = ([
            os.path.join(self.source4, file + ".parquet")
            for file in files])
        
        df_credit_index = (pd
                           .concat([
                               pd.read_parquet(path = path, engine = "pyarrow").melt(id_vars = ["date", "security"])
                               for path in paths]))
        
        if verbose: print("Saving data\n")
        df_credit_index.to_parquet(path = out_path, engine = "pyarrow")
        
    def _slice_ends(self, df: pd.DataFrame) -> pd.DataFrame: 
        
        date_type = df.name[-1]
        if date_type == "max":
            
            df_out = (df
                    .loc[lambda x: x.value == x.value.min()]
                    .head(1))
            
        elif date_type == "min":
            
            df_out = (df
                      .loc[lambda x: x.value == x.value.max()]
                      .head(1))

        return df_out
        
    def prep_cds_data(self, slice_year: int = 2012, verbose: bool = True) -> None: 
        
        if verbose: 
            print("Prepping CDS Data")
            
        out_path = os.path.join(self.credit_path, "PrepCDS.parquet")
        
        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None
        
        ticker_path = os.path.join(self.data_path, "TickerGuide.xlsx")
        df_ticker   = (pd
                .read_excel(io = ticker_path, sheet_name = "credit")
                .loc[lambda x: x.Active == True]
                .loc[lambda x: x.AssetClass == "CreditDefaultSwap"]
                [["Ticker", "SeriesType", "SecurityGroup"]]
                .rename(columns = {"Ticker": "security"}))
        
        cred_path = os.path.join(self.credit_path, "RawCDS.parquet")
    
        df_raw = (pd
                .read_parquet(path = cred_path, engine = "pyarrow")
                .merge(right = df_ticker, how = "inner", on = ["security"])
                .loc[lambda x: x.variable == "PX_LAST"]
                [["date", "SeriesType", "SecurityGroup","value"]])
        
        df_date_slicer = (df_raw
                .drop(columns = ["value"])
                .groupby(["SecurityGroup", "SeriesType"])
                ["date"]
                .agg(["min", "max"])
                .reset_index()
                .melt(id_vars = ["SecurityGroup", "SeriesType"])
                .groupby(["SecurityGroup", "variable"])
                .apply(self._slice_ends)
                .reset_index()
                [["SecurityGroup", "variable", "value"]]
                .assign(variable = lambda x: x.variable + "_date")
                .pivot(index = "SecurityGroup", columns = "variable", values = "value"))

        df_sliced_wider = (df_raw
                .merge(right = df_date_slicer, how = "inner", on = ["SecurityGroup"])
                .loc[lambda x: (x.min_date <= x.date) & (x.date <= x.max_date)]
                .assign(year = lambda x: x.date.dt.year)
                .loc[lambda x: x.year >= slice_year]
                .drop(columns = ["year"])                
                .drop(columns = ["min_date", "max_date"])
                .pivot(index = ["date", "SecurityGroup"], columns = "SeriesType", values = "value"))
        
        df_spread_wider = (df_sliced_wider
                .reset_index()
                .pivot(index = "date", columns = "SecurityGroup", values = "spread")
                .interpolate(method = "time"))
        
        df_spread = (df_spread_wider
                .reset_index()
                .melt(id_vars = "date", value_name = "spread")
                .dropna())
        
        df_out = (df_sliced_wider
                .reset_index()
                .drop(columns = ["spread"])
                .merge(right = df_spread, how = "outer", on = ["date", "SecurityGroup"]))
        
        if verbose: print("Saving data\n")
        df_out.to_parquet(path = out_path, engine = "pyarrow")
        
def main() -> None: 
        
    credit_data = CreditData()
 
    # Raw
    '''
    credit_data.collect_raw_cds_data()
    credit_data.collect_raw_credit_etf()
    credit_data.collect_raw_credit_indices()
    '''
    
    # Prep
    credit_data.prep_cds_data()
    
if __name__ == "__main__": main()