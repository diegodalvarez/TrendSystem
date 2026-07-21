# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 16:34:23 2026

@author: Diego
"""

import numpy as np
import pandas as pd

class AQTools: 
    
    def __init__(self) -> None:
        pass
    
    def drawdown(self, prices_raw: pd.Series) -> dict:
            
        """
        Identifies the maximum drawdown of a price series by tracking the
        running peak and recording the largest subsequent decline.
    
        Parameters
        ----------
        prices_raw : pd.Series
            Time series of asset or portfolio prices indexed by date.
    
        Returns
        -------
        dict
            Dictionary containing:
    
            - max_val : Peak price preceding the maximum drawdown.
            - min_val : Trough price of the maximum drawdown.
            - start_date : Date corresponding to the peak.
            - end_date : Date corresponding to the trough.
        """
        
        prevmaxi    = 0
        prevmini    = 0
        maxi        = 0
        prices      = prices_raw.squeeze().to_list()
        dates       = prices_raw.index.to_list()
    
        for i in range(1, len(prices)):
    
            if prices[i] >= prices[maxi]:
                maxi = i
            else:
                if (prices[maxi] - prices[i]) > (prices[prevmaxi] - prices[prevmini]):
                    prevmaxi = maxi
                    prevmini = i
    
        out_dict = {
            "max_val"   : prices[prevmaxi], 
            "min_val"   : prices[prevmini],
            "start_date": dates[prevmaxi],
            "end_date"  : dates[prevmini]}
    
        return out_dict
    
    def _get_drawdown(self, df: pd.DataFrame) -> pd.DataFrame: 
    
        df_out = (pd.DataFrame.from_dict(
            data   = self.drawdown(df.value),
            orient = "index").
            T)
    
        return df_out
    
    def get_perform_results(self, df_combined: pd.DataFrame) -> pd.DataFrame: 
        
        df_drawdown = (df_combined.apply(
            lambda x: np.cumprod(1 + x) * 100).
            reset_index().
            melt(id_vars = "date").
            groupby("variable").
            apply(self._get_drawdown).
            reset_index().
            assign(drawdown = lambda x: (x.min_val - x.max_val) / x.max_val)
            [["variable", "drawdown"]])
        
        years = (df_combined.index.max() - df_combined.index.min()).days / 365.25
        
        df_cagr = (df_combined.apply(
            lambda x: (1 + x).prod() ** (1 / years) - 1).
            to_frame(name = "CAGR").
            reset_index().
            rename(columns = {"index": "variable"}))
                
        df_vol = (df_combined.agg(
            lambda x: x.std()).
            to_frame(name = "vol").
            reset_index().
            rename(columns = {"index": "variable"}))
        
        df_ratio = (df_combined.agg(
            lambda x: x.mean()).
            to_frame(name = "mean_rtn").
            reset_index().
            rename(columns = {"index": "variable"}).
            merge(right = df_drawdown, how = "inner", on = ["variable"]).
            assign(ratio = lambda x: np.abs((x.mean_rtn * 252) / x.drawdown))
            [["variable", "ratio"]])
        
        df_sharpe = (df_combined.agg(
            lambda x: x.mean() / x.std() * np.sqrt(252)).
            to_frame(name = "sharpe").
            reset_index().
            rename(columns = {"index": "variable"}))
        
        df_stats = (df_cagr.merge(
            right = df_drawdown, how = "inner", on = ["variable"]).
            merge(right = df_vol, how = "inner", on = ["variable"]).
            merge(right = df_ratio, how = "inner", on = ["variable"]).
            merge(right = df_sharpe, how = "inner", on = ["variable"]).
            rename(columns = {
                "drawdown": "MaxDD",
                "vol"     : "Ann. Vol.",
                "ratio"   : "Ann. Rtn / MaxDD",
                "sharpe"  : "Sharpe"}).
            set_index("variable"))
        
        return df_stats