# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 09:47:19 2026

@author: Diego
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm


from tqdm import tqdm
tqdm.pandas()

class CreditHPFilter:
    
    def __init__(self) -> None: 
        
        self.sig_src_path = os.getcwd()
        self.src_path     = os.path.abspath(os.path.join(self.sig_src_path, ".."))
        self.repo_path    = os.path.abspath(os.path.join(self.src_path, ".."))
        self.data_path    = os.path.join(self.repo_path, "data")
        self.cred_path    = os.path.join(self.data_path, "CreditData")
        self.sig_path     = os.path.join(self.data_path, "Signals")

    def _get_hp_filter(
        self,
        df        : pd.DataFrame        ,
        lamb      : int = 1_600,
        hp_window : int = 500,
        vol_target: float = 0.1,
        vol_window: int = 100) -> pd.DataFrame:

        df_tmp   = df.sort_index().copy()
        hp_trend = pd.Series(np.nan, index=df_tmp.index, dtype=float)
        spread   = df_tmp["spread"]

        for i in range(hp_window - 1, len(df_tmp)):
            window = spread.iloc[i - hp_window + 1:i + 1]

            if window.isna().any():
                continue

            _, trend = sm.tsa.filters.hpfilter(
                x=window,
                lamb=lamb)

            hp_trend.iloc[i] = trend.iloc[-1]

        df_out = (
            df_tmp
            .assign(
                hp_trend      = hp_trend,
                hp_cycle      = lambda x: x.spread - x.hp_trend,
                hp_trend_diff = lambda x: x.hp_trend.diff(),
                signal        = lambda x: -np.sign(x.hp_trend_diff.shift(1)),
                px_rtn        = lambda x: x.price.pct_change(),
                signal_rtn    = lambda x: x.signal * x.px_rtn,
                signal_vol    = lambda x: (x.signal_rtn.ewm(span=vol_window, adjust=False).std().shift(1)),
                leverage      = lambda x: (vol_target / (x.signal_vol * np.sqrt(252))),
                vol_rtn       = lambda x: x.signal_rtn * x.leverage))

        return df_out

    def fit_hp_filter(self, verbose: bool = True) -> None: 
        
        '''
        Applied to Credit CDS Indices for now
        '''
        
        if verbose: print("Getting CDS Indices Generic HPFilter")
        
        out_path = os.path.join(self.sig_path, "SocGenCreditGenericHP.parquet")
        
        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None
        
        path  = os.path.join(self.cred_path, "PrepCDS.parquet")
        df_hp = (pd
                .read_parquet(path = path, engine = "pyarrow")
                .dropna()
                .assign(log_spread = lambda x: np.log(x.spread))
                .set_index("date")
                .loc[lambda x: x.SecurityGroup == x.SecurityGroup.min()]
                .groupby("SecurityGroup")
                .apply(self._get_hp_filter))
        
        display(df_hp)
        return-1

        if verbose: print("Saving Data\n")
        df_hp.to_parquet(path = out_path, engine = "pyarrow")

def main() -> None: 
    CreditHPFilter().fit_hp_filter()

if __name__ == "__main__": main()