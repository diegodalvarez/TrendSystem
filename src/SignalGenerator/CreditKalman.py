# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 11:43:37 2026

@author: Diego
"""

import os
import numpy as np
import pandas as pd

from pykalman import KalmanFilter
from scipy.stats import linregress

from tqdm import tqdm
tqdm.pandas()

class CreditKalmanFilter:
    
    def __init__(self) -> None: 
        
        self.sig_src_path = os.getcwd()
        self.src_path     = os.path.abspath(os.path.join(self.sig_src_path, ".."))
        self.repo_path    = os.path.abspath(os.path.join(self.src_path, ".."))
        self.data_path    = os.path.join(self.repo_path, "data")
        self.cred_path    = os.path.join(self.data_path, "CreditData")
        self.sig_path     = os.path.join(self.data_path, "Signals")
        
        self.smooth_windows = {
            "short_term" : 21,
            "medium_term": 21 * 3}
        
    def _get_kf(self, df: pd.DataFrame) -> pd.DataFrame: 
                
        y  = df.log_spread.values
        kf = (KalmanFilter(
            transition_matrices      = [1],
            observation_matrices     = [1],
            transition_covariance    = 0.01,
            observation_covariance   = 1.0,
            initial_state_mean       = y[0],
            initial_state_covariance = 1.0))
        
        state_means, state_covariances = kf.filter(y)
        df_out = df.assign(state_means = state_means)
        return df_out
    
    def _slope(self, x: np.array) -> float:
        t = np.arange(len(x))
        return linregress(t, x).slope
    
    def _get_kf_trend(self, df: pd.DataFrame) -> pd.DataFrame: 
        
        for window in self.smooth_windows.keys():
            
            df[window + "_market"] = (df
                       ["state_means"]
                       .rolling(self.smooth_windows[window])
                       .apply(self._slope, raw = True))
            
            df[window + "_residual"] = (df
                                        ["residual"]
                                        .rolling(self.smooth_windows[window])
                                        .apply(self._slope, raw = True))
            
        return df
        
        
    def fit_generic_kalman(self, verbose: bool = True) -> None: 
        
        '''
        Applied to Credit CDS Indices for now
        '''
        
        if verbose: print("Getting CDS Indices Generic Kalman")
        
        out_path = os.path.join(self.sig_path, "SocGenCreditGenericKalman.parquet")
        
        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None
        
        path  = os.path.join(self.cred_path, "PrepCDS.parquet")
        df_kf = (pd
                .read_parquet(path = path, engine = "pyarrow")
                .drop(columns = ["price"])
                .dropna()
                .assign(log_spread = lambda x: np.log(x.spread))
                .set_index("date")
                .loc[lambda x: x.SecurityGroup == x.SecurityGroup.min()]
                .groupby("SecurityGroup")
                .apply(self._get_kf)
                #.progress_apply(lambda group: self._get_kf(group))
                .reset_index()
                .assign(residual = lambda x: x.state_means - x.log_spread)
                )
        
        df_out = (df_kf
                .set_index("date")
                #.loc[lambda x: x.SecurityGroup == x.SecurityGroup.min()]
                .groupby("SecurityGroup")
                .apply(self._get_kf_trend)
                .reset_index())
        
        if verbose: print("Saving data\n")
        df_out.to_parquet(path = out_path, engine = "pyarrow")
        
def main() -> None: 
        
    CreditKalmanFilter().fit_generic_kalman()
    
if __name__ == "__main__": main()