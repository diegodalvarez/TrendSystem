# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 01:11:52 2026

@author: Diego
"""

import os
import numpy as np
import pandas as pd
from   typing import Callable
from   pykalman import KalmanFilter

from tqdm import tqdm
tqdm.pandas()

class KalmanFilterSignal:
    
    def __init__(self) -> None: 
        
        self.src_path  = os.getcwd()
        self.root_path = os.path.abspath(os.path.join(self.src_path, ".."))
        self.data_path = os.path.join(self.root_path, "data")
        self.fut_path  = os.path.join(self.data_path, "FuturesData")
        self.kf_path   = os.path.join(self.data_path, "KalmanFilter")
        
        if os.path.exists(self.kf_path) == False: 
            os.makedirs(self.kf_path)
            
        self.dt_window    = 10
        self.trend_window = 100
    
    def kalman_local_linear(self, df: pd.DataFrame) -> pd.DataFrame:
    
        price = df.sort_index().px_val 
        y     = np.log(price).values
        
        kf = KalmanFilter(
    
            transition_matrices=np.array([
                [1,1],
                [0,1]
            ]),
    
            observation_matrices=np.array([
                [1,0]
            ]),
    
            transition_covariance=np.array([
                [1e-5,0],
                [0,1e-6]
            ]),
    
            observation_covariance=1e-3,
    
            initial_state_mean=np.array([
                y[0],
                0
            ]),
    
            initial_state_covariance=np.eye(2)
    
        )
    
        state_means, state_covs = kf.filter(y)
    
        df = (pd.DataFrame(
    
            index = price.index,
    
            data={
    
                "log_price": y,
                "trend"    : state_means[:,0],
                "slope"    : state_means[:,1],
                "price"    : price}))
    
        return df

    def kalman_mean_reverting(
            self,
            df : pd.DataFrame, 
            phi: float = 0.98) -> pd.DataFrame:
    
        price = df.sort_index().px_val 
        y     = np.log(price).values
    
        kf = KalmanFilter(
    
            transition_matrices=np.array([
                [1,1],
                [0,phi]
            ]),
    
            observation_matrices=np.array([
                [1,0]
            ]),
    
            transition_covariance=np.array([
                [1e-5,0],
                [0,1e-6]
            ]),
    
            observation_covariance=1e-3,
    
            initial_state_mean=np.array([
                y[0],
                0
            ]),
    
            initial_state_covariance=np.eye(2)
    
        )
    
        state_means, state_covs = kf.filter(y)
    
        df = (pd.DataFrame(
    
            index=price.index,
    
            data={
    
                "log_price": y,
                "trend"    : state_means[:,0],
                "slope"    : state_means[:,1],
                "price"    : price}))
    
        return df
    
    def kalman_acceleration(self, df: pd.DataFrame) -> pd.DataFrame:
    
        price = df.sort_index().px_val
        y     = np.log(price).values
    
        dt = 1.0
    
        kf = KalmanFilter(
    
            transition_matrices=np.array([
                [1, dt, 0.5 * dt**2],
                [0, 1, dt],
                [0, 0, 1]
    
            ]),
    
            observation_matrices=np.array([
                [1, 0, 0]
            ]),
    
            transition_covariance=np.array([
                [1e-5, 0,    0],
                [0,    1e-6, 0],
                [0,    0,    1e-7]
    
            ]),
    
            observation_covariance=1e-3,
    
            initial_state_mean=np.array([
                y[0],
                0,
                0]),
    
            initial_state_covariance=np.eye(3))
    
        state_means, state_covs = kf.filter(y)
    
        df = pd.DataFrame(
    
            index=price.index,
    
            data={
    
                "log_price"   : y,
                "trend"       : state_means[:,0],
                "slope"       : state_means[:,1],
                "acceleration": state_means[:,2],
                "price"       : price})
    
        return df
    
    def _run_and_save(
            self, 
            df      : pd.DataFrame, 
            out_path: str, 
            func    : Callable[[pd.DataFrame], pd.DataFrame], 
            verbose : bool = True,
            save    : bool = True) -> None:
        
        if verbose: 
            print("Starting {}".format(func.__name__))
            
        if os.path.exists(out_path) == True: 
            if verbose: 
                print("Already have data\n")
            
            return None
        
        if verbose: 
            print("Running {}".format(func.__name__))
            
        df_out = (df.groupby(
            "ticker").
            progress_apply(lambda group: func(group)).
            reset_index())
        
        if save == False: 
            print("\n",df_out)
            return None
        
        if verbose: 
            print("\nSaving data at\n{} at {}\n".format(
                self.kf_path,
                out_path.split("\\")[-1]))
        
        df_out.to_parquet(path = out_path, engine = "pyarrow")
            
    def generate_kf_signal(self, verbose: bool = True) -> None: 
        
        fut_path = os.path.join(self.fut_path, "PrepFuturesPX.parquet")
        df_fut   = (pd.read_parquet(
            path = fut_path, engine = "pyarrow").
            set_index("date"))

        path = os.path.join(self.kf_path, "KalmanLocalTrend.parquet")
        self._run_and_save(df_fut, path, self.kalman_local_linear, verbose)
    
        path = os.path.join(self.kf_path, "KalmanMRSlope.parquet")
        self._run_and_save(df_fut, path, self.kalman_mean_reverting, verbose)
        
def main() -> None: 
            
    kf_filter = KalmanFilterSignal()
    kf_filter.generate_kf_signal()
    
if __name__ == "__main__": main()