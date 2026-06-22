# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 01:56:15 2026

@author: Diego
"""

import os
import pandas as pd

class SignalGenerator:
    
    def __init__(self) -> None: 
        
        self.src_path  = os.getcwd()
        self.root_path = os.path.abspath(os.path.join(self.src_path, ".."))
        self.data_path = os.path.join(self.root_path, "data")
        self.fut_path  = os.path.join(self.data_path, "FuturesData")
        self.gen_path  = os.path.join(self.data_path, "GenericSignal")
        
        if os.path.exists(self.gen_path) == False: 
            os.makedirs(self.gen_path)
        
    def signal_preperation(
            self, 
            signal_adj: int = 10, 
            window    : int = 100,
            verbose   : bool = True) -> None: 
        
        '''
        Function for preparing data for signal generation
        '''
        
        if verbose: 
            print("Generating the signal preparation")
        
        out_path = os.path.join(self.gen_path, "FuturesDt.parquet")
        if os.path.exists(out_path) == True: 
            if verbose: 
                print("Already have data\n")
        
            return None
        
        if verbose: 
            print("Calculating signal preparation")
        
        fut_path = os.path.join(self.fut_path, "PrepFuturesPX.parquet")
        df_prep  = (pd.read_parquet(
            path = fut_path, engine = "pyarrow").
            pivot(index = "date", columns = "ticker", values = "px_val").
            diff().
            apply(lambda x: (
                x / x.ewm(span = signal_adj, adjust = False).std())).
            apply(
                lambda x: x.ewm(span = window, adjust = False).mean()).
            reset_index().
            melt(id_vars = "date", value_name = "dt").
            dropna())
        
        if verbose: 
            print("Saving data\n")
            
        df_prep.to_parquet(path = out_path, engine = "pyarrow")
        
    def generate_single_ewma(
        self,
        signal_window: int  = 100,
        verbose      : bool = True) -> None:
        
        '''
        Generating single EWMA signal
        '''
        
        out_path = os.path.join(self.gen_path, "GenericSignal.parquet")
        
        if verbose: 
            print("Getting single ewma signal of {}".format(
                signal_window))
            
        if os.path.exists(out_path) == True:
            if verbose: 
                print("Already have data\n")
                
            return None
        
        if verbose: 
            print("Generating signal")
        
        in_path = os.path.join(self.gen_path, "FuturesDt.parquet")
        df_out  = (pd.read_parquet(
            path = in_path, engine = "pyarrow").
            pivot(index = "date", columns = "ticker", values = "dt").
            apply(lambda x: x.ewm(
                span = signal_window, adjust = False).mean()).
            reset_index().
            melt(id_vars = "date", value_name = "signal").
            dropna())
        
        if verbose: 
            print("Saving signal")
            
        df_out.to_parquet(path = out_path, engine = "pyarrow")

def main() -> None: 
            
    signal_generator = SignalGenerator()
    signal_generator.signal_preperation()
    signal_generator.generate_single_ewma()
    
if __name__ == "__main__": main()