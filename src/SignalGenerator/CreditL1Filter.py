# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 10:13:32 2026

@author: Diego
"""

import os
import numpy as np
import pandas as pd

import cvxpy as cp
import scipy as scipy
import cvxopt as cvxopt 

class L1CreditFilter:
    
    def __init__(self) -> None: 
        
        self.sig_src_path = os.getcwd()
        self.src_path     = os.path.abspath(os.path.join(self.sig_src_path, ".."))
        self.repo_path    = os.path.abspath(os.path.join(self.src_path, ".."))
        self.data_path    = os.path.join(self.repo_path, "data")
        self.cred_path    = os.path.join(self.data_path, "CreditData")
        self.sig_path     = os.path.join(self.data_path, "Signals")
        
    def l1_filter(self, verbose: bool = True, vlambda: int = 50) -> None: 
        
        '''
        Applied to Credit CDS Indices for now
        '''
        
        if verbose: print("Getting CDS Indices Generic L1 Filter")
        
        out_path = os.path.join(self.sig_path, "SocGenCreditGenericL1.parquet")
        
        if os.path.exists(out_path):
            if verbose: print("Already have data\n")
            return None
        
        path  = os.path.join(self.cred_path, "PrepCDS.parquet")
        df_l1 = (pd
                .read_parquet(path = path, engine = "pyarrow")
                .dropna()
                .assign(log_spread = lambda x: np.log(x.spread))
                .set_index("date"))
        
        df_l1_lists = []
        
        groups = df_l1.SecurityGroup.drop_duplicates().sort_values().to_list()
        for group in groups: 
            
            try:
                
                df_tmp = df_l1.loc[lambda x: x.SecurityGroup == group]
                y      = df_tmp.log_spread.to_numpy(dtype = float)
                n      = y.size
                
                e = np.ones((1, n))
                D = scipy.sparse.diags_array([e, -2*e, e], offsets = range(3), shape = (n-2, n))
            
                x    = cp.Variable(shape = n)
                obj  = cp.Minimize(0.5 * cp.sum_squares(y - x) + vlambda * cp.norm(D * x, 1))
                prob = cp.Problem(obj)
                
                prob.solve(solver = cp.CVXOPT, verbose = True)
                
                df_add = df_tmp.assign(l1 = x.value)
                df_l1_lists.append(df_add)
                
            except: 
                if verbose: print("{} Failed".format(group))
                pass 
            
        df_out = pd.concat(df_l1_lists)
        if verbose: print("Saving data\n")
        df_out.to_parquet(path = out_path, engine = "pyarrow")
        
L1CreditFilter().l1_filter()
        