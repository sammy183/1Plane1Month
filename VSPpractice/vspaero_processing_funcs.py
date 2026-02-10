# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 12:22:00 2025

For VSPaero result postprocessing with pandas

@author: NASSAS
"""

import pandas as pd

def extract_vspaero_data(path):
    '''
    Path to CSV exported by VSPAERO
    We want to skip the VSPAERO_History and go straight to VSPAERO_Polar
    
    ONLY works for angle of attack (alpha) sweeps for now!
    The alpha values end up as the row indexes, every other value as columns
    
    For plotting vs alpha, simply call:
        plt.plot(df[KEY]) 
    where KEY includes 'L_D', 'CDtot', 'CLtot', 'CDi', and more
    
    When wanting to plot vs other x-values call them directly, like:
        L_D = df['L_D'].values
        CLtot = df['CLtot'].values
        plt.plot(CLtot, L_D)
    
    '''
    startrow = 0 
    endrow = 0 
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            parts = [p.strip() for p in line.split(',')]
            var = parts[1] # B column
    
            # Skip header-like lines
            if var == 'VSPAERO_Polar':
                startrow = i
                
            if var == 'VSPAERO_Load':
                endrow = i
                break
    
    df = pd.read_csv(path, skiprows = startrow + 4, nrows = endrow-startrow - 5)
    
    # Rename first column to "Variable"
    df = df.rename(columns={df.columns[0]: "Variable"})

    # Melt all numeric columns into long format
    df_long = df.melt(id_vars="Variable", var_name="Alpha", value_name="Value")

    # Pivot so each variable becomes a column
    df_pivot = df_long.pivot(index="Alpha", columns="Variable", values="Value")

    # Ensure numeric sorting of station index
    df_pivot.index = pd.to_numeric(df_pivot.index)
    df_pivot = df_pivot.sort_index()
    
    return(df_pivot)