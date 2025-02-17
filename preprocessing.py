import pandas as pd
import numpy as np

def preprocess(df):
    
    df[df.select_dtypes(include='object').columns] = df.select_dtypes(include='object').fillna('Inconnu')
    df[df.select_dtypes(include='number').columns] = df.select_dtypes(include='number').fillna(-1)

    return df