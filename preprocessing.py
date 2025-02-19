import pandas as pd
import numpy as np

def preprocess(df):

    df[df.select_dtypes(include='object').columns] = df.select_dtypes(include='object').fillna('Inconnu')
    df[df.select_dtypes(include='number').columns] = df.select_dtypes(include='number').fillna(-1)

    numerical_columns = df.select_dtypes(include='number')
    #categorical_columns = df.select_dtypes(include='object')

    for column in numerical_columns:
        if (df[column] == -1).sum() > 0:
            missing_column_name = f"{column}_Missing"
            df[missing_column_name] = (df[column] == -1).astype(int)

    #for column in categorical_columns:
    #    if (df[column] == 'Inconnu').sum() > 0:
    #        missing_column_name = f"{column}_Missing"
    #        df[missing_column_name] = (df[column] == 'Inconnu').astype(int)

    # because that's already done with line 6 and OHE.

    df.loc[df['Laptop Compartment'] == "Yes", "Laptop Compartment"] = int(1)
    df.loc[df['Laptop Compartment'] == "No", "Laptop Compartment"] = int(0)

    df.loc[df['Waterproof'] == "Yes", "Waterproof"] = int(1)
    df.loc[df['Waterproof'] == "No", "Waterproof"] = int(0)

    return df