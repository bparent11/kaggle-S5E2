import pandas as pd
import numpy as np
import json

def preprocess(df, submission=False):
    global target_encoding_dict
    df.loc[df['Laptop Compartment'] == "Yes", "Laptop Compartment"] = int(1)
    df.loc[df['Laptop Compartment'] == "No", "Laptop Compartment"] = int(0)
    df['Laptop Compartment'] = df['Laptop Compartment'].astype(float)

    df.loc[df['Waterproof'] == "Yes", "Waterproof"] = int(1)
    df.loc[df['Waterproof'] == "No", "Waterproof"] = int(0)
    df['Waterproof'] = df['Waterproof'].astype(float)

    df[df.select_dtypes(include='object').columns] = df.select_dtypes(include='object').fillna('Inconnu')
    df[df.select_dtypes(include='number').columns] = df.select_dtypes(include='number').fillna(-1)

    numerical_columns = df.select_dtypes(include='number')
    categorical_columns = df.select_dtypes(include='object')

    for column in numerical_columns:
        if (df[column] == -1).sum() > 0:
            missing_column_name = f"{column}_Missing"
            df[missing_column_name] = (df[column] == -1).astype(int)

    #for column in categorical_columns:
    #    if (df[column] == 'Inconnu').sum() > 0:
    #        missing_column_name = f"{column}_Missing"
    #        df[missing_column_name] = (df[column] == 'Inconnu').astype(int)

    # because that's already done with line 6 and OHE.

    """TARGET ENCODING | Mean, Count, STD and VAR"""

    if submission == False:
        target_encoding_dict = {}
        for column in categorical_columns:
            gb = df.groupby(column).agg(
                Price_mean = ("Price", "mean"),
                Count = ("Price", "count"),
                STD = ("Price", "std"),
                VAR = ("Price", "var")
            )

            for index in gb.index:
                key = f"{column}_{index}"
                mean = gb.loc[index, "Price_mean"]
                count = gb.loc[index, "Count"]
                stdeviation = gb.loc[index, "STD"]
                var = gb.loc[index, "VAR"]

                target_encoding_dict[key] = [mean, count, stdeviation, var]
                #print(key)
                #print(target_encoding_dict[key])
            
            # Renommer les colonnes pour éviter les conflits
            gb.columns = [f"{column}_{col}" for col in gb.columns]

            # Joindre les résultats au DataFrame principal
            df = df.merge(gb, on=column, how='left')


    if submission == True:
        for key in target_encoding_dict:
            initial_column = key.split("_")[0]
            corresponding_value = key.split("_")[1]
            
            iteration=0
            for aggfunc in ["Price_mean", "Count", "STD", "VAR"]:
                df.loc[df[initial_column] == corresponding_value, f"{initial_column}_{aggfunc}"] = target_encoding_dict[key][iteration]
                iteration += 1

    #with open('test_scores_sorted_cv=2_150k_lines.json', 'r') as json_file:
    #    relevant_combinations = json.load(json_file)
    #
    #combinations_to_add = list(relevant_combinations.keys())[:20]
#
    #for combination in combinations_to_add:
    #    separation = combination.split("xxx")
    #    first_col = separation[0]
    #    second_col = separation[1]
    #    df[combination] = df[first_col] * df[second_col]

    return df