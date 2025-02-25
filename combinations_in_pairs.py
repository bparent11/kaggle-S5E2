from itertools import product, combinations
from sklearn.model_selection import cross_val_score, KFold
import xgboost as xgb
import time


def finding_best_combinations_in_pairs(X, y):
    
    # combinations = list(product(X.columns, X.columns)) # cartesian product
    columns_combinations = list(combinations(X.columns, 2)) # combinations without duplicates
    combinations_filtered = list(filter(lambda tuple: tuple[0].split("_")[0] != tuple[1].split("_")[0], columns_combinations)) #filtered
    print(combinations_filtered)
    combination_names = [f"{tpl[0]}*{tpl[1]}" for tpl in combinations_filtered]

    test_score_dict = {}
    KF = KFold(n_splits=2, shuffle=True, random_state=42)

    print(f"Computation begins, there is {len(combinations_filtered)} combinations to compute !")
    for index, new_col in enumerate(combination_names, len(combination_names)-len(combination_names)):
        start_time = time.time()
        temporary_X = X.copy()
        separation = new_col.split("*")
        first_col = separation[0]
        second_col = separation[1]
        temporary_X[new_col] = temporary_X[first_col] * temporary_X[second_col]
        model = xgb.XGBRegressor()
        cv_results = cross_val_score(model, temporary_X, y, cv=KF, scoring='neg_root_mean_squared_error', n_jobs=-1)
        print(f"cross validation {index + 1} / {len(combination_names)} done !")
        test_score_dict[new_col] = cv_results.mean()

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Temps écoulé : {elapsed_time:.6f} secondes")

    print("Final compute, without adding any combination !")
    new_col = "no_column_added"
    model = xgb.XGBRegressor()
    cv_results = cross_val_score(model, X, y, cv=KF, scoring='neg_root_mean_squared_error', n_jobs=-1)
    test_score_dict[new_col] = cv_results.mean()

    sorted_dict = dict(sorted(test_score_dict.items(), key=lambda item: item[1], reverse=True))

    import json
    file_path = 'best_combinations_22_02.json'
    with open(file_path, 'w') as json_file:
        json.dump(sorted_dict, json_file, indent=4)

    return test_score_dict