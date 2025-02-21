from itertools import product, combinations
from sklearn.model_selection import cross_val_score, KFold
import xgboost as xgb
import time

# combinations = list(product(X.columns, X.columns)) # cartesian product
columns_combinations = list(combinations(X.columns, 2)) # combinations without duplicates
print(len(columns_combinations))
combinations_filtered = list(filter(lambda tuple: tuple[0].split("_")[0] != tuple[1].split("_")[0], columns_combinations)) #filtered

# Afficher les combinaisons filtrées et leur nombre
print(combinations_filtered)
print(len(combinations_filtered))

combination_names = [f"{tpl[0]}xxx{tpl[1]}" for tpl in combinations_filtered]
print(combination_names)
print(len(combination_names))

test_score_dict = {}
KF = KFold(n_splits=2, shuffle=True, random_state=42)

for index, new_col in enumerate(combination_names, len(combination_names)-len(combination_names)):
    start_time = time.time()
    temporary_X = X.copy()
    separation = new_col.split("xxx")
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

new_col = "no_column_added"
model = xgb.XGBRegressor()
cv_results = cross_val_score(model, X, y, cv=KF, scoring='neg_root_mean_squared_error', n_jobs=-1)
test_score_dict[new_col] = cv_results.mean()


sorted_dict = dict(sorted(test_score_dict.items(), key=lambda item: item[1]))

import json
file_path = 'test_scores_sorted_cv=4_300k_lines.json'
with open(file_path, 'w') as json_file:
    json.dump(sorted_dict, json_file, indent=4)


## 44 minutes to do it