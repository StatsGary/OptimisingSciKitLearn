# =============================================================================
# 03 Parallel Hyperparameter Tuning
# =============================================================================
from time import time
from sklearn.datasets import make_classification
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import os
import matplotlib.pyplot as plt
# Get CPU cores
cpu_count = os.cpu_count()
print(f"This machine has {cpu_count} cores")
reserved_cpu = 1
final_cpu = int(cpu_count - reserved_cpu)
print("Saving one CPU so my PC does not lock up")

#------------------------------------------------------------------------------
# Create the dataset
#------------------------------------------------------------------------------
X, Y = make_classification(n_samples=1000,
                           n_features=20, 
                           n_informative=15,
                           n_redundant=5)
#------------------------------------------------------------------------------
# Train model on max number of CPUs
#------------------------------------------------------------------------------
rf_model = RandomForestClassifier(n_estimators=100, n_jobs=1)

#------------------------------------------------------------------------------
# Grid search hyperparameter tuning
#------------------------------------------------------------------------------
# Do a K fold cross validation split
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3)
grid = dict()
grid['max_features'] = [1,2,3,4,5]
# Define the grid 

#------------------------------------------------------------------------------
# Create grid search function
#------------------------------------------------------------------------------
def grid_search_cv(model, grid, n_jobs, cv):
    print("[GRID SEARCH INFO] Starting the grid search")
    grid_search = GridSearchCV(model, grid, n_jobs=n_jobs, cv=cv)
    start = time()
    grid_search_fit = grid_search.fit(X, Y)
    end = time()
    finish_time = end - start
    print("[GRID SEARCH EXECUTION TIME] the search finished in: {:.3f} seconds.".format(finish_time))
    return [finish_time, grid_search, grid_search_fit]


#------------------------------------------------------------------------------
# Benchmarking run time by depth of search and number of worker ants
#-----------------------------------------------------------------------------
results_list = list()
# Create a list with number of cores
cores = [core for core in range(1, final_cpu+1)]
# Get the cores list dynamically, as previous example was hard coded
#result = grid_search_cv(rf_model, grid=grid, n_jobs=2, cv=cv)
for n_cor in cores:
    search = grid_search_cv(rf_model, grid=grid, n_jobs=n_cor, cv=cv)
    result = search[0]
    results_list.append(result)

# Generate plot of results
def plot_results(x_val, y_val):
   plt.plot(x_val, y_val, color="blue", linestyle="--", marker="o")
   plt.xlabel("Number of cores")
   plt.ylabel("Run time (secs)")
   plt.show() 
   
plot_results(cores, results_list)

    






