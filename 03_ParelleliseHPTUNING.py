# =============================================================================
# 02 Parallel Hyperparameter Tuning
# =============================================================================
from time import time
from sklearn.datasets import make_classification
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import os

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
model = RandomForestClassifier(n_estimators=100, n_jobs=1)

#------------------------------------------------------------------------------
# Grid search hyperparameter tuning
#------------------------------------------------------------------------------
# Do a K fold cross validation split
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3)
grid = dict()
grid['max_features'] = [1,2,3,4,5]
# Define the grid 

def grid_search(model, grid, n_jobs, cv, **args):
    print("[GRID SEARCH INFO] Starting the grid search")
    grid_search = GridSearchCV(model=model, grid=grid, n_jobs=n_jobs, cv=cv)
    start = time()
    grid_search = grid_search.fit(X, Y)
    end = time()
    #Print out search time
    finish_time = end - start
    print("[GRID SEARCH EXECUTION TIME] the search finished in: {:.3f} seconds.".format(finish_time))
    return [grid_search, finish_time]


#------------------------------------------------------------------------------
# Benchmarking run time by depth of search and number of worker ants
#-----------------------------------------------------------------------------
results_list = list()
# Create a list with number of cores
cores = [core for core in range(1, final_cpu+1)]
# Get the cores list dynamically, as previous example was hard coded
for n_cor in cores:
    grid_search = grid_search(model=model, 
                              grid=grid, 
                              n_jobs=n_cor,
                              cv=cv)
    






