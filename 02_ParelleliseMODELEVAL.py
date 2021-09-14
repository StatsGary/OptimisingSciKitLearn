# =============================================================================
# 03 Parallel Model Evaluation
# =============================================================================
from time import time
from sklearn.datasets import make_classification
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
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

model = RandomForestClassifier(n_estimators=100, n_jobs=1)
#------------------------------------------------------------------------------
# Build function to optimise cv_score
#------------------------------------------------------------------------------
def cross_val_optimised(model, X, Y, workers):
    # Define the evaluation procedure
    strat_cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3)
    #Start the timer
    start = time()
    model_result = cross_val_score(model, X, Y, scoring="accuracy",
                    cv=strat_cv, n_jobs=workers)
    end = time()
    result = end - start
    print("[MODEL EVALUATION INFO] run time is {:.3f}".format(result))
    return [result, model_result]
    
    
#------------------------------------------------------------------------------
# Loop each call of function and append result to 
#------------------------------------------------------------------------------    
results_list = list()
# Create a list with number of cores
cores = [core for core in range(1, final_cpu+1)]
# Get the cores list dynamically, as previous example was hard coded
for n_cor in cores:
    cv_optim = cross_val_optimised(model, X, Y, workers=n_cor)
    result = cv_optim[0]
    results_list.append(result)
     
# Generate plot of results
def plot_results(x_val, y_val):
   plt.plot(x_val, y_val, color="blue", linestyle="--", marker="o")
   plt.xlabel("Number of cores")
   plt.ylabel("Run time (secs)")
   plt.show() 
   
plot_results(cores, results_list)




