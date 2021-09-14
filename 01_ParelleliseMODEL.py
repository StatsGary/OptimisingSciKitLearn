# =============================================================================
# Parellel model processing with Scikit learn
# =============================================================================
from time import time
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
import os
import matplotlib.pyplot as plt

# Get CPU cores
cpu_count = os.cpu_count()
print(f"This machine has {cpu_count} cores")
reserved_cpu = 1
final_cpu = int(cpu_count - reserved_cpu)
print("Saving one CPU so my PC does not lock up")

# =============================================================================
# Parellel model training
# =============================================================================
# Create the dataset
X, Y = make_classification(n_samples=10000,
                           n_features=70, 
                           n_informative=65,
                           n_redundant=5)

# Function to run model and create a timer and list output of the model
def run_model(model, X, Y):
    start = time()
    model = model.fit(X,Y)
    end = time()
    result = end-start
    print("[MODEL INFO] The model ran in {:.2f}".format(result))
    return [model, result]

# =============================================================================
# Single core example
# =============================================================================
#Initialise the model
model_singlecore = RandomForestClassifier(n_estimators=500, n_jobs=1)
# Create a timer and fit the model
model_single_list = run_model(model_singlecore, X, Y)
# =============================================================================
# Multicore example
# =============================================================================
model_multi = RandomForestClassifier(n_estimators=500, n_jobs=final_cpu)
model_ran = run_model(model_multi, X, Y)
# =============================================================================
# Iterate cores
# =============================================================================
results_list = list()
n_cores = [1,2,3,4,5,6,7,8]

for model_it in n_cores:
    model = RandomForestClassifier(n_estimators=500, n_jobs=model_it)
    model_fit = run_model(model, X, Y)
    results = model_fit[1] #Slice the index of the model to get the fit
    results_list.append(results)
    
# Generate plot of results
def plot_results(x_val, y_val):
   plt.plot(x_val, y_val, color="blue", linestyle="--", marker="o")
   plt.xlabel("Run time (secs)")
   plt.ylabel("Number of cores")
   plt.show() 
   
plot_results(n_cores, results_list)

    
    
    
    
    
    