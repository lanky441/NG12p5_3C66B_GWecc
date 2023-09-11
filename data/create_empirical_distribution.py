"""
This code reads the chains from NANOGrav 12.5 year GWB search chains and creates
empirical distributions for the individual pulsar red noise and CURN parmaters.
"""

import numpy as np
import pickle
import json
from enterprise_extensions.empirical_distr import EmpiricalDistribution2D

# Reading chain from "Varying spectral index, 5 frequencies" run from NG 12.5 yr GWB analysis
# The chain can be found at https://nanograv.org/science/data/125-year-stochastic-gravitational-wave-background-search
chain = np.genfromtxt('NG12p5_chain_5f_free_gamma.txt')
print(f"Chain shape = {chain.shape}")

burn = 30000

# Thinning the chain by a factor of 10 and saving it
chain_thinned = chain[burn::10,:]
print(f"Thinned chain shape = {chain_thinned.shape}")
np.savetxt("NG12p5_chain_5f_free_gamma_thinned.txt", chain_thinned)

# Reading the names of the parameters present in the chain
params = np.genfromtxt('NG12p5_chain_5f_free_gamma_params.txt', dtype='str')

emp_dists = []
noise_median = {}

# Calculating the medain values from the posterior distribution of each parameter
for i, param in enumerate(params):
    median = np.median(chain[burn:,i])
    noise_median[param] = median

# Writing the medain values of each parameter in a json file
with open("noise_param_median_5f.json", "w") as outfile:
    json.dump(noise_median, outfile, indent=4)


nparams = len(params)
print(f"Number of params = {nparams}")
num_dists = int(nparams/2)

for i in range(num_dists):
    ndist = i+1
    samples = chain[burn:,2*i:2*ndist]
    param_names = params[2*i:2*ndist]
    print(param_names)
    
    gamma_bins = np.linspace(0, 7, 25)
    log10_A_bins = np.linspace(-20, -11, 25)
    
    # Creating and appending the 2D empirical distributions for individaul pulsar RN and GWB parameters
    emp_dists.append(EmpiricalDistribution2D(param_names, samples.T, 
                                             bins=[gamma_bins, log10_A_bins]))

# Dumping the list of empirical distributions in a pickle file
with open('empirical_distributions_5f.pkl', 'wb') as emp_dist_file:
    pickle.dump(emp_dists, emp_dist_file)