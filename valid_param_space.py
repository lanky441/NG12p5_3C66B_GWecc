# Calculate for which part of the parameter space, our prior is valid
import numpy as np
import matplotlib.pyplot as plt
import json

import enterprise_gwecc
from juliacall import Main as jl

from enterprise.signals.parameter import Uniform


target_params = json.load(open("data/3c66b_params.json", "r"))



def validate(log10_A, eta, e0, gwdist = target_params["gwdist"], log10_F = target_params["log10_F"], 
             tref = target_params["tref"], tmax = 57933.45642396011 * 86400):
    valid, msg = jl.validate_params_target(log10_A, eta, log10_F, e0, gwdist, tref, tmax,)
    if valid: return 1
    return 0

log10_A = Uniform(-12, -6)
eta = Uniform(0.001, 0.25)
e0 = Uniform(0.001, 0.8)

log10_A_param = log10_A('log10_A_3C66B')
eta_param = eta('eta_3C66B')
e0_param = e0('e0_3C66B')

n = 200000

with open(f"valid_param_space.txt", "a") as vp:
    for i in range(n):
        a = log10_A_param.sample()
        et = eta_param.sample()
        e = e0_param.sample()
    
        valid = validate(a, et, e)
        vp.write(f"{e}    {et}    {a}   {valid}" + "\n")
        
        if i%1000 == 0:
            print(f"{100*i/n}% completed...")