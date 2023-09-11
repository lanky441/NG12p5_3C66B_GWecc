"""

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import corner

import wquantiles
import json
import argparse
import os
import sys
import astropy.constants as c

import enterprise_gwecc
from juliacall import Main as jl

# Tsun = 4.92720047e-6
Tsun = (c.GM_sun / c.c**3).to("s").value

def log2linear_weight(value, pmin, pmax):
    mask = np.logical_and(value>=pmin, value<=pmax)
    uniform_prior = 1 / (pmax-pmin)
    linexp_prior = np.log(10) * 10**value / (10**pmax - 10**pmin)
    
    weight = mask * linexp_prior / uniform_prior
    weight /= sum(weight)

    return weight

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--chain_folder", default="all_chains/chains_pulsar_4_Jun23/")
parser.add_argument("-b", "--burn", default=10000)

args = parser.parse_args()
chain_folder = args.chain_folder
burn = int(args.burn)

if "earth" in chain_folder:
    title_txt = 'Earth term-only search'
    name_txt = 'earth'
elif "pulsar" in chain_folder:
    title_txt = 'Earth + Pulsar term search'
    name_txt = 'pulsar'
    

if os.path.isfile(f"{chain_folder}/chain_1.txt"):
    chain_file = f"{chain_folder}/chain_1.txt"
elif os.path.isfile(f"{chain_folder}/chain_1.0.txt"):
    chain_file = f"{chain_folder}/chain_1.0.txt"
else:
    print("Could not find any chain file in the chain folder! Exiting!")

param_names = np.genfromtxt(f"{chain_folder}/params.txt", dtype=str)
target_params = json.load(open("data/3c66b_params.json", "r"))

chain = np.loadtxt(chain_file)
print(f"Chain shape = {chain.shape}")

e_idx = np.where(param_names == "gwecc_e0")[0][0]
eta_idx = np.where(param_names == "gwecc_eta")[0][0]
A_idx = np.where(param_names == "gwecc_log10_A")[0][0]

if False:
    corner.corner(chain[burn:, [e_idx, eta_idx, A_idx]], labels=["e0", "eta", "log10_A"],
                               color='C0', plot_contours=True)
    plt.title("Full posterior for e0, eta, and log10_S0")
    plt.show()

es = chain[burn:, e_idx]
etas = chain[burn:, eta_idx]
gwecc_log10_As = chain[burn:, A_idx]

mask = np.logical_and(es<0.5, etas>0.1)

if False:
    corner.corner(np.array([es[mask], etas[mask], gwecc_log10_As[mask]]).T, labels=["e0", "eta", "log10_A"],
                               color='C0', plot_contours=True)
    plt.title("Limited posterior for e0, eta, and log10_S0")
    plt.show()

# Calculate the weights to convert from log-uniform to uniform posterior
A_post_ws = log2linear_weight(gwecc_log10_As, -12, -6)

# plt.hist(10**gwecc_log10_As, bins=15, weights=A_post_ws)
# plt.xlabel("gwecc_S0 (reweighted)")
# plt.show()

# Calculate mass from log10_A, e0, and eta
Ms_gu = []
Mchs_gu = []
for gwecc_log_A, e, eta in zip(gwecc_log10_As, es, etas):
    M_object = jl.mass_from_gwdist(gwecc_log_A, target_params["log10_F"], e, target_params["gwdist"], eta)
    Ms_gu.append(M_object.m/Tsun)
    Mchs_gu.append(M_object.Mch/Tsun)

Ms = np.array(Ms_gu)
Mchs = np.array(Mchs_gu)
log10_Mchs = np.log10(Mchs)


print("Length of log10_Mch = ", len(Mchs))

print(f"95% upper limit on total mass for valid param range = {np.percentile(Ms[mask], 95)/1e9} 10**9 Msun")
print(f"95% upper limit on chirp mass for valid param range = {np.percentile(Mchs[mask], 95)/1e9}  10**9 Msun")
print(f"95% upper limit on S0 for valid param range = {1e9 * 10**np.percentile(gwecc_log10_As[mask], 95)} ns")

logS0_95 = wquantiles.quantile(gwecc_log10_As[mask], A_post_ws[mask], 0.95)
Mch_95 = wquantiles.quantile(Mchs[mask], A_post_ws[mask], 0.95)
log10Mch_95 = wquantiles.quantile(np.log10(Mchs[mask]), A_post_ws[mask], 0.95)

print(f"95% upper limit on reweighted S0 for valid param range = {1e9 * 10**logS0_95} ns")
print(f"95% upper limit on reweighted chirp mass for valid param range = {Mch_95/1e9} 10**9 Msun")
print(f"95% upper limit on reweighted chirp mass for valid param range = {10**(log10Mch_95)/1e9} 10**9 Msun")


# Calculate the uncertainty on S0 and Mch

log10_A_valid = gwecc_log10_As[mask]
A_post_ws_valid = A_post_ws[mask]
log10_Mch_valid = log10_Mchs[mask]

print("Valid samples = ", len(log10_A_valid))

log10_A_lims = []
log10_Mch_lims = []

sample_array = np.arange(len(log10_A_valid))
for i in range(10000):
    samples = np.random.choice(sample_array, size=10000)
    log10_A_lims.append(wquantiles.quantile(log10_A_valid[samples], A_post_ws_valid[samples], 0.95))
    log10_Mch_lims.append(wquantiles.quantile(log10_Mch_valid[samples], A_post_ws_valid[samples], 0.95))

A_lims = 10**np.array(log10_A_lims)*1e9
Mch_lims = 10**np.array(log10_Mch_lims)/1e9

plt.hist(A_lims, bins=20)
plt.xlabel("S0")
plt.show()


plt.hist(Mch_lims, bins=20)
plt.xlabel("Mch")
plt.show()

print(f"95% upper limit on reweighted S0 for valid param range = {np.mean(A_lims)} +/- {np.std(A_lims)} ns")
print(f"95% upper limit on reweighted chirp mass for valid param range = {np.mean(Mch_lims)} +/- {np.std(Mch_lims)} 10**9 Msun")

#

# plt.hist(log10_Ms[mask], bins=15)
# plt.xlabel("log10_M")
# plt.show()

# plt.hist(log10_Mchs[mask], bins=15)
# plt.xlabel("log10_Mch")
# plt.show()

# plt.hist(10**log10_Mchs[mask], bins=15, weights= A_post_ws[mask])
# plt.xlabel("Reweighted Mch")
# plt.show()



ep, etap, logAp, valid = np.genfromtxt("valid_param_space.txt").transpose()

ev = ep[valid==1]
etav = etap[valid==1]
logAv = logAp[valid==1]

# Calculate the weights to convert from log-uniform to uniform prior
A_prior_ws = log2linear_weight(logAv, -12, -6)

# plt.hist(logAv, bins=15, weights=None)
# plt.show()

# print(len(ev), len(etav), len(logAv))

# plt.hist2d(ev, etav, bins=8)
# plt.show()


# Define the number of bins for the first two parameters
num_bins = 8

# Calculate the bin indices for the first two parameters
e_bins = np.linspace(0.001, 0.8, num_bins + 1)
eta_bins = np.linspace(0.001, 0.25, num_bins + 1)

# print(e_bins, eta_bins)

# Digitize the first two parameters to obtain the bin indices
e_bin_indices = np.digitize(es, e_bins)
eta_bin_indices = np.digitize(etas, eta_bins)

ev_bin_indices = np.digitize(ev, e_bins)
etav_bin_indices = np.digitize(etav, eta_bins)


# Initialize an empty array to store the percentile values for each bin
percentiles = np.zeros((num_bins, num_bins))
percentiles_Mch = np.zeros((num_bins, num_bins))
valid_percentiles = np.zeros((num_bins, num_bins))

quantiles = np.zeros((num_bins, num_bins))
quantiles_Mch = np.zeros((num_bins, num_bins))
valid_quantiles = np.zeros((num_bins, num_bins))


# Calculate the 95th percentile value for each bin
fig = plt.figure(figsize=(12, 12))
for i in range(num_bins):
    for j in range(num_bins):
        # Select the data points that fall within the current bin
        mask = (e_bin_indices == i + 1) & (eta_bin_indices == j + 1)
        # Calculate the 95th percentile of the third parameter for the current bin
        percentiles[i, j] = np.percentile(gwecc_log10_As[mask], 95)
        percentiles_Mch[i, j] = np.percentile(log10_Mchs[mask], 95)
        
        quantiles[i, j] = wquantiles.quantile(gwecc_log10_As[mask], A_post_ws[mask], 0.95)
        quantiles_Mch[i, j] = wquantiles.quantile(log10_Mchs[mask], A_post_ws[mask], 0.95)
        
        mask_valid = (ev_bin_indices == i + 1) & (etav_bin_indices == j + 1)
        valid_percentiles[i, j] = np.percentile(logAv[mask_valid], 95)
        valid_quantiles[i, j] = wquantiles.quantile(logAv[mask_valid], A_prior_ws[mask_valid], 0.95)
        
        plt.subplot(8, 8, 64 - 8*(j+1) + i + 1)
#         plt.hist(10**logAv[mask_valid], bins=10, alpha=0.5, density=True, weights=A_prior_ws[mask_valid])
#         plt.hist(10**gwecc_log10_As[mask], bins=10, alpha=0.5, density=True, weights= A_post_ws[mask])
        plt.hist(logAv[mask_valid], bins=10, alpha=0.5, density=True, weights=None,
                 label=f"{(e_bins[i] + e_bins[i+1])/2:.2f}, {(eta_bins[j] + eta_bins[j+1])/2:.2f} prior")
        plt.hist(gwecc_log10_As[mask], bins=10, alpha=0.5, density=True, weights=None,
                 label="posterior")
        plt.yticks([])
#         plt.legend()
        if i == 0: plt.ylabel(f"{(eta_bins[j] + eta_bins[j+1])/2:.2f}", fontsize=14)
        if j == 7: plt.title(f"{(e_bins[i] + e_bins[i+1])/2:.2f}", fontsize=14)
        if j!=0:
            plt.xticks(ticks=[-11, -9, -7], labels=[])
        else:
            plt.xticks(ticks=[-11, -9, -7], fontsize=14)
        plt.xlim(-12, -6)
        fig.text(0.5, 0.02, r'$log_{10}S_0$', ha='center', fontsize=16)
        fig.text(0.01, 0.5, r'$\eta\,\rightarrow$', va='center', rotation='vertical', fontsize=16)
        fig.suptitle(r'$e_0\,\rightarrow$', fontsize=16)
plt.show()


# print(percentiles, valid_percentiles)


# print("Difference in posteior 95% and prior 95%:\n", 100*(percentiles - valid_percentiles)/valid_percentiles)

# Colormap plot for log_S0
fig = plt.figure(figsize=(10, 8))

# Create a colormap plot
ax = fig.add_subplot(111)
im = ax.imshow(quantiles.T, origin='lower', cmap='winter', aspect='auto', extent=[np.min(es), np.max(es), np.min(etas), np.max(etas)])
ax.set_xlabel(r'$e_0$', fontsize=16)
ax.set_ylabel(r'$\eta$', fontsize=16)
ax.set_title(title_txt, fontsize=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Add text annotations to the colormap plot
for i in range(num_bins):
    for j in range(num_bins):
        value = quantiles[i, j]
        valid_value = valid_quantiles[i, j]
        if (value - valid_value)/valid_value > 0.05:
            txt=ax.text((e_bins[i] + e_bins[i + 1]) / 2, (eta_bins[j] + eta_bins[j + 1]) / 2, 
                    f'{value:.2f}\n({valid_value:.2f})', fontsize=14,
                    color='black', ha='center', va='center')
            txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])
        else:
            txt=ax.text((e_bins[i] + e_bins[i + 1]) / 2, (eta_bins[j] + eta_bins[j + 1]) / 2, 
                    f'{value:.2f}\n({valid_value:.2f})', fontsize=14,
                    color='red', ha='center', va='center')
            txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])


# Create the colorbar
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = plt.colorbar(im, cax=cax)
cbar.set_label(r'95% upper limit on log$_{10}(S_0/ {\rm\, s})$', fontsize=14)

plt.yticks(fontsize=14)

# Adjust subplot spacing
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)

# Save the figure with desired size and aspect ratio
plt.savefig(f'Figures/{name_txt}_term_upperlim_lim_S0.pdf', dpi=300, bbox_inches='tight')

plt.tight_layout()
plt.show()


# Colormap plot for log_Mch
fig = plt.figure(figsize=(10, 8))

# Create a colormap plot
ax = fig.add_subplot(111)
im = ax.imshow(quantiles_Mch.T, origin='lower', cmap='winter', aspect='auto', extent=[np.min(es), np.max(es), np.min(etas), np.max(etas)])
ax.set_xlabel(r'$e_0$', fontsize=16)
ax.set_ylabel(r'$\eta$', fontsize=16)
ax.set_title(title_txt, fontsize=15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Add text annotations to the colormap plot
for i in range(num_bins):
    for j in range(num_bins):
        value = quantiles[i, j]
        valid_value = valid_quantiles[i, j]
        Mch_value = quantiles_Mch[i, j]
        if (value - valid_value)/valid_value > 0.05:
            txt=ax.text((e_bins[i] + e_bins[i + 1]) / 2, (eta_bins[j] + eta_bins[j + 1]) / 2, 
                    f'{Mch_value:.2f}', fontsize=14,
                    color='black', ha='center', va='center')
            txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])
        else:
            txt=ax.text((e_bins[i] + e_bins[i + 1]) / 2, (eta_bins[j] + eta_bins[j + 1]) / 2, 
                    f'{Mch_value:.2f}', fontsize=14,
                    color='red', ha='center', va='center')
            txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])


# Create the colorbar
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = plt.colorbar(im, cax=cax)
cbar.set_label(r'95% upper limit on log$_{10}$ (M$_{\rm ch}$/M$_{\odot}$)', fontsize=14)

plt.yticks(fontsize=14)

# Adjust subplot spacing
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)

# Save the figure with desired size and aspect ratio
plt.savefig(f'Figures/{name_txt}_term_upperlim_lim_Mch.pdf', dpi=300, bbox_inches='tight')

plt.tight_layout()
plt.show()
