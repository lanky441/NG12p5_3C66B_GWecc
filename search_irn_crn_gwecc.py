"""
Run this script to perform a targeted search for GWs from an eccentric SMBHB in PTA data set.
All the inputs and settings for this search is read from a json file, e.g., 
'irn_crn_gwecc_psrterm.json' and 'irn_crn_gwecc_earth.json' for E+P and E-only searches, respectively.

The par and tim files used for this paper are from NANOGrav 12.5 year narrowabnd data set and can be found at
https://nanograv.org/science/data/125-year-pulsar-timing-array-data-release

Run this code with a single temperature chain using:
python search_irn_crn_gwecc.py -s irn_crn_gwecc_{earth/psrterm}.json

To run using parallel tampering with 4 temparature chains, use:
mpirun -np 4 python search_irn_crn_gwecc.py -s irn_crn_gwecc_{earth/psrterm}.json
"""

print("Starting search script")
import numpy as np
import json
import glob
import argparse
import os
import shutil

from enterprise import constants as const
from enterprise.pulsar import Pulsar
import enterprise.signals.parameter as parameter
from enterprise.signals import utils
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
from enterprise.signals import deterministic_signals
from enterprise.signals import signal_base
from enterprise.signals import selections

from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc

from enterprise_extensions.sampler import JumpProposal as JP
import get_groups_jumps #import get_ew_groups, draw_from_many_par_prior

from enterprise_gwecc import gwecc_target_block, PsrDistPrior
from juliacall import Main as jl
import juliacall


parser = argparse.ArgumentParser()
parser.add_argument("-s", "--setting", default="irn_crn_gwecc_psrterm.json")

args = parser.parse_args()
setting_file = args.setting
setting = json.load(open(f"{setting_file}", "r"))

datadir = setting["datadir"]
target_params = json.load(open(setting["target_params"], "r"))
psrdist_info = json.load(open(setting["psrdist_info"], "r"))
empirical_distr = setting["empirical_distr"]
nfile = setting["noise_dict"]

psrlist_exclude = setting["psr_exclude"]
psrlist_include = setting["psr_include"]

gamma_vary = setting["gamma_vary"]
name = setting["name"]
emax = setting["emax"]
etamin = setting["etamin"]
psrterm = setting["psrterm"]
tie = setting["tie_psrterm"]

x0_median = setting["x0_median"]
Niter = setting["Niter"]
if os.path.isdir("all_chains"):
    chaindir = "all_chains/" + setting["chaindir"]
else:
    chaindir = setting["chaindir"]
hotchains = setting["write_hotchain"]
resume = setting["resume"]
make_groups = setting["make_groups"]
add_jumps = setting["add_jumps"]


if tie or not psrterm:
    priors = {
        "tref": target_params["tref"],
        "cos_gwtheta": target_params["cos_gwtheta"],
        "gwphi": target_params["gwphi"],
        "gwdist": target_params["gwdist"],
        "psi": parameter.Uniform(0.0, np.pi)(f"{name}_psi"),
        "cos_inc": parameter.Uniform(-1, 1)(f"{name}_cos_inc"),
        "eta": parameter.Uniform(etamin, 0.25)(f"{name}_eta"),
        "log10_F": target_params["log10_F"],
        "e0": parameter.Uniform(0.001, emax)(f"{name}_e0"),
        "gamma0": parameter.Uniform(0.0, np.pi)(f"{name}_gamma0"),
        "gammap": 0.0,
        "l0": parameter.Uniform(0.0, 2 * np.pi)(f"{name}_l0"),
        "lp": 0.0,
        "log10_A": parameter.Uniform(-12, -6)(f"{name}_log10_A"),
        "psrdist": PsrDistPrior(psrdist_info),
    }
else:
    priors = {
        "tref": target_params["tref"],
        "cos_gwtheta": target_params["cos_gwtheta"],
        "gwphi": target_params["gwphi"],
        "gwdist": target_params["gwdist"],
        "psi": parameter.Uniform(0.0, np.pi)(f"{name}_psi"),
        "cos_inc": parameter.Uniform(-1, 1)(f"{name}_cos_inc"),
        "eta": parameter.Uniform(etamin, 0.25)(f"{name}_eta"),
        "log10_F": target_params["log10_F"],
        "e0": parameter.Uniform(0.001, emax)(f"{name}_e0"),
        "gamma0": parameter.Uniform(0.0, np.pi)(f"{name}_gamma0"),
        "gammap": parameter.Uniform(0.0, np.pi),
        "l0": parameter.Uniform(0.0, 2 * np.pi)(f"{name}_l0"),
        "lp": parameter.Uniform(0.0, 2 * np.pi),
        "log10_A": parameter.Uniform(-12, -6)(f"{name}_log10_A"),
        "psrdist": PsrDistPrior(psrdist_info),
    }

jobid = None 
try:
    jobid = os.environ["SLURM_JOBID"]
except:
    print("Not a slurm run!")
             
if jobid is not None:
    print(f"jobid = {jobid}")
    
print(f"Chain directory = {chaindir}")

parfiles = sorted(glob.glob(f"{datadir}par/*gls.par"))
timfiles = sorted(glob.glob(f"{datadir}tim/*.tim"))

if psrlist_exclude is not None:
    is_excluded = (
        lambda x: x.split("/")[-1].split(".")[0].split("_")[0] not in psrlist_exclude
    )
    parfiles = [x for x in parfiles if is_excluded(x)]
    timfiles = [x for x in timfiles if is_excluded(x)]

# whitelist supersedes blacklist.
if psrlist_include != "all":
    is_included = (
        lambda x: x.split("/")[-1].split(".")[0].split("_")[0] in psrlist_include
    )
    parfiles = [x for x in parfiles if is_included(x)]
    timfiles = [x for x in timfiles if is_included(x)]

# print(parfiles, timfiles)

ephemeris = setting["ephem"]

psrs = [Pulsar(par, tim, ephem=ephemeris) for par, tim in zip(parfiles, timfiles)]
psrlist = [psr.name for psr in psrs]
[print(pname) for pname in psrlist]


with open(nfile, "r") as f:
    noisedict = json.load(f)


# find the maximum time span to set GW frequency sampling
tmin = np.min([p.toas.min() for p in psrs])
tmax = np.max([p.toas.max() for p in psrs])
Tspan = tmax - tmin
print("tmax = MJD ", np.max(tmax) / 86400)
print("Tspan = ", Tspan / const.yr, "years")


# define selection by observing backend
selection = selections.Selection(selections.by_backend)


# white noise parameters
efac = parameter.Constant()
equad = parameter.Constant()
ecorr = parameter.Constant()  # we'll set these later with the params dictionary

# red noise parameters
log10_A = parameter.Uniform(-20, -11)
gamma = parameter.Uniform(0, 7)

# GW parameters (initialize with names here to use parameters in common across pulsars)
log10_A_gw = parameter.Uniform(-20, -11)("gwb_log10_A")
gamma_gw = (
    parameter.Uniform(0, 7)("gwb_gamma")
    if gamma_vary
    else parameter.Constant(13 / 3)("gwb_gamma")
)


# white noise
ef = white_signals.MeasurementNoise(efac=efac, selection=selection)
eq = white_signals.TNEquadNoise(log10_tnequad=equad, selection=selection)
ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=selection)

# red noise (powerlaw with 30 frequencies)
pl = utils.powerlaw(log10_A=log10_A, gamma=gamma)
rn = gp_signals.FourierBasisGP(spectrum=pl, components=30, Tspan=Tspan)

# gwb (no spatial correlations)
cpl = utils.powerlaw(log10_A=log10_A_gw, gamma=gamma_gw)
gw = gp_signals.FourierBasisGP(spectrum=cpl, components=5, Tspan=Tspan, name="gwb")

# for spatial correlations you can do...
# spatial correlations are covered in the hypermodel context later
# orf = utils.hd_orf()
# crn = gp_signals.FourierBasisCommonGP(cpl, orf,
#                                       components=30, Tspan=Tspan, name='gw')

# to add solar system ephemeris modeling...
bayesephem = setting["bayesephem"]
if bayesephem:
    eph = deterministic_signals.PhysicalEphemerisSignal(use_epoch_toas=True)

# timing model
tm = gp_signals.TimingModel(use_svd=True)

# eccentric signal
wf = gwecc_target_block(
    **priors, spline=True, psrTerm=psrterm, tie_psrTerm=tie, name=""
)

s = (
    ef + eq + ec + rn + tm + eph + gw + wf
    if bayesephem
    else ef + eq + ec + rn + tm + gw + wf
)
models = [s(p) for p in psrs]
pta = signal_base.PTA(models)


# set white noise parameters with dictionary
pta.set_default_params(noisedict)


print(pta.params)
# print(pta.summary())

write_invalid_params = False

# custom function to get lnprior
def gwecc_target_prior_my(pta, gwdist, tref, tmax, log10_F, name="gwecc"):
    def gwecc_target_prior_fn(params):
        param_map = pta.map_params(params)
        
        log10_A = param_map[f"{name}_log10_A"]
        eta = param_map[f"{name}_eta"]
        e0 = param_map[f"{name}_e0"]
        
        pta_prior = pta.get_lnprior(param_map)
        
        if pta_prior == -np.inf:
            return pta_prior
        
        valid, msg = jl.validate_params_target(
            log10_A,
            eta,
            log10_F,
            e0,
            gwdist,
            tref,
            tmax,
        )
        
        if valid:
            return pta_prior
        else:
            # print("Invalid param space.")
            if write_invalid_params:
                with open(f"{chaindir}/invalid_params.txt", "a") as nvp:
                    nvp.write(f"{log10_A}    {eta}   {e0}    {msg}" + "\n")
            return -np.inf

    return gwecc_target_prior_fn


get_lnprior = gwecc_target_prior_my(
    pta,
    target_params["gwdist"],
    target_params["tref"],
    tmax,
    log10_F=target_params["log10_F"],
    name=name,
)

# custom function to get lnlikelihood
def gwecc_target_likelihood_my(pta):
    def gwecc_target_likelihood_fn(params):
        param_map = pta.map_params(params)
        try:
            lnlike = pta.get_lnlikelihood(param_map)
        except juliacall.JuliaError as err_julia:
            print(err_julia.args[0])
            lnlike = -np.inf
        return lnlike

    return gwecc_target_likelihood_fn


get_lnlikelihood = gwecc_target_likelihood_my(pta)

with open(f"{datadir}/noise_param_median_5f.json", "r") as npmf:
    median_params = json.load(npmf)

# set initial parameters from dict or draw from prior
x0 = []
if x0_median:
    for p in pta.param_names:
        if "gwecc" in p:
            x0.append(target_params[p])
        elif "psrdist" in p:
            x0.append(psrdist_info[p.split("_")[0]][0])
        elif "gammap" in p:
            x0.append(np.pi/2)
        elif "lp" in p:
            x0.append(np.pi)
        else:
            x0.append(median_params[p])
    x0 = np.hstack(x0)
else:
    lnprior_x0 = -np.inf
    while lnprior_x0 == -np.inf:
        x0 = np.hstack([p.sample() for p in pta.params])
        lnprior_x0 = get_lnprior(x0)

print(f"x0 = {x0}")
print(f"lnprior(x0) = {get_lnprior(x0)}")
print(f"lnlikelihood(x0) = {get_lnlikelihood(x0)}")

groups = get_groups_jumps.get_ew_groups(pta, name=name) if make_groups else None
print(f"groups = {groups}")

ndim = len(x0)

# set up the sampler:
# initial jump covariance matrix
cov = np.diag(np.ones(ndim) * 0.01**2)

sampler = ptmcmc(
    ndim,
    get_lnlikelihood,
    get_lnprior,
    cov,
    groups=groups,
    outDir=chaindir,
    resume=resume,
)

# write parameter names
np.savetxt(f"{chaindir}/params.txt", list(map(str, pta.param_names)), fmt="%s")

# write list of pulsars
np.savetxt(f"{chaindir}/psrlist.txt", np.array(psrlist), fmt="%s")

# save setting.json file
with open(f"{chaindir}/setting.json", "w") as f:
    json.dump(setting, f, indent = 4)

# copying the python files to the chain directory for reference
shutil.copy("search_irn_crn_gwecc.py", f"{chaindir}/search_irn_crn_gwecc.py")
shutil.copy("get_groups_jumps.py", f"{chaindir}/get_groups_jumps.py")

# save groups file
with open(f"{chaindir}/groups.txt", "w") as f:
    f.write(str(groups))

#save jobID
if jobid is not None:
    with open(f"{chaindir}/jobid.txt", "w") as f:
        f.write(jobid)

        
if add_jumps:
    jp = JP(pta, empirical_distr=empirical_distr)
    jpLD = get_groups_jumps.JumpProposalLD(pta, empirical_distr=None)

    # if 'red noise' in jp.snames:
    #     sampler.addProposalToCycle(jp.draw_from_red_prior, 20)
    if empirical_distr:
        sampler.addProposalToCycle(jp.draw_from_empirical_distr, 30)

    sampler.addProposalToCycle(jp.draw_from_prior, 10)

    # draw from ewf priors
    ew_params = [x for x in pta.param_names if name in x]
    for ew in ew_params:
        if "log10_A" in ew:
            sampler.addProposalToCycle(jp.draw_from_par_prior(ew), 10)
        else:
            sampler.addProposalToCycle(jp.draw_from_par_prior(ew), 5)

    # draw from gwb priors
    gwb_params = [x for x in pta.param_names if "gwb" in x]
    for param in gwb_params:
        sampler.addProposalToCycle(jp.draw_from_par_prior(param), 2)
        
    # draw from psrdist, lp, gammap priors
    # psr_params = [x for x in pta.param_names if any(y in x for y in ['psrdist', 'gammap', 'lp'])]
    lp_params = [x for x in pta.param_names if 'lp' in x]
    gammap_params = [x for x in pta.param_names if 'gammap' in x]
    psrdist_params = [x for x in pta.param_names if 'psrdist' in x]
    
    print(f"lp params = {lp_params}, gammap_params = {gammap_params}, psrdist params = {psrdist_params}")
    
    if len(lp_params) !=0:
        sampler.addProposalToCycle(jpLD.draw_from_many_par_prior(lp_params, 'lp'), 10)
    
    if len(gammap_params) !=0:
        sampler.addProposalToCycle(jpLD.draw_from_many_par_prior(gammap_params, 'gammap'), 10)
    
    if len(psrdist_params) !=0:
        sampler.addProposalToCycle(jpLD.draw_from_many_par_prior(psrdist_params, 'psrdist'), 10)

write_invalid_params = True
        
sampler.sample(
    x0, Niter, SCAMweight=20, AMweight=25, DEweight=15, writeHotChains=hotchains
)

print("Sampler run completed successfully.")


if jobid is not None and os.path.isfile(f"slurm-{jobid}.out"):
    shutil.move(f"slurm-{jobid}.out", f"{chaindir}/slurm-{jobid}.out")