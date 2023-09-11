import numpy as np
import glob
import os

def get_ew_groups(pta, name='gwecc'):
    """Utility function to get parameter groups for CW sampling.
    These groups should be appended to the usual get_parameter_groups()
    output.
    """
    params = pta.param_names
    ndim = len(params)
    # groups = [list(np.arange(0, ndim))]
    groups = []

    snames = np.unique([[qq.signal_name for qq in pp._signals] 
                        for pp in pta._signalcollections])
    
    print(f"snames = {snames}")
    
    if 'red noise' in snames:

        # create parameter groups for the red noise parameters
        rnpsrs = [p.split('_')[0] for p in params if 'red_noise_log10_A' in p]
        for psr in rnpsrs:
            groups.extend([[params.index(psr + '_red_noise_gamma'), params.index(psr + '_red_noise_log10_A')]])    
    
    psrdist_params = [ p for p in params if 'psrdist' in p ]
    lp_params = [ p for p in params if 'lp' in p ]
    gammap_params = [ p for p in params if 'gammap' in p ]
    
    if len(psrdist_params) !=0:
        for pd, lp, gp in zip (psrdist_params, lp_params, gammap_params):
            groups.extend([[params.index(pd), params.index(lp), params.index(gp)]])
    else:
        print("Not searching for psrdist!")
        
    if 'gwb_gamma' in params:
        groups.extend([[params.index('gwb_gamma'), params.index('gwb_log10_A')]])
        
    if f'{name}_e0' in params:
        gpars = [x for x in params if name in x] #global params
        groups.append([params.index(gp) for gp in gpars]) #add global params

        #pair global params
        groups.extend([[params.index(f'{name}_log10_A'), params.index(f'{name}_eta')]])
        groups.extend([[params.index(f'{name}_log10_A'), params.index(f'{name}_e0')]])
        groups.extend([[params.index(f'{name}_eta'), params.index(f'{name}_e0')]])
        groups.extend([[params.index(f'{name}_log10_A'), params.index(f'{name}_eta'), params.index(f'{name}_e0')]])
        groups.extend([[params.index(f'{name}_log10_A'), params.index(f'{name}_cos_inc')]])
        groups.extend([[params.index(f'{name}_gamma0'), params.index(f'{name}_psi')]])
        # groups.extend([[params.index(f'{name}_gamma0'), params.index(f'{name}_l0')]])
        # groups.extend([[params.index(f'{name}_psi'), params.index(f'{name}_l0')]])
        
    return groups



class JumpProposalLD(object):

    def __init__(self, pta, snames=None, empirical_distr=None, f_stat_file=None, save_ext_dists=False, outdir='./chains'):
        """Set up some custom jump proposals"""
        self.params = pta.params
        self.pnames = pta.param_names
        self.psrnames = pta.pulsars
        self.ndim = sum(p.size or 1 for p in pta.params)
        self.plist = [p.name for p in pta.params]

        # parameter map
        self.pmap = {}
        ct = 0
        for p in pta.params:
            size = p.size or 1
            self.pmap[str(p)] = slice(ct, ct+size)
            ct += size

        # parameter indices map
        self.pimap = {}
        for ct, p in enumerate(pta.param_names):
            self.pimap[p] = ct

        # collecting signal parameters across pta
        if snames is None:
            allsigs = np.hstack([[qq.signal_name for qq in pp._signals]
                                 for pp in pta._signalcollections])
            self.snames = dict.fromkeys(np.unique(allsigs))
            for key in self.snames:
                self.snames[key] = []

            for sc in pta._signalcollections:
                for signal in sc._signals:
                    self.snames[signal.signal_name].extend(signal.params)
            for key in self.snames:
                self.snames[key] = list(set(self.snames[key]))
        else:
            self.snames = snames

        # empirical distributions
        if isinstance(empirical_distr, list):
            # check if a list of emp dists is provided
            self.empirical_distr = empirical_distr

        # check if a directory of empirical dist pkl files are provided
        elif empirical_distr is not None and os.path.isdir(empirical_distr):

            dir_files = glob.glob(empirical_distr+'*.pkl')  # search for pkls

            pickled_distr = np.array([])
            for idx, emp_file in enumerate(dir_files):
                try:
                    with open(emp_file, 'rb') as f:
                        pickled_distr = np.append(pickled_distr, pickle.load(f))
                except:
                    try:
                        with open(emp_file, 'rb') as f:
                            pickled_distr = np.append(pickled_distr, pickle.load(f))
                    except:
                        print(f'\nI can\'t open the empirical distribution pickle file at location {idx} in list!')
                        print("Empirical distributions set to 'None'")
                        pickled_distr = None
                        break

            self.empirical_distr = pickled_distr

        # check if single pkl file provided
        elif empirical_distr is not None and os.path.isfile(empirical_distr):  # checking for single file
            try:
                # try opening the file
                with open(empirical_distr, 'rb') as f:
                    pickled_distr = pickle.load(f)
            except:
                # second attempt at opening the file
                try:
                    with open(empirical_distr, 'rb') as f:
                        pickled_distr = pickle.load(f)
                # if the second attempt fails...
                except:
                    print('\nI can\'t open the empirical distribution pickle file!')
                    pickled_distr = None

            self.empirical_distr = pickled_distr

        # all other cases - emp dists set to None
        else:
            self.empirical_distr = None

        if self.empirical_distr is not None:
            # only save the empirical distributions for parameters that are in the model
            mask = []
            for idx, d in enumerate(self.empirical_distr):
                if d.ndim == 1:
                    if d.param_name in pta.param_names:
                        mask.append(idx)
                else:
                    if d.param_names[0] in pta.param_names and d.param_names[1] in pta.param_names:
                        mask.append(idx)
            if len(mask) >= 1:
                self.empirical_distr = [self.empirical_distr[m] for m in mask]
                # extend empirical_distr here:
                print('Extending empirical distributions to priors...\n')
                self.empirical_distr = extend_emp_dists(pta, self.empirical_distr, npoints=100_000,
                                                        save_ext_dists=save_ext_dists, outdir=outdir)
            else:
                self.empirical_distr = None

        if empirical_distr is not None and self.empirical_distr is None:
            # if an emp dist path is provided, but fails the code, this helpful msg is provided
            print("Adding empirical distributions failed!! Empirical distributions set to 'None'\n")

        # F-statistic map
        if f_stat_file is not None and os.path.isfile(f_stat_file):
            npzfile = np.load(f_stat_file)
            self.fe_freqs = npzfile['freqs']
            self.fe = npzfile['fe']

    def draw_from_many_par_prior(self, par_names, string_name):
        # Preparing and comparing par_names with PTA parameters
        par_names = np.atleast_1d(par_names)
        par_list = []
        name_list = []
        for par_name in par_names:
            pn_list = [n for n in self.plist if par_name in n]
            if pn_list:
                par_list.append(pn_list)
                name_list.append(par_name)
        if not par_list:
            raise UserWarning("No parameter prior match found between {} and PTA.object."
                              .format(par_names))
        par_list = np.concatenate(par_list,axis=None)

        def draw(x, iter, beta):
            """Prior draw function generator for custom par_names.
            par_names: list of strings
            The function signature is specific to PTMCMCSampler.
            """

            q = x.copy()
            lqxy = 0

            # randomly choose parameter
            idx_name = np.random.choice(par_list)
            idx = self.plist.index(idx_name)

            # if vector parameter jump in random component
            param = self.params[idx]
            if param.size:
                idx2 = np.random.randint(0, param.size)
                q[self.pmap[str(param)]][idx2] = param.sample()[idx2]

            # scalar parameter
            else:
                q[self.pmap[str(param)]] = param.sample()

            # forward-backward jump probability
            lqxy = (param.get_logpdf(x[self.pmap[str(param)]]) -
                    param.get_logpdf(q[self.pmap[str(param)]]))

            return q, float(lqxy)

        name_string = string_name
        draw.__name__ = 'draw_from_{}_prior'.format(name_string)
        return draw