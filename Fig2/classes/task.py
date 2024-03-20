import numpy as np

class working_memory_task(object):
    def __init__(self, modelparams,  connectivity):
        self.N = modelparams['N_block']
        self.n_ctx =  modelparams['n_ctx'] 
        self.p = connectivity.hebbian_symmetric.p
        self.ind_stim = 1
        self.t_start = modelparams['t_start']
        self.period = modelparams['period'] 
        self.t_stim = modelparams['t_stim']
        
        #pattern external input
        self.patterns_sym = connectivity.hebbian_symmetric.patterns_current_sym
        self.patterns_asym = connectivity.hebbian_symmetric.patterns_current_asym
        self.g_p = modelparams['g_p']
        self.i_pat = np.zeros(self.N * self.n_ctx)
        self.ind_asym = 0
        self.pat_input()
        

    #input pattern hebbian 
    def pat_input(self, ind_pat=0):
        ind1 = self.ind_stim * self.N #index start
        ind2 = (self.ind_stim + 1) * self.N#index end
        stim = self.patterns_asym[self.ind_stim, :, ind_pat, self.ind_asym] #pattern
        self.i_pat[ind1:ind2] = self.g_p * stim
    
    def input_current(self,t):
        ''' input current '''
        if  self.t_start <= t and  t <= self.t_start + self.t_stim:
            return self.i_pat
        else:   
            return np.zeros(self.N * self.n_ctx)

