import numpy as np

class working_memory_task(object):
    def __init__(self, modelparams,  connectivity):
        self.N = modelparams['N_block']
        self.n_ctx =  modelparams['n_ctx'] 
        self.p = connectivity.hebbian_symmetric.p
        self.ind_stim = [1]
        self.t_start = modelparams['t_start']
        self.period = modelparams['period'] 
        self.t_stim = modelparams['t_stim']
        
        #pattern external input
        self.patterns_sym = connectivity.hebbian_symmetric.patterns_current_sym
        self.patterns_asym = connectivity.hebbian_symmetric.patterns_current_asym
        self.g_p = modelparams['g_p']
        self.g_offset = modelparams['g_offset']
        self.i_pat = np.zeros((2, self.N * self.n_ctx))
        self.ind_asym = 0
        self.i_offset = np.zeros(self.N * self.n_ctx)

        
    
    #input pattern symmetric hebbian 
    def pat_input_symmetric(self, ind_pat=0):
        for l in self.ind_stim:
            ind1 = l * self.N #index start
            ind2 = (l + 1) * self.N#index end
            stim1 = self.patterns_sym[l, :, 0, self.ind_asym] #pattern
            stim2 = self.patterns_sym[l, :, 1, self.ind_asym]
            self.i_pat[0,ind1:ind2] = self.g_p * stim1
            self.i_pat[1,ind1:ind2] = self.g_p * stim2

    #fast input
    def pat_fast_input_symmetric(self, ind_pat=0):
        for l in self.ind_stim:
            ind1 = l * self.N #index start
            ind2 = (l + 1) * self.N#index end
            stim2 = self.patterns_sym[l, :, ind_pat, self.ind_asym]
            self.i_pat[1,ind1:ind2] = self.g_p * stim2
        self.i_pat[0,:] = np.zeros(self.N * self.n_ctx)
    
     #input pattern symmetric hebbian 
    def update_i_offset(self,ind_offset, ind_off=0):
        for l in ind_offset:
            ind1 = l * self.N #index start
            ind2 = (l + 1) * self.N#index end
            stim = self.patterns_sym[l, :, ind_off, 0] #pattern
            self.i_offset[ind1:ind2] = self.g_offset * stim
       

    
    def input_current(self,t):
        ''' input current '''
        if  self.t_start <= t and  t <= self.t_start + self.t_stim:
            return self.i_pat[1] + self.i_offset
        else:   
            return self.i_pat[0] + self.i_offset

