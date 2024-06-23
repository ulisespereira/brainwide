import numpy as np
from scipy import sparse 
import pdb
import matplotlib.pyplot as plt
from scipy import sparse 

class SpinesCount:
    '''spines count per area'''
    def __init__(self, modelparams, con_params):
        #spine counts
        self.A_spines = modelparams['A_spines']
        self.offset_spines = modelparams['offset_spines']
        self.spines = con_params['spines']
        self.spines_t1t2 = con_params['hierarchy_t1t2']#con_params['spines']
        self._spines_transformation(self.spines)
    
    def _spines_transformation(self,spines):
        fun=lambda x:self.offset_spines + self.A_spines * x
        return fun(spines)


class GaussianConnectivity:
    ''' Random gaussian connectivity'''
    def __init__(self, modelparams, con_params):
        np.random.seed(modelparams['seed']) #random seed
        self.mat = con_params['fln'] #fln matrix
        self.N = modelparams['N_block'] # # neurons per region
        self.n_ctx = modelparams['n_ctx']  # cortical regions

        #amplitude connections
        self.amp_loc = modelparams['amp_loc_noise']
        self.amp_lr = modelparams['amp_lr_noise']

    def random_projections(self, ind_post, ind_pre):
        '''norally distributed random connectivity'''
        i = ind_post[0]
        j = ind_pre[0]
        K = self.N
        if i == j:
            amp = self.amp_loc 
        else:
            amp = self.amp_lr 
        rand = np.random.normal(0, 1, (self.N, self.N)) 
        rand_con =  (amp/np.sqrt(K)) * rand
        return rand_con
   
class HebbianConnectivity:
    ''' Hebbian connectivity'''
    def __init__(self, modelparams, con_params):
        #np.random.seed()
        self.N = modelparams['N_block'] # # neurons per region
        self.n_ctx = modelparams['n_ctx'] 
        self.sln = con_params['sln'] #fln matrix
        #amplitude connections

        #connectivity of symmetric vs asymmetric
        self.amp_loc = modelparams['amp_loc']
        self.amp_lr = modelparams['amp_lr']
        self.alpha_sym = modelparams['alpha_sym']

        #patterns
        self.subnet_indexes = [[]] #list with indexes
        self.p = len(self.subnet_indexes)
        self.amp_indexes = [[0, 0] for l in range(self.p)]
        self.prob_pat = 0.5

        #symmetri, asymmetric last index
        self.p_sym = 1
        self.p_asym = 3
        self.patterns_current_sym = 2 * np.random.binomial(1, self.prob_pat, size=(self.n_ctx, self.N, self.p, self.p_sym)) - 1
        self.patterns_current_asym = 2 * np.random.binomial(1, self.prob_pat, size=(self.n_ctx, self.N, self.p, self.p_asym)) - 1

    def update_indexes(self, indexes, amp_indexes):
        self.subnet_indexes = indexes
        self.p = len(self.subnet_indexes)
        self.amp_indexes = amp_indexes
        self.patterns_current_sym = 2 * np.random.binomial(1, self.prob_pat, size=(self.n_ctx, self.N, self.p, self.p_sym)) - 1
        self.patterns_current_asym = 2 * np.random.binomial(1, self.prob_pat, size=(self.n_ctx, self.N, self.p, self.p_asym)) - 1


    def hebbian_projections(self, ind_post, ind_pre):
        '''Hebbian connectivity using the covariance rule '''
        i = ind_post[0]
        j = ind_pre[0]

        hebb_sym = np.zeros((self.N, self.N))
        hebb_asym = np.zeros((self.N, self.N))
        hebb_asym_2 = np.zeros((self.N, self.N))
        for l in range(self.p):
            ind = self.subnet_indexes[l]
            if (i in ind) * (j in ind):
                amp_ind_sym = self.amp_indexes[l][0]
                amp_ind_asym = self.amp_indexes[l][1]
            else:
                amp_ind_sym = 0
                amp_ind_asym = 0
                
            #symmetric
            pat_pre_sym = self.patterns_current_sym[j, :, l, 0]
            pat_post_sym = self.patterns_current_sym[i, :, l, 0]

            #asymmetric
            pat_pre_asym = self.patterns_current_asym[j, :, l, 0:self.p_asym-1]
            pat_post_asym = self.patterns_current_asym[i, :, l, 1:self.p_asym]
            pat_pre_asym_2 = self.patterns_current_asym[j, :, l, 0:self.p_asym-1]
            pat_post_asym_2 = self.patterns_current_asym[i, :, l, 0:self.p_asym-1]
                
            if i == j:
                amp_loc = self.amp_loc
                amp_loc_sym = amp_loc * amp_ind_sym/self.N
                amp_loc_asym = amp_loc * amp_ind_asym/self.N

                # recurrent symmetric connectivity
                hebb_sym +=  amp_loc_sym * np.outer(pat_post_sym, pat_pre_sym)
                # recurrent asymmetric connectivity
                hebb_asym +=  amp_loc_asym * np.einsum('ik,jk->ij', pat_post_asym, pat_pre_asym)
                hebb_asym_2 +=  self.alpha_sym * amp_loc_asym * np.einsum('ik,jk->ij', pat_post_asym_2, pat_pre_asym_2)
            else:
                amp_lr = self.amp_lr
                amp_lr_sym = amp_lr * amp_ind_sym/self.N
                amp_lr_asym = amp_lr * amp_ind_asym/self.N

                #symmetric-symmetric projections
                hebb_sym += amp_lr_sym * np.outer(pat_post_sym, pat_pre_sym)
                #asymmetric-symmetric projections
                hebb_asym += amp_lr_asym * np.einsum('ik,jk->ij', pat_post_asym, pat_pre_asym)
                hebb_asym_2 +=  self.alpha_sym * amp_lr_asym * np.einsum('ik,jk->ij', pat_post_asym_2, pat_pre_asym_2)

        return hebb_sym + hebb_asym + hebb_asym_2
    
class CorticalConnectivity:
    def __init__(self, modelparams, con_params):
        self.device = 'cpu'
        np.random.seed(modelparams['seed']) #random seed
        self.mat = con_params['fln'] #fln matrix
        self.sln = con_params['sln'] #fln matrix
        self.hierarchy = con_params['hierarchy'] #hierarchy
        self.N = modelparams['N_block'] # # neurons per region
        self.indexes_areas = []
        for  l in range(self.mat.shape[0]):
            self.indexes_areas.append((l, 'R')) #rest of the areas
        self.n_ctx = modelparams['n_ctx']
   
        self.spines_count = SpinesCount(modelparams, con_params)
        self.spines = self.spines_count._spines_transformation(self.spines_count.spines_t1t2)#self.spines_count.spines_t1t2
        self.random = GaussianConnectivity(modelparams, con_params)
        self.hebbian_symmetric = HebbianConnectivity(modelparams, con_params)

    def _make_synaptic_weights(self, ind_post, ind_pre): 
        ''' synaptic connectivity'''
        i = ind_post[0] #index post-synaptic
        rand =  self.random.random_projections(ind_post, ind_pre) # random
        hebb_sym = self.hebbian_symmetric.hebbian_projections(ind_post, ind_pre)
        con =   self.spines[i] * (hebb_sym  +  rand)
        return con

    def functional_connectivity(self):
        '''cortical fuctional connectivity '''
        blocks = []
        s1=0
        for ind_post in self.indexes_areas:
            bck = []
            i = ind_post[0]
            s2 = 0
            print('area = ', i)
            for ind_pre in self.indexes_areas:
                j = ind_pre[0]
                mat = self.mat[i, j]
                syn_weights = self._make_synaptic_weights(ind_post, ind_pre)
                if i == j:
                    con =  syn_weights
                elif mat == 0:
                    con = sparse.coo_matrix((self.N, self.N))
                    con =  con.tocsr()
                else:
                    #across areas
                    N2bar = np.random.binomial(self.N**2, mat)
                    row_ind = np.random.randint(0, high = self.N, size = N2bar)
                    column_ind = np.random.randint(0, high = self.N, size = N2bar)
                    data =  syn_weights[row_ind, column_ind]
                    con = sparse.coo_matrix((data, (row_ind, column_ind)), shape=(self.N,self.N))
                    con =  con.tocsr()
                bck.append(con)
                s2+=1
            blocks.append(bck)
            s1+=1
        mat = sparse.bmat(blocks)
        return mat
    
    
    
    
    
