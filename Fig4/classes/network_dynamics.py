import numpy as np
import pdb


class ObjectDecoding:
    '''Decoding object'''
    def __init__(self, CorticalConnectivity):
        self.pats_sym = CorticalConnectivity.hebbian_symmetric.patterns_current_sym
        self.pats_asym = CorticalConnectivity.hebbian_symmetric.patterns_current_asym
        self.N = CorticalConnectivity.N
        self.n_ctx = CorticalConnectivity.n_ctx
        self.p = CorticalConnectivity.hebbian_symmetric.p
        self.indexes = CorticalConnectivity.indexes_areas
        self.n_rand = 50
        rand_pats_curr = np.random.binomial(1, .5, (self.n_rand, self.N))
        self.rand_pats = rand_pats_curr
        self.is_sym = True

    def overlaps(self, rates, ind_area, ind_pat, ind_s_a=0):
        '''overlaps one area'''
        r = rates[ind_area * self.N:(ind_area + 1) * self.N]
        g_sym = self.pats_sym[ind_pat, :, :, :]
        g_asym = self.pats_asym[ind_pat, :, :, :]
        #std_r = np.std(r)
        #std_g = np.std(g)
        #print(std_r, np.sqrt(self.intg2))
        ovs_sym = np.einsum('j,jsk->sk', r, g_sym)/(self.N )#* std_g * std_r)
        ovs_asym = np.einsum('j,jsk->sk', r, g_asym)/(self.N )
        #std_g = np.std(g, axis = 0)
        #ov_norm = np.array([ovs[l]/(std_r * std_g[l]) for l in range(g.shape[1])])
        #ovs = np.array([[np.corrcoef(r, g[ind_pat, :, k])[0, 1] for k in range(self.p)] for l in range(self.n_ctx)])
        #ovs = np.array([np.corrcoef(r, g[:, k])[0, 1] for k in range(self.p)])
        return ovs_sym, ovs_asym
    
    def overlaps_random(self, rates, ind_area):
        '''overlaps one area with random patterns'''
        r = rates[ind_area * self.N:(ind_area+1) * self.N]
        g = self.rand_pats
        n_pats = g.shape[0]
        ovs = np.array([np.corrcoef(r, g[k, :])[0, 1] for k in range(n_pats)])
        return ovs
        
    def overlaps_all_areas(self, rates):
        overlaps_sym_list = []
        overlaps_asym_list = []
        ind_area = 0
        for ind_pat, _ in self.indexes:
            ovs_sym, ovs_asym = self.overlaps(rates, ind_area, ind_pat)
            overlaps_sym_list.append(ovs_sym)
            overlaps_asym_list.append(ovs_asym)
            ind_area+=1
        return overlaps_sym_list, overlaps_asym_list
    
    def overlaps_all_areas_random(self, rates):
        overlaps_list = []
        for ind_area in range(self.n_ctx):
            ovs = self.overlaps_random(rates, ind_area)
            overlaps_list.append(ovs)
        return overlaps_list



class QuenchedDisorder:
    '''Decoding object'''
    def __init__(self, CorticalConnectivity):
        self.N = CorticalConnectivity.N
        self.n_ctx = CorticalConnectivity.n_ctx
        self.indexes = CorticalConnectivity.indexes_areas

    def del0(self, rates, ind_area):
        '''overlaps one area'''
        r = rates[ind_area * self.N:(ind_area+1) * self.N]
        del0 = np.mean((r-np.mean(r)) * (r - np.mean(r)))
        return del0
        
    def del0_all_areas(self, rates):
        del0_list = []
        for ind_area in range(self.n_ctx):
            d0 = self.del0(rates, ind_area)
            del0_list.append(d0)
            ind_area+=1
        del0_list = np.array(del0_list)
        return del0_list

class TransferFunctions:
    '''transfer functions inferred from data'''
    def __init__(self, modelparams):
        self.rm = modelparams['rm']
        self.b = modelparams['beta']
        self.h0 = modelparams['h0']	
        self.a = modelparams['amp_offset']
	
    def tf(self,h):
        phi = self.rm/(1.+np.exp(-self.b * (h-self.h0)))
        return phi

    def tf_ht(self,h):
        phi = self.rm * 0.5 * ( 2 * self.a - 1 + np.tanh(self.b * (h -self.h0)))
        return phi

class NetworkDynamics:
    '''This class creates the connectivity matrix'''
    def __init__(self, modelparams, connectivity):
        fun_con = connectivity.functional_connectivity()
        self.subnet_indexes = connectivity.hebbian_symmetric.subnet_indexes
        self.disorder = QuenchedDisorder(connectivity)
        self.overlaps = ObjectDecoding(connectivity)
        self.ovs_random = False
        #connectivity matrices
        self.mat = fun_con #cortex to cortex
        self.N_ctx = fun_con.shape[0] #number neurons cortex
        self.N_recorded = modelparams['N_recorded'] # recorded neuron per area
        self.res = int(connectivity.N/self.N_recorded) #resolution saved neurons per area
        self.indexes = np.arange(0, self.N_ctx, self.res) 
        #simulation params
        self.period = modelparams['period']
        self.dt = modelparams['dt']
        self.sigma_noise = modelparams['sigma_noise']
        
        # transfer function
        self.TF = TransferFunctions(modelparams)

        #dynamics excitatory
        self.tau_ctx = modelparams['tau_ctx'] 
        self.sigma_ld = modelparams['sigma_ld']
        self.tau_ld = modelparams['tau_ld']
        self.mu_ld = modelparams['mu_ld']
        
        #input current
        self.input_ctx = lambda x: np.zeros(self.N_ctx)
        #indeces for saving neurons
        self.indexes_ctx = np.arange(0, self.N_ctx,1)
                # transitions 
        self.index_transition = [0, 1] #indices for transition between subnetworks
        self.transitions = False
        
    def field_noise_low(self,noi):
        gauss = np.random.normal(0,1, 2)
        con = np.sqrt((2 * self.dt * self.sigma_ld**2)/self.tau_ld)
        field = -(self.dt *(noi - self.mu_ld))/self.tau_ld  +  con  * gauss
        return field

    def fields_ld(self, overlaps, noise_struc):
        '''Fields low-dimensional'''
        ovs = np.array(overlaps)
        pats_0 = self.overlaps.pats[:, :, self.index_transition[0]] 
        pats_1 = self.overlaps.pats[:, :, self.index_transition[1]] 
        noise_ff = noise_struc[0] *  np.einsum('ij,i->ij',pats_1, ovs[:, self.index_transition[0]])
        noise_fb = noise_struc[1] *  np.einsum('ij,i->ij',pats_0, ovs[:, self.index_transition[1]])
        return (noise_ff + noise_fb).flatten()

    def fields(self,h_ctx,  t):
        '''Fields cortex'''
        rates = self.TF.tf_ht(h_ctx)
        gauss = np.random.normal(0, 1, h_ctx.shape[0])
        const = np.sqrt((2 * self.sigma_noise**2 * self.tau_ctx)/self.dt)
        cur_ctx = self.mat.dot(rates) + self.input_ctx(t)
        noise = const * gauss
        fld_ctx = - h_ctx + cur_ctx + noise
        return fld_ctx/self.tau_ctx

    def euler(self, h_ctx, t, overlaps, noise_struc):
        #integrating dynamics
        field_ld = 0
        if self.transitions == True:
            field_ld = self.fields_ld(overlaps, noise_struc)
        field_ctx = self.fields(h_ctx, t)
        h_ctx = h_ctx + self.dt * (field_ctx + field_ld)
        return h_ctx
                
    def dynamics(self, h0_ctx):
        '''Dynamics of the large-scale network'''
        time = np.arange(0, self.period, self.dt)
        noise_struc = np.random.normal(0, 1, 2)
        hn_ctx =  h0_ctx
        rates_ctx = []
        del0 = []
        overlaps_sym = []
        overlaps_asym = []
        overlaps_random = []
        for t in time:

            rn_ctx = self.TF.tf_ht(hn_ctx)
            ovs_sym, ovs_asym = self.overlaps.overlaps_all_areas(rn_ctx)

            hn_ctx = self.euler(hn_ctx, t, ovs_sym, noise_struc)
            noise_struc = noise_struc + self.field_noise_low(noise_struc)

            rates_ctx.append(rn_ctx[self.indexes])
            overlaps_sym.append(ovs_sym)
            overlaps_asym.append(ovs_asym)
            if self.ovs_random == True:
                ovs_rand = self.overlaps.overlaps_all_areas_random(rn_ctx)
                overlaps_random.append(ovs_rand)
            d0 = self.disorder.del0_all_areas(hn_ctx)
            del0.append(d0)
            print('Simulation t=',round(t,3))
        rates_ctx = np.array(rates_ctx)
        del0 = np.array(del0)
        overlaps_sym = np.array(overlaps_sym)
        overlaps_asym = np.array(overlaps_asym)
        overlaps_random = np.array(overlaps_random)
        results = dict(
                time = time,
                rates_ctx = rates_ctx,
                del0 = del0,
                overlaps_sym = overlaps_sym,
                overlaps_asym = overlaps_asym,
                overlaps_random = overlaps_random
                )
        return results

