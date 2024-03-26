import numpy as np
import torch as tch


class SMFTSolver(tch.nn.Module):
    def __init__(self, params):
        super(SMFTSolver, self).__init__()
        self.device = tch.device(params['device'])
        self.spines = params['spines']
        self.fln = params['fln']
        self.con_hebb = tch.tensor(params['con_hebb'], dtype=tch.float32, device=self.device)
        self.con_rand = tch.tensor(params['con_rand'], dtype=tch.float32, device=self.device)
        self.n_iterations = params['n_iterations']

        self.ovs_init = params['ovs_init']
        self.dels_init = 0.01 * tch.ones(self.ovs_init.shape, device=self.device)

        #structural connectivity overlap
        structural_conn = self.fln + np.eye(self.fln.shape[0])
        self.struct_overlap = np.einsum('i,ij->ij', self.spines, structural_conn)
        self.struct_overlap = tch.tensor(self.struct_overlap, dtype=tch.float32, device=self.device)

        #structural connectivity delta
        self.struct_delta = np.einsum('i,ij->ij', self.spines**2, structural_conn)
        self.struct_delta = tch.tensor(self.struct_delta, dtype=tch.float32, device=self.device)

        # transfer function:
        self.phi = lambda x: 0.5 * (1 + tch.tanh(params['beta']* (x - params['h0'])))#tch.relu
        self.phi_prime = lambda x: 0.5 * params['beta']**4 * (1 - tch.tanh(params['beta']* (x - params['h0']))**2)
        self.xmin = params['xmin']
        self.xmax = params['xmax']
        self.dx = params['dx']
        self.x = tch.arange(self.xmin, self.xmax, self.dx).to(self.device)
        gauss = tch.exp(-self.x**2/2)/np.sqrt(2*np.pi)
        self.gauss =tch.tensor(gauss, dtype=tch.float32, device=self.device)

        self.dy = 0.001
        self.y = tch.arange(-5, 5, self.dy).to(self.device)
        gauss_y = tch.exp(-self.y**2/2)/np.sqrt(2*np.pi)
        self.gauss_y =tch.tensor(gauss_y, dtype=tch.float32, device=self.device)
            
        self.to(self.device)

    def update(self, overlaps, deltas):
        con_hebb = self.con_hebb * self.struct_overlap
        ones = tch.ones_like(self.x).to(self.device)
        mu = tch.einsum('i,j->ji', overlaps, ones) #@ con_hebb.T#tch.einsum('i,j->ij', overlaps @ con_hebb.T, ones)
        std = tch.einsum('i,j->ij', tch.sqrt(deltas), self.x)
        phi_p = .5 * self.phi((mu @ con_hebb.T).T + std) 
        phi_n = .5 * self.phi((-mu @ con_hebb.T).T + std)
        int_p = tch.trapz(phi_p * self.gauss, dx = self.dx)
        int_n = tch.trapz(phi_n * self.gauss, dx = self.dx)
        overlapsp1 = int_p - int_n
        rates = int_p + int_n

        con_rand = (self.con_rand**2) * self.struct_delta
        phi2_p = self.phi((mu @ con_hebb.T).T  + std)**2
        phi2_n = self.phi((-mu @ con_hebb.T).T + std)**2
        int_p = tch.trapz(phi2_p * self.gauss, dx = self.dx)
        int_n = tch.trapz(phi2_n * self.gauss, dx = self.dx)
        int = (int_p + int_n)/2.
        deltasp1 = (int.T @ con_rand.T).T
        
        #update
        update_delta = deltas + .05 * (deltasp1-deltas)
        update_overlaps = overlaps + .05 *(overlapsp1-overlaps)
        overlapsp1 =  update_overlaps
        deltasp1 = update_delta
   
        return overlapsp1, deltasp1, rates
    
    def jacobian(self, overlaps, deltas):
        con_hebb = self.con_hebb * self.struct_overlap
        ones = tch.ones_like(self.y).to(self.device)
        mu = tch.einsum('i,j->ji', overlaps, ones) #@ con_hebb.T#tch.einsum('i,j->ij', overlaps @ con_hebb.T, ones)
        std = tch.einsum('i,j->ij', tch.sqrt(deltas), self.y)
        phi_prime_p = .5 * self.phi_prime((mu @ con_hebb.T).T + std)
        phi_prime_n = .5 * self.phi_prime((-mu @ con_hebb.T).T + std)

        # diagonal term
        int_prime2_p =tch.trapz(phi_prime_p**2 * self.gauss_y, dx = self.dy)
        int_prime2_n =tch.trapz(phi_prime_n**2 * self.gauss_y, dx = self.dy)
        int_prime = (int_prime2_p + int_prime2_n)/2.
        

        #identity matrix
        #eye = tch.eye(self.fln.shape[0], device=self.device)
        con_rand = (self.con_rand**2) * self.struct_delta
        d2vddelta2m1 = tch.einsum('j, ij->ij',int_prime, con_rand)
        return d2vddelta2m1
    
    def forward(self, ovs):
        ''' this function computes the dynamics of the MFT'''
        overlaps = tch.zeros((self.n_iterations, ovs.shape[0]), device=self.device)
        rates = tch.zeros((self.n_iterations, ovs.shape[0]), device=self.device)
        deltas = tch.zeros((self.n_iterations, ovs.shape[0]), device=self.device)
        jac = tch.zeros( self.fln.shape, device=self.device)
        overs = self.ovs_init
        dels = self.dels_init
        for i in range(self.n_iterations):
            overs, dels, fr = self.update(overs, dels)
            #print(overs)
            overlaps[i,:] = overs
            deltas[i,:] = dels
            rates[i,:] = fr
        jac = self.jacobian(overs, dels)
        return overlaps, deltas,rates, jac