# This is a simple test configuration with a single pass through one amplifier to test the models.
# The one amplifier will have 3 disks just as the Vulcan 208's do.

import numpy as np
import params
from common.integrators import integrator_fwd, trapper

def simple(alpha0, intensity_spec_in, lambda_in, tau, sigma_in):
    _G_ = []
    _N_ = []
    _E_ = []
    # 1. set initial inversion for each disk.
    #   N_disk holds the value of the inversion seen by fronts of pulses for all 12 disks
    N_0 = np.log(alpha0)/ params.sigmao
    N_disk = np.ones(3)*N_0

    # 2. set loss coefficients for components
    ltrans = params.loss_transmission
    intensity_spec = intensity_spec_in

    # 2.5. record initial energy
    energy = trapper(trapper(intensity_spec, lambda_in), tau) * params.area
    _E_.append(energy)

    # 3. looping through amplifiers
    for i in range(0,1):
        for k in range(0,len(N_disk)):
            # 4a. the k'th disk has front inversion...
            N_in = np.zeros(len(tau))
            N_in[0] = N_disk[k]
            # 4b. integrating to find N(back) and Nnext = N(back + 1)
            intensity_spec = intensity_spec*ltrans
            N, Nnext = integrator_fwd(N_in, intensity_spec, lambda_in, tau, sigma_in)
            # 4c. front of the next pulse that passes through the disk will see Nnext
            N_disk[k] = Nnext
            
            # 4d. calculating intensity after disk using the gain factor
            G = np.zeros((len(tau), len(lambda_in)))
            for j in range(0,len(tau)):
                G[j] = np.exp(sigma_in*N[j])

            intensity_spec = intensity_spec*G
            intensity_spec = intensity_spec*ltrans
            energy = trapper(trapper(intensity_spec, lambda_in), tau) * params.area
            _E_.append(energy)
            _G_.append(G)
            _N_.append(N)

    return intensity_spec, _G_, _N_, _E_