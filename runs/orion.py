# Including losses in the Orion 4-pass system. See "Inclusing Losses" document Mathcha.

# Note this is for forward integration through the system only.
#   Inputs:
#       N_0:                initial total inversion per disk, calculated from small sigal gain
#       intensity_spec_in:  spectral intensity profile going into the system
#       lambda_in:          wavelength range of spectral profile
#       tau:                time axis of profile in laser frame coordinates
#       sigma_in:           cross section data ar within wavelength range
import numpy as np
from common.integrators import integrator_fwd, trapper
import common.setup as setup
import params

def orion(alpha0, intensity_spec_in, lambda_in, tau, sigma_in):
    _G_ = []
    _N_ = []
    _E_ = []
    # 1. set initial inversion for each disk.
    #   N_disk holds the value of the inversion seen by fronts of pulses for all 12 disks
    sigma_max = np.max(setup.sigma_)
    N_0 = np.log(alpha0) / sigma_max
    N_disk = np.ones(12)*N_0

    # 2. set loss coefficients for components
    ltrans = params.loss_transmission
    llturn = params.loss_lturn
    lpin = params.loss_pinhole
    intensity_spec = intensity_spec_in

    # 2.5. record initial energy
    energy = trapper(trapper(intensity_spec, lambda_in), tau) * params.area
    _E_.append(energy)

    # 3. losses before even reaching amplifiers
    intensity_spec = intensity_spec*(ltrans**6)*lpin

    # 4. looping through amplifiers
    for i in range(0,4):
        for k in range(0,12):
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

            # 4e. recording energy, gain factor, and inversion array
            energy = trapper(trapper(intensity_spec, lambda_in), tau) * params.area
            _E_.append(energy)
            _G_.append(G)
            _N_.append(N)

        # 4f. so now have intensity_spec after passing through all disks and have N_disk[k] for each disk for the next pass
        # Losses after i'th pass before reaching next amplifier pass
        if i == 0 or i == 2:  # After first and third pass
            intensity_spec = intensity_spec*(ltrans**5)
        if i == 1:  # After second pass
            intensity_spec = intensity_spec*(ltrans**15)*(llturn**2)*(lpin**2)
        if i == 3:  # After fourth and final pass
            intensity_spec = intensity_spec*(ltrans**6)*(lpin)

        N_disk = np.flip(N_disk)    # Reverse order of going through disks each pass

    return intensity_spec, _G_, _N_, _E_
