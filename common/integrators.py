#       Integrating I guess
#  Making an integrator function
# Inputs :
#  - Nin        : inversion density array with Nin[0] front of pulse inversion
#  - Iin        : initial pulse spectral intensity profile
#  - lambda_in  : bandwidth array in lambda
#  - sg         : cross section array
#  - tau        : time axis (not really needed, only dt would suffice)

# Returns :
#  - N : filled inversion density array
import numpy as np
from scipy.constants import pi, hbar, c

def integrator_fwd(Nin, Iin, lambda_in, tau, sg):
    # 1. simply defined timestep
    dt = tau[1] - tau[0]

    # 2. integration loop
    for i in range(0,len(tau)):
        # 2a. all normal elements in tau range
        if i < len(tau)-1:
            # Quantity to be integrated over w
            C = (lambda_in/(hbar*2*pi*c)) * Iin[i] * (np.exp(sg*Nin[i])-1)
            # doing it...
            Nin[i+1] = Nin[i] - dt*trapper(C, lambda_in)

        # 3. this is what any tau after the laser (e.g. a next laser) would see
        elif i == len(tau)-1:
            C = (lambda_in/(hbar*2*pi*c)) * Iin[i] * (np.exp(sg*Nin[i])-1)
            # doing it...
            Nnext = Nin[i] - dt*trapper(C, lambda_in)

    return Nin, Nnext

# following function takes care of monochromatic case of np.trapz() as it would give a 0 for it.
def trapper(function, axis):
    # 1. monochromatic case, just multiply by the wavelength
    if len(axis) == 1:
        return np.squeeze(function[:]*axis[0])
    # 2. broadband case, trapezoidal integral
    else:
        return np.trapz(function, axis)

