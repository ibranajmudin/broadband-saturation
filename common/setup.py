import numpy as np
import pandas as pd
import params
from scipy.constants import c, hbar, pi

def setup_cross(path = "saturation/resources/glass_data.csv", bandwidth = params.bandwidth, offset = params.offset):
    # 1. read xsection data
    # path  -   path to cross section data in /resources\
    data = pd.read_csv(path, skiprows=[0,1,2,3])
    data = data.dropna(axis=1) # Removes an unwanted NA row
    lambda_ = np.array(data["l [nm]"])*1e-9
    sigma_ = np.array(data["Signal, FC(l)"])

    # alternate normalisation given x-section at 1053 nm not 1052.75 nm.
    factor = sigma_[sigma_.argmax()] / sigma_[sigma_.argmax() + 2]
    sigma_ = np.array(data["Signal, FC(l)"])*(params.sigmao*factor)

    # 2. determine pulse's central wavelength (including potential offset)
    idx_cent = sigma_.argmax() + offset
    l_cent = lambda_[idx_cent]

    # 3. determine if broadband or monochromatic
    # 3a. desired indexes within 'wavelength range'
    l_min = l_cent - bandwidth/2
    l_max = l_cent + bandwidth/2
    idx_min = np.abs(lambda_-l_min).argmin()
    idx_max = np.abs(lambda_-l_max).argmin()

    # 3b. monochromatic case
    if idx_min == idx_max:
        print("\n**    RUNNING MONOCHROMATIC BEAM    **")
        print(f"**    CENTRAL WAVELENGTH {l_cent*1e9:.3f} nm    **")
        lambda_in = np.array([l_cent])
        sigma_out = np.array([sigma_[idx_cent]])
        print(f"**    RELATIVE X SECTION {sigma_out[0]/params.sigmao:.4f}    **")
        fluence = (2*pi*hbar*c) / (lambda_in[0]*sigma_out[0])
        print(f"**    SATURATION FLUENCE {fluence*1e-4:.3f} J/cm^2    **\n")

    # 3c. broadband case
    else:
        print("\n**    RUNNING BROADBAND BEAM    **")
        print(f"**    CENTRAL WAVELENGTH {l_cent*1e9:.3f} nm    **")
        print(f"**    BANDWIDTH {(lambda_[idx_max]-lambda_[idx_min])*1e9:.3f} nm    **\n")
        lambda_in = lambda_[idx_min:idx_max+1]
        sigma_out = sigma_[idx_min:idx_max+1]


    return lambda_in, sigma_out

def intensity_profile(duration, win_size, lambda_in, I_init, n = params.nsteps):
    # 1. initialise tau axis
    tau = np.linspace(0, win_size, int(n*win_size+1))*duration

    # 2. initialise intensity profile in time and spectra
    #   for I0, first index is tau, second index is w
    I0 = np.zeros((np.size(tau), np.size(lambda_in)))

    # 3. treating first monochromatic and then broadband case separately
    if np.size(lambda_in) == 1:
        delta_lambda = lambda_in[0]
    else:
        delta_lambda = np.max(lambda_in) - np.min(lambda_in)

    # 4. positioning pulse to be centered in the tau axis window
    #   centering for both odd and even lengths of tau array (because i'm pedantic)
    if np.size(tau) % 2 == 0:
        t_idx_min = int(np.size(tau)/2 - n/2)
        t_idx_max = int(np.size(tau)/2 + n/2) - 1

    elif np.size(tau) % 2 != 0:
        t_idx_min = int(np.size(tau)/2 - n/2)
        t_idx_max = int(np.size(tau)/2 + n/2)


    I0[t_idx_min:t_idx_max+1] = I_init / delta_lambda
    # print(delta_omega)

    return tau, I0, t_idx_min, t_idx_max


# To allow for access throughout 
data = pd.read_csv("saturation/resources/glass_data.csv", skiprows=[0,1,2,3])
data = data.dropna(axis=1) # Removes an unwanted NA row
lambda_ = np.array(data["l [nm]"])*1e-9
sigma_ = np.array(data["Signal, FC(l)"])

# alternate normalisation given x-section at 1053 nm not 1052.75 nm.
factor = sigma_[sigma_.argmax()] / sigma_[sigma_.argmax() + 2]
sigma_ = np.array(data["Signal, FC(l)"])*(params.sigmao*factor)

