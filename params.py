import numpy as np
from scipy.constants import pi, hbar, c

sigmao = 3.9e-24                         # normalisation x-section from data

sigmao = (2*pi*hbar*c)/(4.6e4*1053e-9)   # harrys fluence x-section for 1053 nm to normalise

alpha_amp = 2.1
alpha_disk = np.cbrt(alpha_amp)         # small signal gain per disk


bandwidth = 10e-9                           # bandwidth in [m], (0 if monocromatic)

duration = 10e-9                        # pulse durations in [s]
diameter = 0.16                         # beam diameter in [m]
area = pi*((diameter/2)**2)*1.5         # beam transverse area in [m^2]

loss_transmission = 0.996
loss_lturn = 0.84
loss_pinhole = 0.99
loss_pepc = 0.90

nsteps = 150

offset = 14

e_in = 1e-6
e_out = 2000


