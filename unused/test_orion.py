
from common.setup import intensity_profile
from common.integrators import trapper
from runs.orion import orion


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from scipy.constants import pi

def test_orion(lambda_in, sigma_in):
    # # Some fundamental parameters
    # sigma0 = 3.9e-24    #   Max cross section (as data comes from relative)
    # bandwidth = 10e-9   #   Desired bandwidth of signal

    # # Read X data
    # lambda_, sigma_ = get_data("saturation/resources/glass_data.csv", sigma0)

    # # Define desired cross section range
    # bandwidth   =   10e-9
    # lambda_in, sigma_in = setup_cross(lambda_, sigma_, bandwidth)

    alpha_disk = np.cbrt(2.1)
    # Make initial I profile
    energy_in = 0.05     #   200 mJ of input energy
    duration = 10e-9    #   10 ns pulse duration
    diameter = 0.175    #   175 mm beam diameter
    area = 1.5*pi*((diameter/2)**2)

    intensity_peak_in   =    energy_in/ (area*duration)

    tau, intensity_spec_in, idx_min, idx_max = intensity_profile(duration, 1.5, lambda_in, np.max(intensity_peak_in))
    
    # run orion loop
    intensity_spec_out, _G_ , _N_ = orion(alpha_disk, intensity_spec_in, lambda_in, tau, sigma_in)
    _G_ = np.array(_G_)
    _N_ = np.array(_N_)
    print(np.shape(_G_))
    print(np.shape(_N_))
    
    # plot initial and final pulses
    intensity_tot_in = trapper(intensity_spec_in[:,:], lambda_in)
    intensity_tot_out = trapper(intensity_spec_out[:,:], lambda_in)
    energy_in = trapper(intensity_tot_in, tau) * area
    print(f"energy in = {energy_in*1e3:.4f}", "mJ")
    energy = trapper(intensity_tot_out, tau) * area
    print(f"energy_out = {energy*1e-3:.4f}", "kJ")

    intensity_tot_in_units = intensity_tot_in*1e-4
    intensity_tot_out_units = intensity_tot_out*1e-4

    # Print energies

    # Normalised plots on same axis

    fig, ax1 = plt.subplots(dpi=100)

    ax1.plot(tau*1e9, intensity_tot_in_units, color="b", linewidth = 3, linestyle = '-')
    ax2 = ax1.twinx()
    ax2.plot(tau*1e9, intensity_tot_out_units, color="r", linewidth = 2)

    ax1.grid()
    ax1.set_xlabel(r"$\tau$ [ns]", fontsize = 11)

    ax1.set_title("Initial and output total intensity profiles")

    ax1.set_ylim(bottom=0, top = np.max(intensity_tot_in_units)*5.2)
    ax2.set_ylim(bottom=0, top = np.max(intensity_tot_out_units)*1.05)

    ax1.set_ylabel(r"$I_{in}$ W/cm$^2$", fontsize = 11)
    ax2.set_ylabel(r"$I_{out}$ W/cm$^2$", fontsize = 11)

    line1 = r"$I_{in}" + fr"=${np.max(intensity_tot_in*1e4):.2e} W/cm$^2$" + "\n"
    line2 = r"$I_{out}^{max}" + fr"=${np.max(intensity_tot_out*1e4):.2e} W/cm$^2$" + "\n"
    line3 = r"$T_{pulse} = " + fr"{duration*1e9}$ ns"
    txt = line1 + line2 + line3
    plt.text(0.65, 0.83, txt, fontsize=9.5, transform=plt.gca().transAxes,  horizontalalignment='left',
            bbox=dict(facecolor='w', alpha=1, edgecolor='black', boxstyle='round,pad=0.2'))

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    ax1.yaxis.set_major_formatter(formatter)
    ax2.yaxis.set_major_formatter(formatter)


    # ax1.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
    fig.tight_layout()
    plt.show()
    print("Peak amplificaton ratio = ", np.max(intensity_tot_out)/np.max(intensity_tot_in))




