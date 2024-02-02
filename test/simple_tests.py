# Testing the simple configuration

from common.setup import intensity_profile
from common.integrators import trapper
from runs.simple import simple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import ScalarFormatter
from scipy.constants import pi, hbar, c
from common.setup import setup_cross

import params

# --------------------------------------------------------------------------------------------------------------------------------------
def simple_v1():
    print("\n**    RUNNING SIMPLE CONFIG    **")
    lambda_in, sigma_in = setup_cross()

    # ------------------------------------------------------------------------------------------------
    # Initialising scanning arrays and parameters
    fluence_in = np.linspace(0.1, 10, 51)*1e4
    fluence_out_analytic = np.zeros(len(fluence_in))
    fluence_out_numeric = np.zeros(len(fluence_in))

    idx_cent = int((len(lambda_in)-1)/2)        # central wavelength idx generalisation for broadband case.

    fluence_sat = (2*pi*hbar*c)/(sigma_in[idx_cent]*lambda_in[idx_cent])
    print(fluence_sat)
    G0 = params.alpha_amp**(sigma_in[idx_cent]/params.sigmao)
    print(f"G0 = {G0:.4f}")

    ltrans = params.loss_transmission

    # ------------------------------------------------------------------------------------------------
    # Scanning analytical
    for el in range(0, len(fluence_in)):
        fluence = fluence_in[el]
        for p in range(0,3):
            fluence = fluence*ltrans
            a = np.exp(fluence/fluence_sat) - 1
            b = (G0**(1/3))*(a) + 1
            fluence = fluence_sat*np.log(b)
            fluence = fluence*ltrans
        fluence_out_analytic[el] = fluence

    # ------------------------------------------------------------------------------------------------
    # Scanning numerical
    for el in range(0, len(fluence_in)):
        tau, intensity_spec_in, imin, imax = intensity_profile(params.duration, 1.5, lambda_in, fluence_in[el]/params.duration)
        intensity_spec_out, _G_, _N_, _E_ = simple(params.alpha_disk, intensity_spec_in, lambda_in, tau, sigma_in)
        intensity_tot_out = trapper(intensity_spec_out, lambda_in)

        fluence_out_numeric[el] = trapper(intensity_tot_out, tau)
    # ------------------------------------------------------------------------------------------------
    # Plotting
    fig, ax = plt.subplots(1,1, figsize=(5.5, 3.5), dpi=200)
    figd, axd = plt.subplots(1,1, figsize=(5.5, 3.5), dpi=200)
    ax.plot(fluence_in*1e-4, fluence_out_analytic/fluence_in, linestyle="None", marker = "D", markersize=3, color="black")
    ax.plot(fluence_in*1e-4, fluence_out_numeric/fluence_in, linestyle="None", marker = "x", markersize=3, color="red")

    ax.set_title("Analytic and numerical calculation for single amplifier", fontsize=11)
    ax.legend(["Analytical","Numerical"], fontsize=10)
    ax.set_xlabel(r"$\Gamma_{in}$ [J/cm$^2$]", fontsize=11)
    ax.set_ylabel(r"$K = \Gamma_{out}$ / $\Gamma_{in}$", fontsize=11)
    print(len(lambda_in))
    txt1 = fr"$\lambda = $ {lambda_in[idx_cent]*1e9:.2f} nm" + "\n"
    txt2 = fr"$\Delta\lambda = $ {(lambda_in[-1] - lambda_in[0])*1e9:.3f} nm" + "\n"
    txt3 = f"Transmission = {ltrans*100:.1f} %" + "\n"
    txt4 = fr"$\alpha_0 = {params.alpha_amp}$" + "\n"
    txt5 = fr"$\Gamma_s = {fluence_sat*1e-4:.2f}$ [J/cm$^2$]"
    txt = txt1 + txt2 + txt3 + txt4 +txt5
    ax.text(0.6, 0.4, txt, transform=ax.transAxes, fontsize=8.5,
        bbox=dict(facecolor='w', alpha=0.7, edgecolor='black', boxstyle='round,pad=0.2'))

    axd.plot(fluence_in*1e-4, 100*(fluence_out_numeric - fluence_out_analytic)/(fluence_out_analytic),
        linestyle="None", marker = "o", markersize=3, color="slateblue")

    axd.set_title("Numerical error", fontsize=11)
    axd.set_xlabel(r"$\Gamma_{in}$ [J/cm$^2$]", fontsize=11)
    axd.set_ylabel(r"$(K_{\text{numeric}} - K_{\text{analytic}}) / K_{\text{analytic}}$ [%]")
    txt = fr"nsteps = {int(params.nsteps)}"
    axd.text(0.7, 0.3, txt, transform=ax.transAxes, fontsize=9,
        bbox=dict(facecolor='w', alpha=0.7, edgecolor='black', boxstyle='round,pad=0.2'))


    fig.tight_layout()
    figd.tight_layout()
    plt.show()

# --------------------------------------------------------------------------------------------------------------------------------------
def simple_v2():
    print("\n**    RUNNING SIMPLE CONFIG    **")

    # ------------------------------------------------------------------------------------------------
    # Monochromatic run to get reference
    lambda_in, sigma_in = setup_cross(bandwidth = 0)

    fluence_in = np.linspace(0.1, 10, 51)*1e4
    fluence_out_analytic = np.zeros(len(fluence_in))
    fluence_out_numeric = np.zeros(len(fluence_in))

    idx_cent = int((len(lambda_in)-1)/2)        # central wavelength idx generalisation for broadband case.

    fluence_sat = (2*pi*hbar*c)/(sigma_in[idx_cent]*lambda_in[idx_cent])
    G0 = params.alpha_amp**(sigma_in[idx_cent]/params.sigmao)

    ltrans = params.loss_transmission

    # ------------------------------------------------------------------------------------------------
    # Scanning analytical
    for el in range(0, len(fluence_in)):
        fluence = fluence_in[el]
        for p in range(0,3):
            fluence = fluence*ltrans
            a = np.exp(fluence/fluence_sat) - 1
            b = (G0**(1/3))*(a) + 1
            fluence = fluence_sat*np.log(b)
            fluence = fluence*ltrans
        fluence_out_analytic[el] = fluence

    # ------------------------------------------------------------------------------------------------
    # Doing bandwidth vs average error scan for changing bandwidth
    bands = np.linspace(0, 10, 41)*1e-9

    K = []
    # For each bandwidth run a numerical calculation, then find average error, then store average error.
    for dl in bands:
        print(f"{dl*1e9:.2f}")
        lambda_in, sigma_in = setup_cross(bandwidth = dl)
        for el in range(0, len(fluence_in)):
            # run numerical calculation 
            tau, intensity_spec_in, imin, imax = intensity_profile(params.duration, 1.5, lambda_in, fluence_in[el]/params.duration, n = 400)
            intensity_spec_out, _G_, _N_, _E_ = simple(params.alpha_disk, intensity_spec_in, lambda_in, tau, sigma_in)
            intensity_tot_out = trapper(intensity_spec_out, lambda_in)

            fluence_out_numeric[el] = trapper(intensity_tot_out, tau)

        # find average percentage error
        pe = np.average(np.abs(100*(fluence_out_numeric - fluence_out_analytic)/fluence_out_analytic))
        K.append(pe)

    K = np.array(K)
    idx_cent = int((len(lambda_in)-1)/2)
    fig, ax = plt.subplots(1,1, figsize = (5.5, 3.5), dpi = 200)
    ax.plot(bands*1e9, K, linestyle = "None", marker = "+", markersize=3, color="darkblue")
    ax.set_xlabel(r"$\Delta\lambda$ [nm]", fontsize=11)
    ax.set_ylabel(r"$\Delta K$ [%]", fontsize=11)
    ax.set_title(fr"Average error vs bandwidth scan at $\lambda={lambda_in[idx_cent]*1e9:.2f}$ nm", fontsize=11)
    ax.text(0.05, 0.6, "nsteps = 400", transform=ax.transAxes, fontsize=9,
        bbox=dict(facecolor='w', alpha=0.7, edgecolor='black', boxstyle='round,pad=0.2'))
    fig.tight_layout()
    plt.show()

# --------------------------------------------------------------------------------------------------------------------------------------

def simple_v3():
    print("\n**    RUNNING SIMPLE CONFIG    **")

    # ------------------------------------------------------------------------------------------------
    # Monochromatic run to get reference
    lambda_in, sigma_in = setup_cross(bandwidth = 0)

    fluence_in = np.linspace(0.1, 10, 51)*1e4
    fluence_out_analytic = np.zeros(len(fluence_in))
    fluence_out_numeric = np.zeros(len(fluence_in))

    idx_cent = int((len(lambda_in)-1)/2)        # central wavelength idx generalisation for broadband case.

    fluence_sat = (2*pi*hbar*c)/(sigma_in[idx_cent]*lambda_in[idx_cent])
    G0 = params.alpha_amp**(sigma_in[idx_cent]/params.sigmao)

    ltrans = params.loss_transmission

    # ------------------------------------------------------------------------------------------------
    # Scanning analytical
    for el in range(0, len(fluence_in)):
        fluence = fluence_in[el]
        for p in range(0,3):
            fluence = fluence*ltrans
            a = np.exp(fluence/fluence_sat) - 1
            b = (G0**(1/3))*(a) + 1
            fluence = fluence_sat*np.log(b)
            fluence = fluence*ltrans
        fluence_out_analytic[el] = fluence

    # ------------------------------------------------------------------------------------------------
    # Scanning number of timesteps
    N = np.linspace(10, 200, 191)
    K = np.array([])
    for dn in N:
        print(dn)
        for el in range(0, len(fluence_in)):
            # run numerical calculation 
            tau, intensity_spec_in, imin, imax = intensity_profile(params.duration, 1.5, lambda_in, fluence_in[el]/params.duration, n = int(dn))
            intensity_spec_out, _G_, _N_, _E_ = simple(params.alpha_disk, intensity_spec_in, lambda_in, tau, sigma_in)
            intensity_tot_out = trapper(intensity_spec_out, lambda_in)

            fluence_out_numeric[el] = trapper(intensity_tot_out, tau)

        # find average percentage error
        pe = np.average(np.abs(100*(fluence_out_numeric - fluence_out_analytic)/fluence_out_analytic))
        K = np.append(K, pe)

    K = np.array(K)
    idx_cent = int((len(lambda_in)-1)/2)
    fig, ax = plt.subplots(1,1, figsize = (5.5, 3.5), dpi = 200)
    ax.plot(N, K, linestyle = "-", linewidth=0.5,color="firebrick",
        marker = "+", markersize=3, markerfacecolor="mediumseagreen", markeredgecolor = "mediumseagreen")
    ax.set_xlabel(r"n steps", fontsize=11)
    ax.set_ylabel(r"$\Delta K$ [%]", fontsize=11)
    ax.set_title(fr"Average error vs number of steps scan at $\lambda={lambda_in[idx_cent]*1e9:.2f}$ nm", fontsize=11)
    fig.tight_layout()
    plt.show()