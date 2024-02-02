from common.setup import intensity_profile
from common.integrators import trapper
from runs.orion import orion


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import ScalarFormatter
from scipy.constants import pi
from common.setup import setup_cross

import params


def iterate_v1(lambda_in, sigma_in):
    # PARAMS
    energy_in = 0.15        #   Input energy guess
    duration = 10e-9        #   10 ns pulse duration
    diameter = 0.175        #   175 mm beam diameter

    energy_out = 2000       #   Output energy
    
    intensity_peak_in   =   energy_in  / (1.5*(pi*((diameter/2)**2))*duration)
    intensity_peak_out  =   energy_out / (1.5*(pi*((diameter/2)**2))*duration)

    alpha_disk = np.cbrt(2.1)

    # Make initial signal guess (gonna be flat)
    tau, intensity_spec_in, imin, imax = intensity_profile(duration, 1.5, lambda_in, intensity_peak_in)

    # Define target 
    tau, intensity_spec_out, imin, imax = intensity_profile(duration, 1.5, lambda_in, intensity_peak_out)

    # Iterate through with P value
    P = 0.7

    # fig, [ax1, ax2, ax3] = plt.subplots(1,3, figsize=(10,4), dpi=150)
    fig, [ax1, ax2] = plt.subplots(1,2, figsize=(10,4), dpi=150)

    legend_in = []
    legend_out = []

    its = 5
    colors = cm.rainbow(np.linspace(0, 1, its))
    for i in range(0, its):
        print("iteration:", i)
        # ----------------------------------------------------------------------------------------
        # 1. print input energy for each run
        intensity_tot_in = np.trapz(intensity_spec_in[:,:], lambda_in)
        energy_in = np.trapz(intensity_tot_in, tau) * (1.5*pi*((diameter/2)**2))
        print(f"{energy_in*1e3:.4f}", "mJ")
        
        # ----------------------------------------------------------------------------------------
        # 2. pass through orion and get an outcome
        intensity_spec, _G_, _N_ = orion(alpha_disk,intensity_spec_in , lambda_in, tau, sigma_in)
        # ----------------------------------------------------------------------------------------
        # 3. print output energy for reference and plot output pulse
        intensity_tot = np.trapz(intensity_spec[:,:], lambda_in)
        energy = np.trapz(intensity_tot, tau) * (1.5*pi*((diameter/2)**2))
        print(f"{energy*1e-3:.4f}", "kJ")

        # 4a. uncomment for plotting all pulses (no spectral)
        ax1.plot(tau*1e9, intensity_tot_in*1e-4, linewidth=2, color=colors[i])
        ax2.plot(tau*1e9, intensity_tot*1e-4, linewidth=2, color=colors[i])
        

        # ----------------------------------------------------------------------------------------
        # 5. adjust input accordingly
        intensity_spec_in[imin:imax+1] = intensity_spec_in[imin:imax+1]*(P*(intensity_spec_out[imin:imax+1]/intensity_spec[imin:imax+1]-1)+1)
        
        legend_in.append(str(i) + r" - $E_{in} = $" f"{energy_in*1e3:.0f} mJ" )
        legend_out.append(str(i) + r" - $E_{out} = $" f"{energy*1e-3:.2f} kJ" )

    # ----------------------------------------------------------------------------------------
    # 6. Formatting your axis and figures
    legend_in = np.array(legend_in)
    legend_out = np.array(legend_out)
    ax1.set_xlabel(r"$\tau$ [ns]", fontsize=10); ax2.set_xlabel(r"$\tau$ [ns]", fontsize=10)
    ax1.set_ylabel(r"$I_{in}$[W/cm$^2$]", fontsize=10); ax2.set_ylabel(r"$I_{out}$[W/cm$^2$]", fontsize=10)
    ax1.set_title(r"Input pulse $S$", fontsize=10)
    ax2.set_title(r"Corresponding output pulses $R$", fontsize=10)
    ax1.legend(legend_in, fontsize=8, title="iteration")
    ax2.legend(legend_out, fontsize=8, title="iteration")

    formatter = ScalarFormatter(useMathText=True, useOffset=False)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    ax1.yaxis.set_major_formatter(formatter)
    ax1.get_yaxis().get_offset_text().set_position((-0.12, 0))

    ax2.yaxis.set_major_formatter(formatter)
    ax2.get_yaxis().get_offset_text().set_position((-0.12, 0))

    formatter = ScalarFormatter(useMathText=True, useOffset=False)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))

    # plt.margins(0)
    ax1.set_ylim(bottom = 0, top = (np.max(intensity_tot_in*1e-4)*1.6))
    ax2.set_ylim(bottom = 0, top = (np.max(intensity_tot*1e-4)*2.3))

    fig.tight_layout()
    plt.show()



# This script does the iterative process and plots the final iteration

def iterate_v4():
    print("\n**    RUNNING ORION CONFIG    **")

    lambda_in, sigma_in = setup_cross("saturation/resources/glass_data.csv")

    # PARAMS
    intensity_peak_in   =   params.e_in  / (params.area*params.duration)
    intensity_peak_out  =   params.e_out / (params.area*params.duration)

    # Make initial signal guess (gonna be flat), and target pulse (also flat)
    tau, intensity_spec_in, imin, imax = intensity_profile(params.duration, 1.5, lambda_in, intensity_peak_in)
    tau, intensity_spec_out, imin, imax = intensity_profile(params.duration, 1.5, lambda_in, intensity_peak_out)

    # Iterate through with P value
    P = 0.75

    # fig, [ax1, ax2, ax3] = plt.subplots(1,3, figsize=(10,4), dpi=150)
    fig, [ax1, ax2] = plt.subplots(1,2, figsize=(8,3), dpi=200)
    figs, axs = plt.subplots(1,1, figsize=(5,3.5), dpi=200)
    figo, axo = plt.subplots(1,1, figsize=(5,3.5), dpi=200)
    fign, axn = plt.subplots(1,1, figsize=(5,3.5), dpi=200)
    fige, axe = plt.subplots(1,1, figsize=(5,3.5), dpi=200)

    legend_in = []
    legend_out = []

    its = 14
    for it in range(0, its):
        print("iteration:", it, end = " ")
        # ----------------------------------------------------------------------------------------
        # 1. find intensity signal and print input energy for each run 
        intensity_tot_in = trapper(intensity_spec_in[:,:], lambda_in)
        energy_in = trapper(intensity_tot_in, tau) * params.area
        print(f"Ein = {energy_in*1e3:.2f}", "mJ", end = " ")
        
        # ----------------------------------------------------------------------------------------
        # 2. pass through orion and get an outcome
        intensity_spec, _G_, _N_, _E_ = orion(params.alpha_disk, intensity_spec_in , lambda_in, tau, sigma_in)
        # ----------------------------------------------------------------------------------------
        # 3. find intensity output and print output energy for reference
        intensity_tot = trapper(intensity_spec[:,:], lambda_in)
        energy = trapper(intensity_tot, tau) * params.area
        print(f"Eout = {energy*1e-3:.3f}", "kJ")

        # ----------------------------------------------------------------------------------------
        # 4a. adjust input accordingly with iteration formula and find 'corrected' I(t)
        if it < its - 1: # exclude final correction as it's not looped back to calc a new output
            intensity_spec_in[imin:imax+1] = intensity_spec_in[imin:imax+1]*(P*(intensity_spec_out[imin:imax+1]/intensity_spec[imin:imax+1]-1)+1)
            intensity_tot_in_new = trapper(intensity_spec_in[:,:], lambda_in)

            # 4b. average the normalised spectrum
            intensity_spec_in_avg = np.average(intensity_spec_in[imin:imax+1], axis=0)
            intensity_spec_in_avg_norm = intensity_spec_in_avg/trapper(intensity_spec_in_avg[:], lambda_in)

            # 4c. and use linear relationship I(l,t)=I(t)*F(l) to get final version of 'correction'
            intensity_spec_in = np.outer(intensity_tot_in_new, intensity_spec_in_avg_norm)

    # ----------------------------------------------------------------------------------------
    # Analysis of results
    ax1.plot(tau*1e9, intensity_tot_in*1e-4, linewidth=2, color="r")
    ax2.plot(tau*1e9, intensity_tot*1e-4, linewidth=2, color="r")
    # initial spectral difference
    axs.plot(lambda_in*1e9, (intensity_spec_in[imin]*1e-9)/intensity_tot_in[imin], color="mediumblue", linewidth = 2.2)
    axs.plot(lambda_in*1e9, (intensity_spec_in[imax]*1e-9)/intensity_tot_in[imax], color="mediumseagreen")
    # final spectral difference
    axo.plot(lambda_in*1e9, (intensity_spec[imin]*1e-9)/intensity_tot[imin], color="mediumblue")
    axo.plot(lambda_in*1e9, (intensity_spec[imax]*1e-9)/intensity_tot[imax], color="mediumseagreen")
    # average intensity spectrum
    # intensity_spec_in_avg = np.average(intensity_spec_in[imin:imax+1], axis=0)

    # finding critical energy
    # use _G_ which containes gain factors experienced by pulse in order.
    intensity_spec_crit = intensity_spec_in*(params.loss_transmission**65)*(params.loss_pinhole**2)
    for g in _G_[0:24] : intensity_spec_crit=intensity_spec_crit*g
    intensity_tot_crit = trapper(intensity_spec_crit[:,:], lambda_in)
    energy_crit = trapper(intensity_tot_crit, tau) * params.area
    print("CRITICAL ENERGY = " + f"{energy_crit:.2f}" + " J")

    Nfront = np.array([])
    Nback = np.array([])
    for n in _N_:
        Nfront = np.append(Nfront, n[imin])
        Nback = np.append(Nback, n[imax])
    xax = np.linspace(1, 48, 48)
    axn.plot(xax, np.exp(Nfront*params.sigmao)**3, color = 'royalblue', linestyle="None", marker='D', markersize = 2)
    axn.plot(xax, np.exp(Nback*params.sigmao)**3, color='maroon', linestyle="None", marker='o', markersize = 2)

    _E_ = np.array(_E_)
    axe.plot(_E_, color='maroon', linestyle="None", marker='o', markersize = 2)
    # ----------------------------------------------------------------------------------------
    # 6. formatting axis and figure
    legend_in = np.array(legend_in)
    legend_out = np.array(legend_out)
    ax1.set_xlabel(r"$\tau$ [ns]", fontsize=10); ax2.set_xlabel(r"$\tau$ [ns]", fontsize=10)
    ax1.set_ylabel(r"$I_{in}$[W/cm$^2$]", fontsize=10); ax2.set_ylabel(r"$I_{out}$[W/cm$^2$]", fontsize=10)
    ax1.set_title(r"Input pulse", fontsize=10)
    ax2.set_title(r"Corresponding output pulse", fontsize=10)
    fig.subplots_adjust(wspace=0.3)

    formatter = ScalarFormatter(useMathText=True, useOffset=False)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    ax1.yaxis.set_major_formatter(formatter)
    ax1.get_yaxis().get_offset_text().set_position((-0.12, 0))

    ax2.yaxis.set_major_formatter(formatter)
    ax2.get_yaxis().get_offset_text().set_position((-0.12, 0))

    formatter = ScalarFormatter(useMathText=True, useOffset=False)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    # ax3.yaxis.set_major_formatter(formatter)
    # ax3.get_yaxis().get_offset_text().set_position((-0.12, 0))

    # plt.margins(0)
    ax1.set_ylim(bottom = 0, top = (np.max(intensity_tot_in*1e-4)*1.6))
    ax2.set_ylim(bottom = 0, top = (np.max(intensity_tot*1e-4)*2.3))
    ax1.grid()
    ax2.grid()

    text1 = r"$E_{\text{in}} = $" + f"{energy_in*1e3:.0f} mJ"
    ax1.text(0.04, 0.9, text1, transform=ax1.transAxes, fontsize=8, 
            bbox=dict(facecolor='papayawhip', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.2'))

    text2a = r"$E_{\text{crit}} = $" + f"{energy_crit:.1f} J" + "\n"
    text2b = r"$E_{\text{out}} = $" + f"{energy*1e-3:.2f} kJ"
    text2 = text2a + text2b
    ax2.text(0.04, 0.84, text2, transform=ax2.transAxes, fontsize=8, 
            bbox=dict(facecolor='papayawhip', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.2'))

    axn.legend(["front", "back"], fontsize = 10)
    axn.set_title("Small signal gain of each disk passage", fontsize = 11)
    axn.set_xlabel("Disk passed in time order", fontsize = 10)
    axn.set_ylabel(r"$\alpha_0$ - small signal gain per disk", fontsize = 10)
    ymin, ymax = axn.get_ylim()
    axn.vlines([12.5, 24.5, 36.5], ymin=ymin, ymax=ymax, colors='black', linewidth=0.6, alpha=0.8, linestyle="dashed")
    axn.set_ylim(bottom = ymin, top = ymax)
    axn.set_xlim(left= xax[0], right = xax[-1]*1.05 )
    xvals = [0.05, 0.285, 0.525, 0.765]
    yvals = [0.4, 0.4, 0.4, 0.4]
    txts = ["1st pass", "2nd, pass", "3rd pass", "4th pass"]
    for x,y,txt in zip(xvals,yvals,txts):
        axn.text(x, y, txt, transform = axn.transAxes, fontsize = 7)

    # both spectral graphs
    for ax in [axs, axo]:
        ax.set_xlabel(r"$\lambda$ [nm]")
        ax.set_ylabel(r"$F(\lambda)$ [/nm]")
        # axs.set_ylabel(r"$F(\lambda)$ [/nm]")
        if ax == axs: txt = "input"
        elif ax == axo: txt = "output"
        ax.set_title("Normalised " + txt + " spectral profiles", fontsize=10)
        ax.legend(["front", "back"])
    #     axs.text(0.75, 0.47, "front",transform=axs.transAxes,
    #             rotation = 48, rotation_mode='anchor', transform_rotates_text=True)
    #     axs.text(0.77, 0.30, "average",transform=axs.transAxes,
    #             rotation = 45, rotation_mode='anchor', transform_rotates_text=True)
    #     axs.text(0.8, 0.20, "back",transform=axs.transAxes,
    #             rotation = 35, rotation_mode='anchor', transform_rotates_text=True)

        ymin, ymax = ax.get_ylim()
        ax.set_xlim(left = lambda_in[0]*1e9, right = lambda_in[-1]*1e9)
        ax.set_ylim(bottom=0)
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-2, 2))
        plt.gca().xaxis.set_major_formatter(formatter)
        plt.gca().yaxis.set_major_formatter(formatter)

        ax.vlines(lambda_in*1e9, ymin=0, ymax=ymax/100, colors='black', linewidth=1, alpha=1)


    fig.tight_layout(w_pad = 3)
    figs.tight_layout()
    figo.tight_layout()
    fign.tight_layout()
    fige.tight_layout()
    plt.show()



