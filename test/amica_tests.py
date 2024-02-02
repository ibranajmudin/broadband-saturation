from common.setup import intensity_profile
from common.integrators import trapper
from runs.amica import amica
from common.setup import setup_cross
import common.setup as setup

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import ScalarFormatter
from scipy.constants import pi

import params

def iterate_amica():
    print("\n**    RUNNING AMICA CONFIG    **")

    lambda_in, sigma_in = setup_cross("saturation/resources/glass_data.csv")

    # PARAMS
    intensity_peak_in   =   params.e_in  / (params.area*params.duration)
    intensity_peak_out  =   params.e_out / (params.area*params.duration)

    # Make initial signal guess (gonna be flat), and target pulse (also flat)
    tau, intensity_spec_in, imin, imax = intensity_profile(params.duration, 1.5, lambda_in, intensity_peak_in)
    tau, intensity_spec_out, imin, imax = intensity_profile(params.duration, 1.5, lambda_in, intensity_peak_out)

    # Iterate through with P value
    P = 0.7

    # fig, [ax1, ax2, ax3] = plt.subplots(1,3, figsize=(10,4), dpi=150)
    fig, [ax1, ax2] = plt.subplots(1,2, figsize=(8,3), dpi=200)
    figs, axs = plt.subplots(1,1, figsize=(5,3.5), dpi=200)
    figo, axo = plt.subplots(1,1, figsize=(5,3.5), dpi=200)
    fign, axn = plt.subplots(1,1, figsize=(5,3.5), dpi=200)
    fige, axe = plt.subplots(1,1, figsize=(5,3.5), dpi=200)

    legend_in = []
    legend_out = []

    its = 6
    for it in range(0, its):
        print("iteration:", it, end = " ")
        # ----------------------------------------------------------------------------------------
        # 1. find intensity signal and print input energy for each run 
        intensity_tot_in = trapper(intensity_spec_in[:,:], lambda_in)
        energy_in = trapper(intensity_tot_in, tau) * params.area
        print(f"Ein = {energy_in*1e6:.2f}", "µJ", end = " ")
        
        # ----------------------------------------------------------------------------------------
        # 2. pass through orion and get an outcome
        intensity_spec, _G_, _N_, _E_ = amica(params.alpha_disk, intensity_spec_in , lambda_in, tau, sigma_in)
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
    # Analysis of results - plotting and more plotting
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

    Nfront = np.array([])
    Nback = np.array([])
    for n in _N_:
        Nfront = np.append(Nfront, n[imin])
        Nback = np.append(Nback, n[imax])
    xax = np.linspace(1, len(Nfront), len(Nfront))
    axn.plot(xax, np.exp(Nfront*params.sigmao)**3, color = 'royalblue', linestyle="None", marker='D', markersize = 2)
    axn.plot(xax, np.exp(Nback*params.sigmao)**3, color='maroon', linestyle="None", marker='o', markersize = 2)

    _E_ = np.array(_E_)
    _E_ = np.append(_E_, energy)
    axe.plot(_E_*1e-3, color='blueviolet', linewidth=2)
    axe.set_title("Energy after passing through the disks")
    axe.set_xlabel("Disk #")
    axe.set_ylabel("Energy [kJ]")
    axe.set_xlim(left=0)
    axe.set_ylim(bottom=0)

    texte0a = r"$\lambda_{\text{cent}} = $" + f"{setup.lambda_[np.argmax(setup.sigma_) + params.offset]*1e9:.2f} nm" + "\n"
    texte0b = r"$\Delta \lambda = $" + f"{params.bandwidth*1e9:.1f} nm" + "\n"
    text0 = texte0a + texte0b
    texte1 = r"$\alpha_0$ = " + f"{params.alpha_amp:.1f}" + "\n"
    texte2 = r"$T_{\text{trans}} = $" + f"{params.loss_transmission*100:.1f} %" + "\n"
    texte3 = r"$T_{\text{pepc}} = $" + f"{params.loss_pepc*100:.1f} %" + "\n"
    texte4 = r"$T_{\text{pin}} = $" + f"{params.loss_pinhole*100:.1f} %"
    texte = text0 + texte1 + texte2 + texte3 + texte4
    axe.text(0.1, 0.85, texte, transform=ax1.transAxes, fontsize = 6, 
            bbox=dict(facecolor='lavender', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.2'))
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

    text1 = r"$E_{\text{in}} = $" + f"{energy_in*1e6:.1f} µJ"
    ax1.text(0.04, 0.9, text1, transform=ax1.transAxes, fontsize=8, 
            bbox=dict(facecolor='papayawhip', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.2'))

    text2 = r"$E_{\text{out}} = $" + f"{energy*1e-3:.2f} kJ"
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
