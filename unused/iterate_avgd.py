from common.setup import intensity_profile
from runs.orion import orion

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import ScalarFormatter
from scipy.constants import pi

# This script takes the average spectrum of the final pulse.
# Then normalises this pulse, and then makes a new input pulse
# using this normalised spectrum, and calculates what would happen when
# re-input into orion.

def iterate_avgd(lambda_in, sigma_in):
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
    figs, axs = plt.subplots(1,1, figsize=(5,4), dpi=150)

    legend_in = []
    legend_out = []

    its = 11
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
        # 3. print output energy for reference
        intensity_tot = np.trapz(intensity_spec[:,:], lambda_in)
        energy = np.trapz(intensity_tot, tau) * (1.5*pi*((diameter/2)**2))
        print(f"{energy*1e-3:.4f}", "kJ")

        # ----------------------------------------------------------------------------------------
        # 4. use this conditional to only plot the final pulse
        if i == its-1:
            # 4a.
            # ax1.plot(tau*1e9, intensity_tot_in*1e4, linewidth=2, color=colors[i])
            # ax2.plot(tau*1e9, intensity_tot*1e4, linewidth=2, color=colors[i])
            # axs.plot(lambda_in*1e9, intensity_spec_in[imin]*1e-9, color="b")
            # axs.plot(lambda_in*1e9, intensity_spec_in[imax]*1e-9, color="b")
            # average intensity spectrum
            intensity_spec_in_avg = np.average(intensity_spec_in[imin:imax+1], axis=0)
            # normalised average intensity spectrum
            intensity_spec_in_avg_norm = intensity_spec_in_avg/np.trapz(intensity_spec_in_avg[:], lambda_in)

            axs.plot(lambda_in*1e9, intensity_spec_in_avg_norm*1e-9, color="r")

            intensity_spec_in_new = np.outer(intensity_tot_in, intensity_spec_in_avg_norm)
            intensity_spec_new, _G_, _N_ = orion(alpha_disk,intensity_spec_in_new , lambda_in, tau, sigma_in)

            intensity_tot_in_new = np.trapz(intensity_spec_in_new[:,:], lambda_in)
            intensity_tot_new = np.trapz(intensity_spec_new[:,:], lambda_in)

            energy_in = np.trapz(intensity_tot_in_new, tau) * (1.5*pi*((diameter/2)**2))
            energy = np.trapz(intensity_tot_new, tau) * (1.5*pi*((diameter/2)**2))

            ax1.plot(tau*1e9, intensity_tot_in_new*1e-4, linewidth=2, color=colors[i])
            ax2.plot(tau*1e9, intensity_tot_new*1e-4, linewidth=2, color=colors[i])

        
        # ----------------------------------------------------------------------------------------
        # 5. adjust input using iterative formula accordingly
        intensity_spec_in[imin:imax+1] = intensity_spec_in[imin:imax+1]*(P*(intensity_spec_out[imin:imax+1]/intensity_spec[imin:imax+1]-1)+1)


        legend_in.append(str(i) + r" - $E_{in} = $" f"{energy_in*1e3:.0f} mJ" )
        legend_out.append(str(i) + r" - $E_{out} = $" f"{energy*1e-3:.2f} kJ" )

    # ----------------------------------------------------------------------------------------
    # 6. formatting axis and figure
    legend_in = np.array(legend_in)
    legend_out = np.array(legend_out)
    ax1.set_xlabel(r"$\tau$ [ns]", fontsize=10); ax2.set_xlabel(r"$\tau$ [ns]", fontsize=10)
    ax1.set_ylabel(r"$I_{in}$[W/cm$^2$]", fontsize=10); ax2.set_ylabel(r"$I_{out}$[W/cm$^2$]", fontsize=10)
    ax1.set_title(r"Input pulse $S$", fontsize=10)
    ax2.set_title(r"Corresponding output pulse $R$", fontsize=10)
    # ax1.legend(legend_in, fontsize=8, title="iteration")
    # ax2.legend(legend_out, fontsize=8, title="iteration")
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

    text1 = r"$E_{in} = $" + f"{energy_in*1e3:.0f} mJ"
    ax1.text(0.04, 0.9, text1, transform=ax1.transAxes, 
            bbox=dict(facecolor='w', alpha=0.7, edgecolor='black', boxstyle='round,pad=0.2'))

    text2 = r"$E_{out} = $" + f"{energy*1e-3:.2f} kJ"
    ax2.text(0.04, 0.9, text2, transform=ax2.transAxes, 
            bbox=dict(facecolor='w', alpha=0.7, edgecolor='black', boxstyle='round,pad=0.2'))

    axs.set_xlabel(r"$\lambda$ [nm]")
    axs.set_ylabel(r"$F(\lambda)$ [/nm]")
    # axs.set_ylabel(r"$F(\lambda)$ [/nm]")
    axs.set_title(r"Normalised average input spectral profile $F(\lambda)$", fontsize=10)

    ymin, ymax = axs.get_ylim()
    axs.set_xlim(left = lambda_in[0]*1e9, right = lambda_in[-1]*1e9)
    axs.set_ylim(bottom=0)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.gca().yaxis.set_major_formatter(formatter)
    axs.vlines(lambda_in*1e9, ymin=0, ymax=ymax/100, colors='black', linewidth=1, alpha=1)

    fig.tight_layout()
    figs.tight_layout()
    plt.show()




