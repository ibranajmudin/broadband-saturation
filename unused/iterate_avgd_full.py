from common.setup import intensity_profile
from runs.orion import orion

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import ScalarFormatter
from scipy.constants import pi

# This script does the iterative process and plots all iterations onto one graph

def iterate_avgd_full(lambda_in, sigma_in):
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
        # 3. print output energy for reference and plot output pulse
        intensity_tot = np.trapz(intensity_spec[:,:], lambda_in)
        energy = np.trapz(intensity_tot, tau) * (1.5*pi*((diameter/2)**2))
        print(f"{energy*1e-3:.4f}", "kJ")

        # 4a. uncomment for plotting all pulses (no spectral)
        ax1.plot(tau*1e9, intensity_tot_in*1e-4, linewidth=2, color=colors[i])
        ax2.plot(tau*1e9, intensity_tot*1e-4, linewidth=2, color=colors[i])
        
        # ----------------------------------------------------------------------------------------
        # 4b. uncomment for plotting final pulses average spectral shape throughout pulse (with spectral)
        # if i == its-1:
        #     ax1.plot(tau*1e9, intensity_tot_in*1e-4, linewidth=2, color=colors[i])
        #     ax2.plot(tau*1e9, intensity_tot*1e-4, linewidth=2, color=colors[i])
        #     axs.plot(lambda_in*1e9, intensity_spec_in[imin]*1e-9, linewidth=1.5, color=colors[i])
        #     axs.plot(lambda_in*1e9, intensity_spec_in[imax]*1e-9, linewidth=1.5, color=colors[i])
        #     # average intensity spectrum
        #     intensity_spec_in_avg = np.average(intensity_spec_in[imin:imax+1], axis=0)
        #     axs.plot(lambda_in*1e9, intensity_spec_in_avg*1e-9, color="r")  
        # ----------------------------------------------------------------------------------------
        # 5a. adjust input accordingly with iteration formula and find 'corrected' I(t)
        intensity_spec_in[imin:imax+1] = intensity_spec_in[imin:imax+1]*(P*(intensity_spec_out[imin:imax+1]/intensity_spec[imin:imax+1]-1)+1)
        intensity_tot_in_new = np.trapz(intensity_spec_in[:,:], lambda_in)

        # 5b. average the normalised spectrum
        intensity_spec_in_avg = np.average(intensity_spec_in[imin:imax+1], axis=0)
        intensity_spec_in_avg_norm = intensity_spec_in_avg/np.trapz(intensity_spec_in_avg[:], lambda_in)

        # 5c. and use linear relationship I(l,t)=I(t)*F(l) to get final version of 'correction'
        intensity_spec_in = np.outer(intensity_tot_in_new, intensity_spec_in_avg_norm)

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

    # text1 = r"$E_{in} = $" + f"{energy_in*1e3:.0f} mJ"
    # ax1.text(0.04, 0.9, text1, transform=ax1.transAxes, 
    #         bbox=dict(facecolor='w', alpha=0.7, edgecolor='black', boxstyle='round,pad=0.2'))

    # text2 = r"$E_{out} = $" + f"{energy*1e-3:.2f} kJ"
    # ax2.text(0.04, 0.9, text2, transform=ax2.transAxes, 
    #         bbox=dict(facecolor='w', alpha=0.7, edgecolor='black', boxstyle='round,pad=0.2'))
    # plt.xticks(fontsize=8)
    # plt.yticks(fontsize=8)

    fig.tight_layout()
    plt.show()




