import config as cf
import data_containers as dc

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import collections  as mc
from matplotlib.legend_handler import HandlerTuple
import itertools as itr
import math
import colorcet as cc


def plot_raw_noise_daqch():
    fig = plt.figure(figsize=(12,4))
    gs = gridspec.GridSpec(nrows=2, 
                           ncols=1, wspace=0.08)
    
    #ax_mean = fig.add_subplot(gs[0,0])
    ax_std  = fig.add_subplot(111) #gs[1,0], sharex=ax_mean)

    ch = np.linspace(0, cf.n_tot_channels, cf.n_tot_channels, endpoint=False)
    #ax_mean.plot(dc.evt_list[-1].noise_raw.ped_mean)
    #ax_mean.set_ylabel('Mean Ped [ADC]')
    ax_std.scatter(ch, dc.evt_list[-1].noise_raw.ped_rms)
    ax_std.set_ylabel('RMS Ped [ADC]')
    ax_std.set_xlabel('DAQ Channel Number')
    ax_std.set_title('Raw Noise')

    plt.tight_layout()
    plt.show()



def plot_raw_noise_view():
    fig = plt.figure(figsize=(12,4))
    gs = gridspec.GridSpec(nrows=1, 
                           ncols=3)

    axs  = [fig.add_subplot(gs[0,x]) for x in range(3)]

    for iv in range(cf.n_view):
        vchan = [i for i in range(cf.n_tot_channels) if dc.chmap[i].view == iv ]
        rms = [dc.evt_list[-1].noise_raw.ped_rms[i] for i in vchan]
        ch = np.linspace(0, cf.view_nchan[iv], cf.view_nchan[iv], endpoint=False)
        axs[iv].scatter(ch, rms)
        axs[iv].set_ylabel('RMS Ped [ADC]')
        axs[iv].set_xlabel('Channel Number')
        axs[iv].set_title('View '+str(iv)+'/'+cf.view_name[iv])


    plt.tight_layout()
    plt.show()



