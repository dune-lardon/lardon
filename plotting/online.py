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
from matplotlib.colors import LogNorm
from .save_plot import *



def plot_noise_all_crps(option=None, to_be_shown=False):
    #if(
    
    fig = plt.figure(figsize=(12,4))
    gs = gridspec.GridSpec(nrows=2, 
                           ncols=2)#, wspace=0.08)
    

    #ax_crps  = [fig.add_subplot(gs[i,j]) for i in range(2) for j in range(2)]
    ax_crps  = [fig.add_subplot(gs[1,0]), fig.add_subplot(gs[1,1]), fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1])] 
    

    #print(len(raw), " and ", len(filt))

    ch_start = [6144, 9216, 0, 3072]
    nch = 3072
    crp_name = cf.module_name#['CRP 2', 'CRP 3', 'CRP 5', 'CRP 4']

    for i in range(cf.n_module):

        print('module ', i)
        ax_crps[i].set_title(crp_name[i])
        chan = np.linspace(ch_start[i], ch_start[i]+nch, nch, endpoint=False)
        print(i, len(chan), " vs ", len(dc.evt_list[-1].noise_raw[i].ped_rms))
        print(chan[:2])
        print(chan[-1])
    
        ax_crps[i].scatter(chan, dc.evt_list[-1].noise_raw[i].ped_rms, s=2, c='k', label='raw')
        ax_crps[i].scatter(chan, dc.evt_list[-1].noise_filt[i].ped_rms, s=2, c='r', label='filtered')

        for vch in [951.5, 1903.5]:
            ax_crps[i].axvline(ch_start[i]+vch, c='k', ls='dashed')
        ax_crps[i].set_xlim(chan[0], chan[-1])
        ax_crps[i].set_ylim(0, 50)
    for ax in ax_crps:
        ax.set_ylabel('Pedestal RMS [ADC]')
        ax.set_xlabel('Global Channel Nb')

    plt.tight_layout()
    plt.show()
