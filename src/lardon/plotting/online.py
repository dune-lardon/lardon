import lardon.config as cf
import lardon.data_containers as dc
import lardon.channel_mapping as chmap

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
from lardon.plotting.save_plot import *



def plot_noise_all_crps(option=None, to_be_shown=False):

    
    fig = plt.figure(figsize=(12,4))
    gs = gridspec.GridSpec(nrows=2, 
                           ncols=2)#, wspace=0.08)
    


    ax_crps  = [fig.add_subplot(gs[1,0]), fig.add_subplot(gs[1,1]), fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1])] 
    

    ch_start = [0, 3072, 6144, 9216]
    nch = 3072
    crp_name = cf.module_name

    raw = chmap.arange_in_glob_channels(list(itr.chain.from_iterable(x.ped_rms for x in dc.evt_list[-1].noise_raw)))
    filt = chmap.arange_in_glob_channels(list(itr.chain.from_iterable(x.ped_rms for x in dc.evt_list[-1].noise_filt)))
    print(len(raw))
    
    for i in range(cf.n_module):

        ax_crps[i].set_title(crp_name[i])
        chan = np.linspace(ch_start[i], ch_start[i]+nch, nch, endpoint=False)
        ax_crps[i].scatter(chan, raw[ch_start[i]:ch_start[i]+3072], s=2, c='k', label='raw')
        ax_crps[i].scatter(chan, filt[ch_start[i]:ch_start[i]+3072], s=2, c='r', label='filtered')

        for vch in [951.5, 1903.5]:
            ax_crps[i].axvline(ch_start[i]+vch, c='k', ls='dashed')
        ax_crps[i].set_xlim(chan[0], chan[-1])
        ax_crps[i].set_ylim(0, 50)
    for ax in ax_crps:
        ax.set_ylabel('Ped. RMS [ADC]')
        ax.set_xlabel('Global Channel Nb')

    plt.tight_layout()


    save_with_details(fig, option, 'noise_all_crps')


    if(to_be_shown):
        plt.show()
