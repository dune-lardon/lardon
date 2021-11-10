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


def draw_current_waveform(view, ch, ax=None, **kwargs):    
    
    ax = plt.gca() if ax is None else ax
    
    ax.plot(dc.data[view, ch, :], **kwargs)
    return ax



def plot_wvf_current_daqch(daqch_list, adc_min=-1, adc_max=-1):
    """ daqch_list should be a list of daqchs """
    n_wvf = len(daqch_list)

    fig = plt.figure(figsize=(12, 3*n_wvf))
    gs = gridspec.GridSpec(nrows=n_wvf, ncols=1)
    ax = [fig.add_subplot(gs[i,0]) for i in range(n_wvf)]

    
    for i in range(n_wvf):
        daqch = daqch_list[i]
        legend = "DAQ ch"+str(daqch)
        view, ch = cf.chmap[daqch].view, cf.chmap[daqch].vchan
        ax[i] = draw_current_waveform(view, ch, ax=ax[i], label=legend, c='k')
        ax[i].set_ylabel('ADC')
        ax[i].legend(loc='upper right')
        ax[i].set_xlim([0, cf.n_sample])
        if(adc_min > -1):
            ax[i].set_ybound(lower=adc_min)
        if(adc_max > -1):
            ax[i].set_ybound(upper=adc_max)
            
    for a in ax[:-1]:
        a.tick_params(labelbottom=False)
    ax[-1].set_xlabel('Time')
    

    plt.tight_layout()
