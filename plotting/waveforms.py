import config as cf
import data_containers as dc
import channel_mapping as cmap

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import collections  as mc
from matplotlib.legend_handler import HandlerTuple
import itertools as itr
import math
import colorcet as cc


def draw_current_waveform(daqch, view, ch, ax=None, **kwargs):    
    
    ax = plt.gca() if ax is None else ax
    if(daqch > 0):
        ax.plot(dc.data_daq[daqch, :], **kwargs)
    else:
        ax.plot(dc.data[view, ch, :], **kwargs)
    return ax



def plot_wvf_current_vch(vch_list, adc_min=-1, adc_max=-1, tmin=0, tmax=cf.n_sample, option=None, to_be_shown=False):
    cmap.arange_in_view_channels()
    n_wvf = len(vch_list)

    fig = plt.figure(figsize=(12, 3*n_wvf))
    gs = gridspec.GridSpec(nrows=n_wvf, ncols=1)
    ax = [fig.add_subplot(gs[i,0]) for i in range(n_wvf)]

    
    for i in range(n_wvf):
        view, ch = vch_list[i]
        legend = "v"+str(view)+" ch"+str(ch)
        ax[i] = draw_current_waveform(-1, view, ch, ax=ax[i], label=legend, c='k')
        ax[i].set_ylabel('ADC')
        ax[i].legend(loc='upper right')
        ax[i].set_xlim([tmin, tmax])
        if(adc_min > -1):
            ax[i].set_ybound(lower=adc_min)
        if(adc_max > -1):
            ax[i].set_ybound(upper=adc_max)
            
    for a in ax[:-1]:
        a.tick_params(labelbottom=False)
    ax[-1].set_xlabel('Time')
    

    plt.tight_layout()

    run_nb = str(dc.evt_list[-1].run_nb)
    evt_nb = str(dc.evt_list[-1].trigger_nb)
    elec   = dc.evt_list[-1].elec

    if(option):
        option = "_"+option
    else:
        option = ""


    plt.savefig(cf.plot_path+'/waveforms'+option+'_'+elec+'_run_'+run_nb+'_evt_'+evt_nb+'.png')

    if(to_be_shown):
        plt.show()

    plt.close()


def plot_wvf_current_daqch(daqch_list, adc_min=-1, adc_max=-1, option=None, to_be_shown=False, tmin=0, tmax=cf.n_sample):
    """ daqch_list should be a list of daqchs """
    n_wvf = len(daqch_list)

    fig = plt.figure(figsize=(12, 3*n_wvf))
    gs = gridspec.GridSpec(nrows=n_wvf, ncols=1)
    ax = [fig.add_subplot(gs[i,0]) for i in range(n_wvf)]

    
    for i in range(n_wvf):
        daqch = daqch_list[i]
        legend = "DAQ ch"+str(daqch)
        view, ch = dc.chmap[daqch].view, dc.chmap[daqch].vchan
        ax[i] = draw_current_waveform(daqch, -1, -1, ax=ax[i], label=legend, c='k')
        ax[i].set_ylabel('ADC')
        ax[i].legend(loc='upper right')
        ax[i].set_xlim([tmin,tmax])
        if(adc_min > -1):
            ax[i].set_ybound(lower=adc_min)
        if(adc_max > -1):
            ax[i].set_ybound(upper=adc_max)
            
    for a in ax[:-1]:
        a.tick_params(labelbottom=False)
    ax[-1].set_xlabel('Time')
    

    plt.tight_layout()

    run_nb = str(dc.evt_list[-1].run_nb)
    evt_nb = str(dc.evt_list[-1].trigger_nb)
    elec   = dc.evt_list[-1].elec

    if(option):
        option = "_"+option
    else:
        option = ""


    plt.savefig(cf.plot_path+'/waveforms'+option+'_'+elec+'_run_'+run_nb+'_evt_'+evt_nb+'.png')

    if(to_be_shown):
        plt.show()

    plt.close()
