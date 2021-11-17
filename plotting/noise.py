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

cmap_fft = cc.cm.CET_CBL2_r
cmap_corr = cc.cm.CET_D9



def plot_raw_noise_daqch(option=None, to_be_shown=False):
    fig = plt.figure(figsize=(12,4))
    gs = gridspec.GridSpec(nrows=2, 
                           ncols=1, wspace=0.08)
    
    #ax_mean = fig.add_subplot(gs[0,0])
    ax_std  = fig.add_subplot(111) #gs[1,0], sharex=ax_mean)

    ch = np.linspace(0, cf.n_tot_channels, cf.n_tot_channels, endpoint=False)
    #ax_mean.plot(dc.evt_list[-1].noise_raw.ped_mean)
    #ax_mean.set_ylabel('Mean Ped [ADC]')
    ax_std.scatter(ch, dc.evt_list[-1].noise_raw.ped_rms,s=2)
    ax_std.set_ylabel('RMS Ped [ADC]')
    ax_std.set_xlabel('DAQ Channel Number')
    ax_std.set_title('Raw Noise')

    plt.tight_layout()


    run_nb = str(dc.evt_list[-1].run_nb)
    evt_nb = str(dc.evt_list[-1].trigger_nb)
    elec   = dc.evt_list[-1].elec

    if(option):
        option = "_"+option
    else:
        option = ""


    plt.savefig(cf.plot_path+'/ped_raw_rms_daqch'+option+'_'+elec+'_run_'+run_nb+'_evt_'+evt_nb+'.png')

    if(to_be_shown):
        plt.show()

    plt.close()


def plot_filt_noise_daqch(option=None, to_be_shown=False):
    fig = plt.figure(figsize=(12,4))
    gs = gridspec.GridSpec(nrows=2, 
                           ncols=1, wspace=0.08)
    
    #ax_mean = fig.add_subplot(gs[0,0])
    ax_std  = fig.add_subplot(111) #gs[1,0], sharex=ax_mean)

    ch = np.linspace(0, cf.n_tot_channels, cf.n_tot_channels, endpoint=False)
    #ax_mean.plot(dc.evt_list[-1].noise_filt.ped_mean)
    #ax_mean.set_ylabel('Mean Ped [ADC]')
    ax_std.scatter(ch, dc.evt_list[-1].noise_filt.ped_rms,s=2)

    ax_std.set_ylabel('RMS Ped [ADC]')
    ax_std.set_xlabel('DAQ Channel Number')
    ax_std.set_title('Filtered Noise')

    plt.tight_layout()


    run_nb = str(dc.evt_list[-1].run_nb)
    evt_nb = str(dc.evt_list[-1].trigger_nb)
    elec   = dc.evt_list[-1].elec

    if(option):
        option = "_"+option
    else:
        option = ""


    plt.savefig(cf.plot_path+'/ped_filt_rms_daqch'+option+'_'+elec+'_run_'+run_nb+'_evt_'+evt_nb+'.png')

    if(to_be_shown):
        plt.show()

    plt.close()



def plot_raw_noise_vch(option=None, to_be_shown=False):
    fig = plt.figure(figsize=(12,4))
    gs = gridspec.GridSpec(nrows=1, 
                           ncols=3)

    axs  = [fig.add_subplot(gs[0,x]) for x in range(3)]

    for iv in range(cf.n_view):
        vchan = [i for i in range(cf.n_tot_channels) if dc.chmap[i].view == iv and dc.chmap[i].vchan >=0 ]
        rms = [dc.evt_list[-1].noise_raw.ped_rms[i] for i in vchan]
        ch = np.linspace(0, cf.view_nchan[iv], cf.view_nchan[iv], endpoint=False)

        axs[iv].scatter(ch, rms,s=2)
        axs[iv].set_ylabel('RMS Ped [ADC]')
        axs[iv].set_xlabel('Channel Number')
        axs[iv].set_title('View '+str(iv)+'/'+cf.view_name[iv])


    plt.tight_layout()


    run_nb = str(dc.evt_list[-1].run_nb)
    evt_nb = str(dc.evt_list[-1].trigger_nb)
    elec   = dc.evt_list[-1].elec

    if(option):
        option = "_"+option
    else:
        option = ""


    plt.savefig(cf.plot_path+'/ped_raw_rms_vch'+option+'_'+elec+'_run_'+run_nb+'_evt_'+evt_nb+'.png')

    if(to_be_shown):
        plt.show()

    plt.close()



def plot_noise_vch(noise_type, vmin=0,vmax=10, option=None, to_be_shown=False):
    if(noise_type==""):
        print("plot_noise_vch needs to know which noise to show (raw or filt)")
        return

    if(noise_type is not 'raw' and noise_type is not 'filt'):
        print("plot_noise_vch needs to know which noise to show (raw or filt) : ", noise_type, ' is not recognized')
        

    fig = plt.figure(figsize=(12,4))
    gs = gridspec.GridSpec(nrows=1, 
                           ncols=3)

    axs  = [fig.add_subplot(gs[0,x]) for x in range(3)]

    rms = np.zeros((cf.n_view, max(cf.view_nchan)))

    for i in range(cf.n_tot_channels):
        view, chan = dc.chmap[i].view, dc.chmap[i].vchan
        if(view >= cf.n_view or view < 0):
            continue
        if(noise_type=='filt'):
            rms[view, chan] = dc.evt_list[-1].noise_filt.ped_rms[i]
        else:
            rms[view, chan] = dc.evt_list[-1].noise_raw.ped_rms[i]

    for iv in range(cf.n_view):

        axs[iv].scatter(np.linspace(0, cf.view_nchan[iv], cf.view_nchan[iv], endpoint=False),rms[iv,:cf.view_nchan[iv]],s=2)

        axs[iv].set_ylabel('RMS Ped [ADC]')
        axs[iv].set_xlabel('Channel Number')
        axs[iv].set_title('View '+str(iv)+'/'+cf.view_name[iv])
        axs[iv].set_xlim(0,cf.view_nchan[iv])
        axs[iv].set_ylim(vmin,vmax)
    plt.tight_layout()


    run_nb = str(dc.evt_list[-1].run_nb)
    evt_nb = str(dc.evt_list[-1].trigger_nb)
    elec   = dc.evt_list[-1].elec

    if(option):
        option = "_"+option
    else:
        option = ""


    plt.savefig(cf.plot_path+'/ped_'+noise_type+'_rms_vch'+option+'_'+elec+'_run_'+run_nb+'_evt_'+evt_nb+'.png')

    if(to_be_shown):
        plt.show()

    plt.close()



def plot_FFT(ps, option=None, to_be_shown=False):
    fig = plt.figure(figsize=(6,6))
    gs = gridspec.GridSpec(nrows=2, 
                           ncols=1,
                           height_ratios=[1, 15])
    
    ax_col = fig.add_subplot(gs[0,:])
    ax_raw = fig.add_subplot(gs[1, :])
    ax_raw.imshow(ps.transpose(), 
                  origin = 'lower', 
                  aspect = 'auto', 
                  interpolation='none',
                  cmap   = cmap_fft, extent=[0, cf.n_tot_channels, 0., 1./cf.sampling/2.], norm=LogNorm(vmin=1e-1, vmax=500))


    ax_raw.set_ylabel('Frequencies [MHz]')
    ax_raw.set_xlabel('DAQ Channels')
    ax_col.set_title('FFT Amplitude')

    cb = fig.colorbar(ax_raw.images[-1], cax=ax_col, orientation='horizontal')
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')

    plt.tight_layout()


    run_nb = str(dc.evt_list[-1].run_nb)
    evt_nb = str(dc.evt_list[-1].trigger_nb)
    elec   = dc.evt_list[-1].elec

    if(option):
        option = "_"+option
    else:
        option = ""


    plt.savefig(cf.plot_path+'/fft_daqch'+option+'_'+elec+'_run_'+run_nb+'_evt_'+evt_nb+'.png')

    if(to_be_shown):
        plt.show()

    plt.close()


def plot_correlation(option=None, to_be_shown=False):
    fig = plt.figure(figsize=(8,8))
    gs = gridspec.GridSpec(nrows=2, 
                           ncols=1,
                           height_ratios=[1, 15])
    
    ax_col  = fig.add_subplot(gs[0,:])
    ax_corr = fig.add_subplot(gs[1, :])

    corr = np.corrcoef(dc.data_daq)
    print(corr.shape)

    ax_corr.imshow(corr,
                   origin = 'lower', 
                   aspect = 'auto', 
                   interpolation='none',
                   cmap   = cmap_corr,
                   extent=[0, cf.n_tot_channels, 0., cf.n_tot_channels],
                   vmin=-1, vmax=1)

    ax_corr.set_ylabel('DAQ Channels')
    ax_corr.set_xlabel('DAQ Channels')
    ax_col.set_title('Correlation Coefficient')

    cb = fig.colorbar(ax_corr.images[-1], cax=ax_col, orientation='horizontal')
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')

    plt.tight_layout()

    run_nb = str(dc.evt_list[-1].run_nb)
    evt_nb = str(dc.evt_list[-1].trigger_nb)
    elec   = dc.evt_list[-1].elec

    if(option):
        option = "_"+option
    else:
        option = ""


    plt.savefig(cf.plot_path+'/correlation_daqch'+option+'_'+elec+'_run_'+run_nb+'_evt_'+evt_nb+'.png')

    if(to_be_shown):
        plt.show()

    plt.close()
