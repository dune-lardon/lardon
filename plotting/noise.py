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



def plot_noise_daqch(noise_type, vrange=[0,10], option=None, to_be_shown=False):
    if(noise_type==""):
        print("plot_noise_vch needs to know which noise to show (raw or filt)")
        return

    if(noise_type != 'raw' and noise_type != 'filt'):
        print("plot_noise_vch needs to know which noise to show (raw or filt) : ", noise_type, ' is not recognized')
        

    fig = plt.figure(figsize=(12,4))
    gs = gridspec.GridSpec(nrows=2, 
                           ncols=1, wspace=0.08)
    

    ax_std  = fig.add_subplot(111)

    ch = np.linspace(0, cf.n_tot_channels, cf.n_tot_channels, endpoint=False)
    if(noise_type=='filt'):
        ax_std.scatter(ch, dc.evt_list[-1].noise_filt.ped_rms,s=2)
    else:
        ax_std.scatter(ch, dc.evt_list[-1].noise_raw.ped_rms,s=2)

    ax_std.set_ylabel('RMS Ped [ADC]')
    ax_std.set_xlabel('DAQ Channel Number')
    ax_std.set_ylim(vrange[0],vrange[1])

    plt.tight_layout()


    run_nb = str(dc.evt_list[-1].run_nb)
    evt_nb = str(dc.evt_list[-1].trigger_nb)
    elec   = dc.evt_list[-1].elec

    if(option):
        option = "_"+option
    else:
        option = ""


    plt.savefig(cf.plot_path+'/ped_'+noise_type+'_rms_daqch'+option+'_'+elec+'_run_'+run_nb+'_evt_'+evt_nb+'.png')

    if(to_be_shown):
        plt.show()

    plt.close()



def plot_noise_globch(noise_type, vrange=[0,10], option=None, to_be_shown=False):
    if(noise_type==""):
        print("plot_noise_vch needs to know which noise to show (raw or filt)")
        return

    if(noise_type != 'raw' and noise_type != 'filt'):
        print("plot_noise_vch needs to know which noise to show (raw or filt) : ", noise_type, ' is not recognized')
        


    fig = plt.figure(figsize=(12,4))
    gs = gridspec.GridSpec(nrows=2, 
                           ncols=1, wspace=0.08)
    

    ax_std  = fig.add_subplot(111)

    rms = np.zeros(cf.n_tot_channels)

    for i in range(cf.n_tot_channels):
        gch = dc.chmap[i].globch
        if(gch < 0):
            continue
        else:
            if(noise_type=='raw'):
                rms[gch] = dc.evt_list[-1].noise_raw.ped_rms[i]
            elif(noise_type=='filt'):
                rms[gch] = dc.evt_list[-1].noise_filt.ped_rms[i]

    ch = np.linspace(0, cf.n_tot_channels, cf.n_tot_channels, endpoint=False)
    ax_std.scatter(ch, rms,s=2)

    ax_std.set_ylabel('RMS Ped [ADC]')
    ax_std.set_xlabel('DAQ Channel Number')
    ax_std.set_ylim(vrange[0],vrange[1])

    plt.tight_layout()


    run_nb = str(dc.evt_list[-1].run_nb)
    evt_nb = str(dc.evt_list[-1].trigger_nb)
    elec   = dc.evt_list[-1].elec

    if(option):
        option = "_"+option
    else:
        option = ""


    plt.savefig(cf.plot_path+'/ped_'+noise_type+'_rms_globch'+option+'_'+elec+'_run_'+run_nb+'_evt_'+evt_nb+'.png')


    if(to_be_shown):
        plt.show()

    plt.close()



def plot_noise_vch(noise_type, vrange=[0,10], option=None, to_be_shown=False):
    if(noise_type==""):
        print("plot_noise_vch needs to know which noise to show (raw or filt)")
        return


    if(noise_type != 'raw' and noise_type != 'filt'):
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
        axs[iv].set_ylim(vrange[0],vrange[1])
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



def plot_FFT_daqch(ps, option=None, to_be_shown=False):
    fig = plt.figure(figsize=(9,6))
    gs = gridspec.GridSpec(nrows=2, 
                           ncols=2,
                           height_ratios=[1, 15], width_ratios=[2,1])
    
    ax_col = fig.add_subplot(gs[0,0])
    ax_2D = fig.add_subplot(gs[1, 0])
    ax_proj = fig.add_subplot(gs[1, 1], xticklabels=[], sharey = ax_2D)
    ax_proj.yaxis.tick_right()
    ax_proj.yaxis.set_label_position("right")
    ax_proj.set_ylabel('Frequencies [MHz]')
    ax_2D.imshow(ps.transpose(), 
                  origin = 'lower', 
                  aspect = 'auto', 
                  interpolation='none',
                  cmap   = cmap_fft, extent=[0, cf.n_tot_channels, 0., cf.sampling/2.], norm=LogNorm(vmin=1e-1, vmax=5))
    

    ax_2D.set_ylabel('Frequencies [MHz]')
    ax_2D.set_xlabel('DAQ Channels')
    ax_col.set_title('FFT Amplitude')

    cb = fig.colorbar(ax_2D.images[-1], cax=ax_col, orientation='horizontal')
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')


    freq = np.linspace(0., cf.sampling/2, int(cf.n_sample/2) + 1)

    ax_proj.plot(np.mean(ps, axis=0), freq,c='k')
    ax_proj.yaxis.tick_right()
    ax_proj.yaxis.set_label_position("right")
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





def plot_FFT_vch(ps, option=None, to_be_shown=False):
    fig = plt.figure(figsize=(12,6))
    gs = gridspec.GridSpec(nrows=2, 
                           ncols=cf.n_view,
                           height_ratios=[1, 15])
    
    ax_col = fig.add_subplot(gs[0, :])#[fig.add_subplot(gs[0, i]) for i in range(3)]
    #axs_v = [fig.add_subplot(gs[1, i]) for i in range(3)]


    gsgs = [gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=2, width_ratios=[3,1],subplot_spec=gs[1,i],wspace=0.05) for i in range(3)]

    axs_2D = [fig.add_subplot(gsgs[i][0, 0]) for i in range(3)]
    axs_proj = [fig.add_subplot(gsgs[i][0, 1],sharey=axs_2D[i]) for i in range(3)]


    freq = np.linspace(0., cf.sampling/2, int(cf.n_sample/2) + 1)


    ps_v = np.zeros((cf.n_view, max(cf.view_nchan), int(cf.n_sample/2)+1))

    for i in range(cf.n_tot_channels):
        view, chan = dc.chmap[i].view, dc.chmap[i].vchan
        if(view >= cf.n_view or view < 0):
            continue
        
        ps_v[view, chan] = ps[i]
            

    
    for i in range(3):

        axs_2D[i].imshow(ps_v[i,:cf.view_nchan[i]].transpose(), 
                         origin = 'lower', 
                         aspect = 'auto', 
                         interpolation='none',
                         cmap   = cmap_fft,
                         extent=[0, cf.view_nchan[i], 0., cf.sampling/2.], 
                         norm=LogNorm(vmin=1e-1, vmax=5))
    
        

        axs_2D[i].set_xlabel('Channels')
        axs_2D[i].set_title('View '+str(i)+'/'+cf.view_name[i])

        if(i==0):
            ax_col.set_title('FFT Amplitude')

            cb = fig.colorbar(axs_2D[i].images[-1], cax=ax_col, orientation='horizontal')
            cb.ax.xaxis.set_ticks_position('top')
            cb.ax.xaxis.set_label_position('top')


        axs_proj[i].plot(np.mean(ps_v[i,:cf.view_nchan[i]], axis=0), freq,c='k')
        #axs_proj[i].yaxis.tick_right()
        #axs_proj[i].yaxis.set_label_position("right")
        
        if(i==0):
            axs_2D[i].set_ylabel('Frequencies [MHz]')
            axs_proj[i].tick_params(labelleft=False)
        if(i==1):
            axs_2D[i].tick_params(labelleft=False)
            axs_proj[i].tick_params(labelleft=False)
        if(i==2):
            axs_proj[i].yaxis.tick_right()
            axs_proj[i].yaxis.set_label_position("right")
            axs_proj[i].set_ylabel('Frequencies [MHz]')
            axs_2D[i].tick_params(labelleft=False)





    #plt.tight_layout()


    run_nb = str(dc.evt_list[-1].run_nb)
    evt_nb = str(dc.evt_list[-1].trigger_nb)
    elec   = dc.evt_list[-1].elec

    if(option):
        option = "_"+option
    else:
        option = ""


    plt.savefig(cf.plot_path+'/fft_vch'+option+'_'+elec+'_run_'+run_nb+'_evt_'+evt_nb+'.png')

    if(to_be_shown):
        plt.show()

    plt.close()



def plot_correlation_globch(option=None, to_be_shown=False):
    data_globch = np.zeros(dc.data_daq.shape)
    for i in range(cf.n_tot_channels):
        gch = dc.chmap[i].globch
        if(gch < 0):
            continue
        data_globch[gch] = dc.data_daq[i]

    plot_correlation(np.corrcoef(data_globch),"glob",option,to_be_shown)


def plot_correlation_daqch(option=None, to_be_shown=False):
    plot_correlation(np.corrcoef(dc.data_daq),"daq",option,to_be_shown)


def plot_correlation(corr,corr_type,option,to_be_shown):
    fig = plt.figure(figsize=(8,8))
    gs = gridspec.GridSpec(nrows=2, 
                           ncols=1,
                           height_ratios=[1, 15])
    
    ax_col  = fig.add_subplot(gs[0,:])
    ax_corr = fig.add_subplot(gs[1, :])

    ax_corr.imshow(corr,
                   origin = 'lower', 
                   aspect = 'auto', 
                   interpolation='none',
                   cmap   = cmap_corr,
                   extent=[0, cf.n_tot_channels, 0., cf.n_tot_channels],
                   vmin=-1, vmax=1)

    if(corr_type=='daq'):
        title = 'DAQ Channels'
    elif(corr_type=='glob'):
        title = 'Global Channels'
    else:
        title = 'Channel'
        print('Warning : ', corr_type, ' is not a recognized name !')

    ax_corr.set_ylabel(title)
    ax_corr.set_xlabel(title)

    jump = 64 if dc.evt_list[-1].elec == "top" else 128
    for i in range(0,cf.n_tot_channels,jump):
        ax_corr.axvline(i, ls=':',lw=.2,c='k')
        ax_corr.axhline(i, ls=':',lw=.2,c='k')

    if(corr_type=='glob'):
        nprev = 0
        for i in range(cf.n_view-1):
            ax_corr.axvline(nprev + cf.view_nchan[i], lw=.4,c='k')
            ax_corr.axhline(nprev + cf.view_nchan[i], lw=.4,c='k')
            nprev += cf.view_nchan[i]
    elif(corr_type=='daq'):
        jump = 640 if dc.evt_list[-1].elec == "top" else 256
        for i in range(0,cf.n_tot_channels,jump):
            ax_corr.axvline(i, ls='--',lw=.4,c='k')
            ax_corr.axhline(i, ls='--',lw=.4,c='k')
            


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


    plt.savefig(cf.plot_path+'/correlation_'+corr_type+'ch'+option+'_'+elec+'_run_'+run_nb+'_evt_'+evt_nb+'.png')


    if(to_be_shown):
        plt.show()

    plt.close()
