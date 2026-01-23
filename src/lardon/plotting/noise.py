import lardon.config as cf
import lardon.data_containers as dc

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

cmap_fft = cc.cm.CET_CBL2_r
cmap_corr = cc.cm.CET_D9
cmap_ed = cc.cm.linear_tritanopic_krjcw_5_95_c24_r

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


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

    ch = np.linspace(0, cf.module_nchan[cf.imod], cf.module_nchan[cf.imod], endpoint=False)
    if(noise_type=='filt'):
        ax_std.scatter(ch, dc.evt_list[-1].noise_filt[cf.imod].ped_rms,s=2)
    else:
        ax_std.scatter(ch, dc.evt_list[-1].noise_raw[cf.imod].ped_rms,s=2)

    ax_std.set_ylabel('RMS Ped [ADC]')
    ax_std.set_xlabel('DAQ Channel Number')
    ax_std.set_ylim(vrange[0],vrange[1])

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    save_with_details(fig, option, 'ped_'+noise_type+'_rms_daqch')
    #plt.savefig(cf.plot_path+'/ped_'+noise_type+'_rms_daqch'+option+'_'+elec+'_run_'+run_nb+'_evt_'+evt_nb+'.png')

    if(to_be_shown):
        plt.show()

    plt.close()



def plot_noise_globch(noise_type, vrange=[0,10], option=None, to_be_shown=False, capa=False):
    if(noise_type==""):
        print("plot_noise_vch needs to know which noise to show (raw or filt)")
        return

    if(noise_type != 'raw' and noise_type != 'filt'):
        print("plot_noise_vch needs to know which noise to show (raw or filt) : ", noise_type, ' is not recognized')
        


    fig = plt.figure(figsize=(12,4))
    gs = gridspec.GridSpec(nrows=2, 
                           ncols=1, wspace=0.08)
    

    ax_std  = fig.add_subplot(111)

    rms = np.zeros(cf.module_nchan[cf.imod])
    daq_start = cf.module_daqch_start[cf.imod]
    
    for i in range(cf.module_nchan[cf.imod]):
        gch = dc.chmap[i+daq_start].globch - daq_start
        if(capa==True):
            ct  = dc.chmap[i+daq_start].tot_capa
        else:
            ct = 1.
        if(gch < 0):
            continue
        else:
            if(noise_type=='raw'):
                rms[gch] = dc.evt_list[-1].noise_raw[cf.imod].ped_rms[i]/ct
            elif(noise_type=='filt'):
                rms[gch] = dc.evt_list[-1].noise_filt[cf.imod].ped_rms[i]/ct

    ch = np.linspace(daq_start, cf.module_nchan[cf.imod]+daq_start, cf.module_nchan[cf.imod], endpoint=False)
    ax_std.scatter(ch, rms,s=2)

    if(capa==True):
        ax_std.set_ylabel('RMS Ped [ADC/pF]')
    else:
        ax_std.set_ylabel('RMS Ped [ADC]')
    ax_std.set_xlabel('Global Channel Number')
    ax_std.set_ylim(vrange[0],vrange[1])
    ax_std.set_xlim(daq_start, cf.module_nchan[cf.imod]+daq_start)

    nprev = daq_start
    for i in range(cf.n_view-1):
        ax_std.axvline(nprev + cf.view_nchan[i], lw=.4,c='k')
        nprev += cf.view_nchan[i]


    plt.tight_layout()


    save_with_details(fig, option, 'ped_'+noise_type+'_rms_globch')



    if(to_be_shown):
        plt.show()

    plt.close()



def plot_noise_vch(noise_type, vrange=[0,10], option=None, to_be_shown=False, calibrated=False):
    if(noise_type==""):
        print("plot_noise_vch needs to know which noise to show (raw or filt)")
        return


    if(noise_type != 'raw' and noise_type != 'filt'):
        print("plot_noise_vch needs to know which noise to show (raw or filt) : ", noise_type, ' is not recognized')
        
    fig = plt.figure(figsize=(10,4))
    gs = gridspec.GridSpec(nrows=1, 
                           ncols=3)

    axs  = [fig.add_subplot(gs[0,x]) for x in range(3)]

    rms = np.zeros((cf.n_view, max(cf.view_nchan)))
    daq_start = cf.module_daqch_start[cf.imod]
    
    for i in range(cf.module_nchan[cf.imod]):
        view, chan = dc.chmap[i+daq_start].view, dc.chmap[i+daq_start].vchan
        calib = 1.
        if(calibrated==True):
            calib = 1e-3*dc.chmap[i+daq_start].gain/1.602e-4
        if(view >= cf.n_view or view < 0):
            continue
        if(noise_type=='filt'):
            rms[view, chan] = dc.evt_list[-1].noise_filt[cf.imod].ped_rms[i]*calib
        else:
            rms[view, chan] = dc.evt_list[-1].noise_raw[cf.imod].ped_rms[i]*calib

    for iv in range(cf.n_view):

        axs[iv].scatter(np.linspace(0, cf.view_nchan[iv], cf.view_nchan[iv], endpoint=False),rms[iv,:cf.view_nchan[iv]],s=2)
        if(calibrated):
            axs[iv].set_ylabel(r'RMS Ped [ke$^-$]')
        else:
            axs[iv].set_ylabel('RMS Ped [ADC]')
        axs[iv].set_xlabel('Channel Number')
        axs[iv].set_title('View '+str(iv)+'/'+cf.view_name[iv])
        axs[iv].set_xlim(0,cf.view_nchan[iv])
        axs[iv].set_ylim(vrange[0],vrange[1])
        axs[iv].axvline(0.5*cf.view_nchan[iv]-0.5, c='k',ls='dotted',lw=.5)
    plt.tight_layout()


    unit = 'ADC'
    if(calibrated):
        unit='ke'


    save_with_details(fig, option, 'ped_'+noise_type+'_'+unit+'_rms_vch')
    #plt.savefig(cf.plot_path+'/ped_'+noise_type+'_'+unit+'_rms_vch'+option+'_'+elec+'_run_'+run_nb+'_evt_'+evt_nb+'.png')

    if(to_be_shown):
        plt.show()

    plt.close()



def plot_FFT_daqch(ps, option=None, to_be_shown=False):
    fig = plt.figure(figsize=(9,6))
    gs = gridspec.GridSpec(nrows=2, 
                           ncols=2,
                           height_ratios=[1, 15], width_ratios=[5,1])
    
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
                  cmap   = cmap_fft, extent=[0, cf.n_tot_channels, 0., cf.sampling[cf.imod]/2.], norm=LogNorm(vmin=1e-1, vmax=5))
    

    ax_2D.set_ylabel('Frequencies [MHz]')
    ax_2D.set_xlabel('DAQ Channels')
    ax_col.set_title('FFT Amplitude')

    cb = fig.colorbar(ax_2D.images[-1], cax=ax_col, orientation='horizontal')
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')


    freq = np.linspace(0., cf.sampling[cf.imod]/2, int(cf.n_sample[cf.imod]/2) + 1)

    ax_proj.plot(np.mean(ps, axis=0), freq,c='k')
    ax_proj.yaxis.tick_right()
    ax_proj.yaxis.set_label_position("right")
    plt.tight_layout()


    save_with_details(fig, option, 'fft_daqch')
    #plt.savefig(cf.plot_path+'/fft_daqch'+option+'_'+elec+'_run_'+run_nb+'_evt_'+evt_nb+'.png')

    if(to_be_shown):
        plt.show()

    plt.close()





def plot_FFT_vch(ps, option=None, to_be_shown=False):
    print("SHAPE OF PS ",ps.shape)
    fig = plt.figure(figsize=(12,6))
    gs = gridspec.GridSpec(nrows=2, 
                           ncols=cf.n_view,
                           height_ratios=[1, 15])
    
    ax_col = fig.add_subplot(gs[0, :])#[fig.add_subplot(gs[0, i]) for i in range(3)]
    #axs_v = [fig.add_subplot(gs[1, i]) for i in range(3)]


    gsgs = [gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=2, width_ratios=[3,1],subplot_spec=gs[1,i],wspace=0.05) for i in range(3)]

    axs_2D = [fig.add_subplot(gsgs[i][0, 0]) for i in range(3)]
    axs_proj = [fig.add_subplot(gsgs[i][0, 1],sharey=axs_2D[i]) for i in range(3)]


    freq = np.linspace(0., cf.sampling[cf.imod]/2, int(cf.n_sample[cf.imod]/2) + 1)


    ps_v = np.zeros((cf.n_view, max(cf.view_nchan), int(cf.n_sample[cf.imod]/2)+1))

    for i in range(cf.n_tot_channels):
        view, chan = dc.chmap[i].view, dc.chmap[i].vchan
        if(view >= cf.n_view or view < 0):
            continue
        
        ps_v[view, chan] = ps[i]
            

    proj_max=[0.3, 0.2, 0.1]

    freq = np.linspace(0, cf.sampling[cf.imod]/2., int(cf.n_sample[cf.imod]/2) + 1)

    """define gaussian low pass filter"""
    gauss_cut = np.where(freq < 0.6, 1., gaussian(freq, 0.6, 0.02))


    for i in range(3):
        if(cf.view_type[i] == 'Induction'):
            vname = "Ind."
        else:
            vname = "Coll."

        axs_2D[i].imshow(ps_v[i,:cf.view_nchan[i]].transpose(), 
                         origin = 'lower', 
                         aspect = 'auto', 
                         interpolation='none',
                         cmap   = cmap_fft,
                         extent=[0, cf.view_nchan[i], 0., cf.sampling[cf.imod]/2.], 
                         norm=LogNorm(vmin=5e-2, vmax=8e-1))
    
        axs_2D[i].set_ylim(0., cf.sampling[cf.imod]/2.)

        axs_2D[i].set_xlabel('View Channel')
        axs_2D[i].set_title('View '+str(i)+'/'+cf.view_name[i]+' ('+vname+')')

        if(i==0):
            ax_col.set_title('FFT Amplitude')

            cb = fig.colorbar(axs_2D[i].images[-1], cax=ax_col, orientation='horizontal')
            cb.ax.xaxis.set_ticks_position('top')
            cb.ax.xaxis.set_label_position('top')

        axs_proj[i].plot(np.mean(ps_v[i,:cf.view_nchan[i]], axis=0), freq,c='k', zorder=100)
        vmin, vmax = axs_proj[i].get_xlim()
        axs_proj[i].plot(gauss_cut*vmax, freq,c='r', zorder=-100)
        #if(i==0):
            #axs_proj[i].set_xlim(0, 0.5)#0.4)
        #axs_proj[i].set_xlim(0, proj_max[i])#0.5)#0.4)
        #axs_proj[i].set_ylim(0., 1.25)
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

    save_with_details(fig, option, 'fft_vch_mod_'+cf.imod)



    #plt.savefig(cf.plot_path+'/fft_vch'+option+'_'+elec+'_run_'+run_nb+'_evt_'+evt_nb+'.png')

    if(to_be_shown):
        plt.show()

    plt.close()



def plot_correlation_globch(option=None, to_be_shown=False):
    data_globch = np.zeros(dc.data_daq.shape)
    daqch_start = cf.module_daqch_start[cf.imod]
    for i in range(cf.module_nchan[cf.imod]):
        gch = dc.chmap[i+daqch_start].globch
        if(gch < 0):
            continue
        data_globch[gch-3072*cf.imod] = dc.data_daq[i]

    #return plot_correlation(np.corrcoef(data_globch),"glob",option,to_be_shown)
    return np.corrcoef(data_globch)

def plot_correlation_daqch(option=None, to_be_shown=False):
    return plot_correlation(np.corrcoef(dc.data_daq),"daq",option,to_be_shown)



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
                   extent=[0, cf.module_nchan[cf.imod], 0., cf.module_nchan[cf.imod]],
                   vmin=-0.2, vmax=0.2)#vmin=-1, vmax=1)

    if(corr_type=='daq'):
        title = 'DAQ Channels'
    elif(corr_type=='glob'):
        title = 'Global Channels'
    else:
        title = 'Channel'
        print('Warning : ', corr_type, ' is not a recognized name !')

    ax_corr.set_ylabel(title)
    ax_corr.set_xlabel(title)

    
    jump = 64 if cf.elec[cf.imod] == "top" else 128
    
    for i in range(0,cf.module_nchan[cf.imod],jump):
        ax_corr.axvline(i, ls=':',lw=.2,c='k')
        ax_corr.axhline(i, ls=':',lw=.2,c='k')

    if(corr_type=='glob'):
        nprev = 0
        for i in range(cf.n_view-1):
            ax_corr.axvline(nprev + cf.view_nchan[i], lw=.4,c='k')
            ax_corr.axhline(nprev + cf.view_nchan[i], lw=.4,c='k')
            nprev += cf.view_nchan[i]
    elif(corr_type=='daq'):
        jump = 640 if dc.evt_list[-1].elec == "top" else 128
        for i in range(0,cf.module_nchan[cf.imod],jump):
            ax_corr.axvline(i, ls='--',lw=.4,c='k')
            ax_corr.axhline(i, ls='--',lw=.4,c='k')
            


    ax_col.set_title('Correlation Coefficient')

    cb = fig.colorbar(ax_corr.images[-1], cax=ax_col, orientation='horizontal')
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')

    plt.tight_layout()
    save_with_details(fig, option, 'correlation_'+corr_type+'ch_mod_'+str(cf.imod))
    

    if(to_be_shown):
        plt.show()

    plt.close()
    
    #return corr


def plot_sticky_finder_daqch(option='', to_be_shown=False):

    fig = plt.figure(figsize=(9,6))

    gs = gridspec.GridSpec(nrows=2, 
                           ncols=1,
                           height_ratios=[1, 15])

    ax_col = fig.add_subplot(gs[0,:])
    ax     = fig.add_subplot(gs[1, :])


    h = np.apply_along_axis(lambda a: np.histogram(a, bins=16384,range=(-0.5,16383.5))[0], 1, dc.data_daq)


    ax.imshow(h.transpose(), 
              origin = 'lower', 
              aspect = 'auto', 
              interpolation='none',
              cmap   = cmap_ed,
              extent = (0,cf.n_tot_channels,-.5,16383.5),#4095.5),
              vmin = 0,
              vmax = 100)

    jump = 64 if dc.evt_list[-1].elec == "top" else 128
    for i in range(0,cf.n_tot_channels,jump):
        ax.axvline(i, ls=':',lw=.4,c='k')
    if(dc.evt_list[-1].elec == "bot"):
         asic = 16
         for i in range(0,cf.n_tot_channels,asic):
             ax.axvline(i, ls=':',lw=.1,c='k')
    
    ax.set_xlabel('DAQ Channel Number')
    
    ax.set_ylabel('ADC')
    ax.set_ylim(-5,16400)#4100)
    ax_col.set_title('ADC appearance frequency')
    cb = fig.colorbar(ax.images[-1], cax=ax_col, orientation='horizontal')
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')

    plt.tight_layout()
    save_with_details(fig, option, 'sticky_daqch')


    #plt.savefig(cf.plot_path+'/sticky_daqch'+option+'_'+elec+'_run_'+run_nb+'_evt_'+evt_nb+'.png', dpi=200)

    '''
    channel = 1550
    htest, etest = np.histogram(dc.data_daq[channel], bins=4096,range=(-0.5,4095.5))
    xbins = etest[:-1]# +0.5
    print(xbins.shape)
    print(xbins[:5])
    figtest = plt.figure(2, figsize=(9,3))
    axtest = figtest.add_subplot(111)
    axtest.step(xbins,htest)

    figtest.savefig(cf.plot_path+'/sticky_ch'+str(channel)+option+'_'+elec+'_run_'+run_nb+'_evt_'+evt_nb+'.png')
    '''

    if(to_be_shown):
        plt.show()

    plt.close()
    
