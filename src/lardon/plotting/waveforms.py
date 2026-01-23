import lardon.config as cf
import lardon.data_containers as dc
import lardon.channel_mapping as chmap

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import collections  as mc
from matplotlib.legend_handler import HandlerTuple
import matplotlib.patches as patches

import itertools as itr
import math
import colorcet as cc
from lardon.plotting.save_plot import *

def draw_current_waveform(daqch, view, ch, ax=None, **kwargs):    
    
    ax = plt.gca() if ax is None else ax
    if(daqch >= 0):
        ax.step(np.linspace(0,cf.n_sample[cf.imod]-1,cf.n_sample[cf.imod]), dc.data_daq[daqch, :], where='mid',**kwargs)
    else:
        ax.step(np.linspace(0,cf.n_sample[cf.imod]-1,cf.n_sample[cf.imod]), dc.data[view, ch, :], where='mid',**kwargs)
    return ax



def plot_wvf_current_vch(vch_list, adc_min=-1, adc_max=-1, tmin=0, tmax=cf.n_sample[cf.imod], option=None, to_be_shown=False):
    chmap.arange_in_view_channels()

    n_wvf = len(vch_list)

    fig = plt.figure(figsize=(12, 3*n_wvf))
    gs = gridspec.GridSpec(nrows=n_wvf, ncols=1)
    ax = [fig.add_subplot(gs[0,0])]
    ax.extend([fig.add_subplot(gs[i,0], sharex=ax[0]) for i in range(1,n_wvf)])

    
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
    
    for a in ax:
        a.axhline(0, lw=1, ls="dashed", c='r')
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)

        
    save_with_details(fig, option, 'waveforms')


    if(to_be_shown):
        plt.show()

    plt.close()


def plot_wvf_current_daqch(daqch_list, adc_min=-1, adc_max=-1, option=None, to_be_shown=False, tmin=0, tmax=cf.n_sample[cf.imod]):
    """ daqch_list should be a list of daqchs """
    n_wvf = len(daqch_list)

    fig = plt.figure(figsize=(12, 3*n_wvf))
    gs = gridspec.GridSpec(nrows=n_wvf, ncols=1)
    ax = [fig.add_subplot(gs[i,0]) for i in range(n_wvf)]

    
    for i in range(n_wvf):
        daqch = daqch_list[i]
        view, ch = dc.chmap[daqch].view, dc.chmap[daqch].vchan
        legend = "DAQ ch"+str(daqch)+" - V"+str(view)+" Ch"+str(ch)

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

    for a in ax:
        a.axhline(0, lw=1, ls="dashed", c='r')


    ax[-1].set_xlabel('Time')
    

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)

    save_with_details(fig, option, 'waveforms')
    #plt.savefig(cf.plot_path+'/waveforms'+option+'_'+elec+'_run_'+run_nb+'_evt_'+evt_nb+'.png')

    if(to_be_shown):
        plt.show()

    plt.close()


def plot_track_wvf_vch(vch_list, adc_min=-1, adc_max=-1, tmin=0, tmax=cf.n_sample[cf.imod], option=None, to_be_shown=False):
    chmap.arange_in_view_channels()
    #print(vch_list)

    
    nview = len(vch_list)
    fig = plt.figure(figsize=(12, 3*nview))
    gs = gridspec.GridSpec(nrows=nview, ncols=1)
    ax = [fig.add_subplot(gs[0,0])]
    ax.extend([fig.add_subplot(gs[i,0], sharex=ax[0]) for i in range(1,nview)])

    
    for i in range(nview):
        n_wvf = len(vch_list[i])
        colors = plt.cm.ocean(np.linspace(0.2,.8,n_wvf))
        for k in range(n_wvf):
            view, ch = vch_list[i][k]
            legend = "v"+str(view)+" ch"+str(ch)
            ax[i] = draw_current_waveform(-1, view, ch, ax=ax[i], label=legend, c=colors[k])
        ax[i].set_ylabel('ADC')
        #ax[i].legend(loc='upper right')
        ax[i].set_xlim([tmin, tmax])
        if(adc_min > -1):
            ax[i].set_ybound(lower=adc_min)
        if(adc_max > -1):
            ax[i].set_ybound(upper=adc_max)
            
    for a in ax[:-1]:
        a.tick_params(labelbottom=False)
    ax[-1].set_xlabel('Time')
    

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)

    save_with_details(fig, option, 'waveforms_track')


    if(to_be_shown):
        plt.show()

    plt.close()




def plot_wvf_current_hits_roi_vch(vch_list, adc_min=-1, adc_max=-1, tmin=0, tmax=cf.n_sample[cf.imod], option=None, to_be_shown=False):

    #chmap.arange_in_view_channels()
    n_wvf = len(vch_list)

    fig = plt.figure(figsize=(12, 3*n_wvf))#(12, 3*n_wvf))
    gs = gridspec.GridSpec(nrows=n_wvf, ncols=1)
    ax = [fig.add_subplot(gs[0,0])]
    ax.extend([fig.add_subplot(gs[i,0], sharex=ax[0]) for i in range(1,n_wvf)])

    daq_start = cf.module_daqch_start[cf.imod]
    print('daq start is : ', daq_start)
    for i in range(n_wvf):
        view, ch = vch_list[i]
        
        """ ugly but it works """
        for k in dc.chmap:
            if(k.view == view and k.vchan == ch and k.module == cf.imod):
                daq_ch = k.daqch
                glob_ch = k.globch#k.daqch
                break

        legend = "v"+str(view)+" ch"+str(ch)
        ax[i] = draw_current_waveform(daq_ch-daq_start, -1, -1, ax=ax[i], label=legend, c='k')
        ax[i].set_ylabel('ADC')
        #ax[i].legend(loc='upper right')

        if(adc_min > -1):
            ax[i].set_ybound(lower=adc_min)
        if(adc_max > -1):
            ax[i].set_ybound(upper=adc_max)



        ymin, ymax = ax[i].get_ylim()
        ped_mean = dc.evt_list[-1].noise_filt[cf.imod].ped_mean[daq_ch-daq_start]
        ped_rms = dc.evt_list[-1].noise_filt[cf.imod].ped_rms[daq_ch-daq_start]
        print('--> ', view, ch, ' daq ', daq_ch, ' ped = ', ped_mean, ' rms = ', ped_rms)
        print('wvf min ',min(dc.data_daq[daq_ch-daq_start]), ' max ', max(dc.data_daq[daq_ch-daq_start]))

              
        """ draw rois """
        ROI = np.r_['-1',0,np.array(~dc.mask_daq[daq_ch-daq_start], dtype=int),0]
        d = np.diff(ROI)

        """ a change from false to true in difference is = 1 """
        start = np.where(d==1)[0]
        """ a change from true to false in difference is = -1 """
        end   = np.where(d==-1)[0]

        for ir in range(len(start)):

            tdc_start = start[ir]
            tdc_stop = end[ir]            
            dt = tdc_stop-tdc_start
            dy = ymax-ymin
            r = patches.Rectangle((tdc_start,ymin),dt,dy,linewidth=.5,edgecolor='none',facecolor='lightgray',zorder=-100)
            ax[i].add_patch(r)

        for ih in dc.hits_list:
            if(ih.daq_channel == daq_ch):
                #ih.dump()
                if(ih.signal != cf.view_type[ih.view]):
                    c='blue'
                else:
                    c='r'
                t_start, t_stop = ih.start,ih.stop
                ax[i].fill_between(np.linspace(t_start,t_stop,t_stop-t_start+1), dc.data_daq[daq_ch-daq_start, t_start:t_stop+1], ped_mean, step='mid',color=c, alpha=0.25,zorder=200)
                ax[i].step(np.linspace(t_start,t_stop,t_stop-t_start+1), dc.data_daq[daq_ch-daq_start, t_start:t_stop+1], linewidth=1, c=c,zorder=250,where='mid')

                ax[i].axvline(t_start, ls='dashed',c=c,lw=.5,zorder=300)
                ax[i].axvline(t_stop, ls='dotted',c=c,lw=.5,zorder=300)


        if(cf.view_type[view] == "Collection"):
            for j in dc.reco['mask']['coll']['low_thr']:#[2, 3.5]:
                ax[i].axhline(ped_mean+j*ped_rms, ls='dashdot',c='orange',lw=.5)

            for j in dc.reco['mask']['coll']['high_thr']:#[2, 3.5]:
                ax[i].axhline(ped_mean+j*ped_rms, ls='dotted',c='orange',lw=.5)
                
        if(cf.view_type[view] == "Induction"):
            for j in dc.reco['mask']['ind']['pos']['low_thr']:#[2, 3.5]:
                ax[i].axhline(ped_mean+j*ped_rms, ls='dashdot',c='orange',lw=.5)
            for j in dc.reco['mask']['ind']['neg']['low_thr']:#[2, 3.5]:
                ax[i].axhline(ped_mean+j*ped_rms, ls='dashdot',c='orange',lw=.5)
                
            for j in dc.reco['mask']['ind']['pos']['high_thr']:#[2, 3.5]:
                ax[i].axhline(ped_mean+j*ped_rms, ls='dotted',c='orange',lw=.5)
            for j in dc.reco['mask']['ind']['neg']['high_thr']:#[2, 3.5]:
                ax[i].axhline(ped_mean+j*ped_rms, ls='dotted',c='orange',lw=.5)



        ax[i].axhline(ped_mean, ls='solid',c='orange',lw=1)
        #print("v", view," ch", ch, " ped = ", ped_mean, " rms = ", ped_rms)
        ax[i].set_xlim([tmin, tmax])
        ax[i].set_ylim([ymin, ymax])
            
    for a in ax[:-1]:
        a.tick_params(labelbottom=False)
    ax[-1].set_xlabel('Time')
    

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)

    save_with_details(fig, option, 'waveforms_roi_hits')

    if(to_be_shown):
        plt.show()

    plt.close()


def plot_wvf_current_hits_roi_daqch(daqch_list, adc_min=-1, adc_max=-1, tmin=0, tmax=cf.n_sample[cf.imod], option=None, to_be_shown=False):
    vch_list = []
    for i in daqch_list:
        view, ch = dc.cmap[i].view, dc.cmap[i].vchan
        vch_list.append( (view,ch) )

    return plot_wvf_current_hits_roi_vch(vch_list, adc_min, adc_max, tmin, tmax, option, to_be_shown)



def plot_wvf_diff_vch(vch_list, adc_min=-1, adc_max=-1, tmin=0, tmax=cf.n_sample[cf.imod], option=None, to_be_shown=False):
    n_wvf = len(vch_list)

    fig = plt.figure(figsize=(12, 3*n_wvf))
    gs = gridspec.GridSpec(nrows=n_wvf, ncols=1)
    ax = [fig.add_subplot(gs[0,0])]
    ax.extend([fig.add_subplot(gs[i,0], sharex=ax[0]) for i in range(1,n_wvf)])

    
    for i in range(n_wvf):
        view, ch = vch_list[i]

        """ ugly but it works """
        for k in dc.chmap:
            if(k.view == view and k.vchan == ch):
                daq_ch = k.daqch
                break        

        legend = "v"+str(view)+" ch"+str(ch)
        ax[i] = draw_current_waveform(daq_ch, -1, -1, ax=ax[i], label=legend, c='gray',lw=.5)
        ped_mean = dc.evt_list[-1].noise_filt.ped_mean[daq_ch]
        wvf = np.r_['-1',ped_mean, dc.data_daq[daq_ch,:]]
        dwvf = np.diff(wvf)
        ax[i].step(np.linspace(0,cf.n_sample[cf.imod]-1,cf.n_sample[cf.imod]), dwvf, where='mid', c='k',lw=1)

        ax[i].set_ylabel('ADC')
        ax[i].legend(loc='upper right')

        if(adc_min > -1):
            ax[i].set_ybound(lower=adc_min)
        if(adc_max > -1):
            ax[i].set_ybound(upper=adc_max)

        ax[i].set_xlim([tmin, tmax])

            
    for a in ax[:-1]:
        a.tick_params(labelbottom=False)
    ax[-1].set_xlabel('Time')
    

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)

    save_with_details(fig, option, 'waveforms_diff')
    if(to_be_shown):
        plt.show()

    plt.close()



def draw_data(data, ax=None, **kwargs):
    
    ax = plt.gca() if ax is None else ax    
    ax.step(np.linspace(0,cf.n_sample[cf.imod]-1,cf.n_sample[cf.imod]), data, where='mid',**kwargs)
    return ax
    

def plot_wvf_evo(data, title="", legends=[], colors=['k','tab:cyan'], adc_min=-1, adc_max=-1, tmin=0, tmax=cf.n_sample[cf.imod], option=None, to_be_shown=False):

    fig = plt.figure(figsize=(12, 3))
    ax = plt.gca()

    for d in range(len(data)):
        ax = draw_data(data[d], ax=ax, c=colors[d], label=legends[d])
        
    ax.set_xlabel('Time')
    ax.set_ylabel('ADC')
    ax.legend(loc='upper right')

    if(adc_min > -1):
        ax.set_ybound(lower=adc_min)
    if(adc_max > -1):
        ax.set_ybound(upper=adc_max)

    ax.set_xlim([tmin, tmax])
    
    ax.set_title(title)
    

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    save_with_details(fig, option, 'waveforms_evo')
    if(to_be_shown):
        plt.show()

    plt.close()



def plot_cumsum_wvf(vch_list, adc_min=-1, adc_max=-1, tmin=0, tmax=cf.n_sample[cf.imod], option=None, to_be_shown=False):
    n_wvf = len(vch_list)

    fig = plt.figure(figsize=(7, 3*n_wvf))#(12, 3*n_wvf))
    gs = gridspec.GridSpec(nrows=n_wvf, ncols=1)
    ax = [fig.add_subplot(gs[0,0])]
    ax.extend([fig.add_subplot(gs[i,0], sharex=ax[0]) for i in range(1,n_wvf)])
    axy = []
    
    for i in range(n_wvf):
        view, ch = vch_list[i]

        """ ugly but it works """
        for k in dc.chmap:
            if(k.view == view and k.vchan == ch):
                daq_ch = k.daqch
                break        

        legend = "v"+str(view)+" ch"+str(ch)
        ax[i] = draw_current_waveform(daq_ch, -1, -1, ax=ax[i], label=legend, c='k',lw=1)
        axy.append(ax[i].twinx())
        csum = np.cumsum(dc.data_daq[daq_ch])
        """
        filt_low = np.zeros(cf.n_sample[cf.imod])
        filt_low += -0.999
        filt_low = np.exp(filt_low)
        print(filt_low[:5])
        csum_filt_low = np.cumsum(dc.data_daq[daq_ch]*filt_low)
        """
        
        #axy[i] = draw_current_waveform(csum, -1, -1, ax=axy[i], c='r',lw=.5)
        axy[i].step(np.linspace(0,cf.n_sample[cf.imod],cf.n_sample[cf.imod]), csum, where='mid', c='r',lw=1)
        #axy[i].step(np.linspace(0,cf.n_sample[cf.imod],cf.n_sample[cf.imod]), csum_filt_low, where='mid', c='r',lw=.5)
        ax[i].set_ylabel('ADC')
        axy[i].set_ylabel('ADC Cum Sum', color='r')
        axy[i].tick_params(axis='y', labelcolor='r')
        ax[i].legend(loc='upper right')

        if(adc_min > -1):
            ax[i].set_ybound(lower=adc_min)
        if(adc_max > -1):
            ax[i].set_ybound(upper=adc_max)

        ax[i].set_xlim([tmin, tmax])

            
    for a in ax[:-1]:
        a.tick_params(labelbottom=False)
    ax[-1].set_xlabel('Time')
    

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)

    save_with_details(fig, option, 'waveforms_cumsum')
    if(to_be_shown):
        plt.show()

    plt.close()
