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


def draw_pds_ED( glob_chans, option=None, to_be_shown=False, draw_peak=False, draw_cluster=False, draw_roi=False):
    fig = plt.figure(figsize=(11,4))
    ax = fig.add_subplot(111)
    
    for chan in glob_chans:
        data_type = dc.chmap_pds[chan].data_type
        daqch = dc.chmap_pds[chan].daqch
        daq_offset = cf.pds_daqch_stream_start if data_type ==  "stream" else cf.pds_daqch_trig_start
        
        ts = dc.evt_list[-1].pds_stream_time if data_type ==  "stream" else dc.evt_list[-1].pds_trig_time
        delta_t = (ts - dc.evt_list[-1].event_time)*1e6 # + dc.evt_list[-1].pds_time_offset[chan]
        delta_t_off = (ts - dc.evt_list[-1].event_time)*1e6  + dc.evt_list[-1].pds_time_offset[chan]
        

        n_sample = cf.n_pds_stream_sample if data_type == "stream" else cf.n_pds_trig_sample

        xx = np.linspace(delta_t, delta_t+(n_sample-1)/cf.pds_sampling, n_sample)
        xx_off = np.linspace(delta_t_off, delta_t_off+(n_sample-1)/cf.pds_sampling, n_sample)
        
        data = dc.data_stream_pds if data_type == "stream" else dc.data_trig_pds

        label = dc.chmap_pds[chan].det+' Ch. '+str(dc.chmap_pds[chan].chan)
                
        
        l = ax.plot(xx, data[daqch-daq_offset], label=label)
        ax.plot(xx_off, data[daqch-daq_offset], c=l[0].get_color(), ls="dashed")
        ax.set_xlabel('Time wrt to trigger [mus]')
        ax.set_ylabel('ADC')
    ax.axhline(0, c='k',ls='dotted',lw=0.5)
    nchans = len(glob_chans)

    if(nchans<20):
        if(nchans>4):
            ax.legend(ncols=2)
        else:
            ax.legend()

        
    plt.tight_layout()

    save_with_details(fig, option, 'ED_pds_'+data_type)

    if(to_be_shown):
        plt.show()

    plt.close()


def draw_pds_start_ED( glob_chans, option=None, to_be_shown=False, draw_peak=False, draw_cluster=False, draw_roi=False):
    fig = plt.figure(figsize=(11,3*len(glob_chans)))
    gs = gridspec.GridSpec(nrows=len(glob_chans), ncols=1)
    axs = [fig.add_subplot(gs[i,0]) for i in range(len(glob_chans))]
    for ax in axs[1:]:
        ax.sharex(axs[0])

    i=0
    for chan in glob_chans:
        ax = axs[i]
        i+=1
        data_type = dc.chmap_pds[chan].data_type
        daqch = dc.chmap_pds[chan].daqch
        daq_offset = cf.pds_daqch_stream_start if data_type ==  "stream" else cf.pds_daqch_trig_start
        
        ts = dc.evt_list[-1].pds_stream_time if data_type ==  "stream" else dc.evt_list[-1].pds_trig_time
        delta_t = (ts - dc.evt_list[-1].event_time)*1e6 # + dc.evt_list[-1].pds_time_offset[chan]
        delta_t_off = (ts - dc.evt_list[-1].event_time)*1e6  + dc.evt_list[-1].pds_time_offset[chan]
        

        n_sample = cf.n_pds_stream_sample if data_type == "stream" else cf.n_pds_trig_sample

        xx = np.linspace(delta_t, delta_t+(n_sample-1)/cf.pds_sampling, n_sample)
        xx_off = np.linspace(delta_t_off, delta_t_off+(n_sample-1)/cf.pds_sampling, n_sample)
        
        data = dc.data_stream_pds if data_type == "stream" else dc.data_trig_pds

        label = dc.chmap_pds[chan].det+' Ch. '+str(dc.chmap_pds[chan].chan)
                
        
        l = ax.plot(xx, data[daqch-daq_offset], color='k', label=label)
        ax.plot(xx_off, data[daqch-daq_offset], c='k', ls="dashed")
        ax.set_xlabel('Time wrt to trigger [mus]')
        ax.set_ylabel('ADC')
        ax.legend()

        for p in dc.pds_peak_list:
            if(p.glob_ch == chan):
                ax.axvline(p.timestamp, c='coral', lw=0.4)

        
    plt.tight_layout()

    save_with_details(fig, option, 'ED_pds_'+data_type)

    if(to_be_shown):
        plt.show()

    plt.close()


    
def draw_all_pds_ED( data_type="stream", option=None, to_be_shown=False, draw_peak=False, draw_cluster=False, draw_roi=False):

    n_tot_chan = cf.n_pds_tot_channels
    
    fig = plt.figure(figsize=(11,9))
    if(data_type == "stream"):
        n_chan = cf.n_pds_stream_channels
        n_sample = cf.n_pds_stream_sample
        data = dc.data_stream_pds
        mask = dc.mask_stream_pds
        daq_offset = cf.pds_daqch_stream_start

        nrows = 8
        ncols = 2
        
    elif(data_type == "trigger"):
        n_chan = cf.n_pds_trig_channels
        n_sample = cf.n_pds_trig_sample
        data = dc.data_trig_pds
        mask = dc.mask_trig_pds
        daq_offset = cf.pds_daqch_trig_start

        nrows = 10
        ncols = 4

    else:
        print('data type ', data_type, ' does not exists for PDS')
        return
        
    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols)

    axs_pds = []
    [axs_pds.append(fig.add_subplot(gs[irow, icol]) )  for icol in range(ncols) for irow in range(nrows)]
    for ax in axs_pds[1:]:
        ax.sharex(axs_pds[0])

    
    k = 0

    xx = np.linspace(0,n_sample-1,n_sample)

    k = 0
    for ipds in range(n_tot_chan):
        
        if(dc.chmap_pds[ipds].data_type != data_type):
            continue

        '''ipds in global channels '''

        
        label = dc.chmap_pds[ipds].det+' Ch. '+str(dc.chmap_pds[ipds].chan)
        daq_ipds = dc.chmap_pds[ipds].daqch - daq_offset
        axs_pds[k].step(xx, data[daq_ipds], where='mid',c="k")
        axs_pds[k].text(0.98, 0.96, label, ha='right', va='bottom', transform=axs_pds[k].transAxes)


        rms = dc.evt_list[-1].noise_pds_filt.ped_rms[daq_ipds+daq_offset]
        #print('channel ', ipds, ' = ',dc.chmap_pds[ipds].det,' rms = ', rms)
        axs_pds[k].axhline(rms, c='orange', lw=0.5, zorder=100)
        axs_pds[k].axhline(-rms, c='orange', lw=0.5, zorder=100)

        
        """ draw rois """
        
        if(draw_roi == True):
            ymin, ymax = axs_pds[k].get_ylim()
            ROI = np.r_['-1',0,np.array(~mask[daq_ipds], dtype=int),0]
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

                axs_pds[k].add_patch(r)


        if(draw_peak==True):
            for p in dc.pds_peak_list:
                if(p.glob_ch == ipds):
                    axs_pds[k].axvline(p.max_t, c='tab:olive', lw=0.4)
                    axs_pds[k].axvline(p.start, c='coral', lw=0.4)

        k = k+1

    for irow in range(nrows):
        for icol in range(ncols):
            k = icol * nrows + irow
            if(irow == nrows-1):
                axs_pds[k].set_xlabel('Time Tick')
            else:
                axs_pds[k].tick_params(labelbottom=False)
                
    plt.tight_layout()

    save_with_details(fig, option, 'ED_pds_'+data_type)

    if(to_be_shown):
        plt.show()

    plt.close()


def draw_pds_peaks(option=None, to_be_shown=False):
    ts_stream = dc.evt_list[-1].pds_stream_time
    ts_trig = dc.evt_list[-1].pds_trig_time
    
    if(ts_stream < 0 and ts_trig < 0):
        return
    elif(ts_stream > 0 and ts_trig < 0):
        ts_ini = ts_stream
        duration = 1e-6*cf.n_pds_stream_sample/cf.pds_sampling
    elif(ts_stream < 0 and ts_trig > 0):
        ts_ini = ts_trig
        duration = 1e-6*cf.n_pds_trig_sample/cf.pds_sampling
    else:
        ts_ini = min(ts_trig, ts_stream)
        ts_stream_end = ts_stream+ 1e-6*cf.n_pds_stream_sample/cf.pds_sampling
        ts_trig_end = ts_trig+ 1e-6*cf.n_pds_trig_sample/cf.pds_sampling
        ts_end = max(ts_trig_end, ts_stream_end)
        duration = ts_end-ts_ini

    delta_t = (ts_ini - dc.evt_list[-1].event_time)*1e6
    duration *= 1e6

    peak_starts = [[] for x in range(cf.n_pds_tot_channels)]    
    [peak_starts[x.glob_ch].append(x.timestamp) for x in dc.pds_peak_list]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for ch in range(cf.n_pds_tot_channels):
        delay = dc.evt_list[-1].pds_time_offset[ch]
        for p in peak_starts[ch]:
            ax.plot([p,p],[ch,ch+1], c='k',lw=1)
            ax.plot([p+delay,p+delay],[ch,ch+1], c='tab:cyan',lw=1)

    ax.set_xlim(delta_t, delta_t+duration)
    ax.set_ylim(0, cf.n_pds_tot_channels)
    ax.axhline(16,c='r',lw=1)
    ax.axhline(32,c='r',lw=1)
    ax.axhline(39,c='r',lw=1)

    ax.axvline(0, c='r',lw=0.5)
    ax.set_xlabel('Time wrt to trigger [mus]')
    ax.set_ylabel('PDS global channel')


    for p in dc.pds_cluster_list:
        ax.axvline(p.timestamp, c='gray',lw=0.5, alpha=0.2, zorder=-100)        
    plt.show()
    
    
    

def charge_pds_zoom(pds_chan, charge_ch_range, charge_t_range, option=None, to_be_shown=False):
    fig = plt.figure()
    gs = gridspec.GridSpec(nrows=2, ncols=1)

    ax_pds = fig.add_subplot(gs[0,0])
    ax_trk = fig.add_subplot(gs[1,0])


    cmap_ed_coll = cc.cm.linear_tritanopic_krjcw_5_95_c24_r
    chmap.arange_in_view_channels()


    """ track part """
    chmin, chmax = charge_ch_range
    tmin, tmax   = charge_t_range

    ax_trk.imshow(dc.data[0, 2, chmin:chmax, tmin:tmax], 
                  origin = 'lower', 
                  aspect = 'auto', 
                  interpolation='none',
                  cmap   = cmap_ed_coll,
                  vmin   = 0, 
                  vmax   = 1500,
                  extent=[tmin, tmax, chmin, chmax])

    ax_trk.set_xlabel('WIB Time tick')
    ax_trk.set_ylabel('Channel')

    for trk in dc.tracks3D_list:
        if(trk.t0_corr > 0):
            tick_t0_corr = trk.t0_corr*cf.sampling
            if(tick_t0_corr>tmin and tick_t0_corr<tmax):
                ax_trk.axvline(tick_t0_corr, c='limegreen', lw=1)
                #print('track t0 = ', tick_t0_corr, trk.t0_corr)

                tick_t0_corr_resamp = tick_t0_corr*32
                ax_pds.axvline(tick_t0_corr_resamp,  c='limegreen', lw=1, ls='dotted')
                
                tstart = trk.ini_time
                ax_trk.axvline(tstart, c='r', lw=1)
                

                tstart_resamp = tstart*32
                ax_pds.axvline(tstart_resamp,  c='r', lw=1, ls='dotted')


    
    for h in dc.hits_list:
        if(h.view==2 and h.module==0 and h.channel >= chmin and h.channel <chmax and h.start > tmin and h.stop < tmax):            
            r = patches.Rectangle((h.start, h.channel),h.stop-h.start,1, linewidth=.5,edgecolor='k',facecolor='none')
            
            ax_trk.add_patch(r)
            h.dump()
            
    
    tmin_pds = tmin*32
    tmax_pds = tmax*32
    xx = np.linspace(tmin_pds, tmax_pds-1, tmax_pds-tmin_pds)

    color = ['k', 'steelblue']
    for chan,c in zip(pds_chan, color):
        ax_pds.step(xx, dc.data_pds[chan, tmin_pds:tmax_pds], where='mid',c=c, label='PDS Channel '+str(dc.chmap_pds[chan].chan))
        for p in dc.pds_peak_list:
            if(p.glob_ch == chan):
                ax_pds.axvline(p.max_t, c='yellowgreen', lw=1)
                ax_pds.axvline(p.start, c='coral', lw=1)
                
                
                tpds_resamp = p.max_t/32
                tstart_pds_resamp = p.start/32
                if(tpds_resamp>tmin and tpds_resamp < tmax):
                    
                    ax_trk.axvline(tstart_pds_resamp, c='coral', lw=1, ls='dotted')
                    ax_trk.axvline(tpds_resamp, c='yellowgreen', lw=1, ls='dotted')

                
    ax_pds.set_xlim(tmin_pds, tmax_pds)
    ax_pds.set_xlabel('DAPHNE Time tick')
    ax_pds.legend(frameon=False, loc='upper right')
    


    
    tns_pds_delay = dc.evt_list[-1].pds_time_ns-dc.evt_list[-1].charge_time_ns
    fig.suptitle('DAPHNE-WIB events delay = '+str(tns_pds_delay)+' ns')

    fig.tight_layout()



    save_with_details(fig, option, 'ED_coll_pds_zoom_track')

    if(to_be_shown):
        plt.show()

    plt.close()
