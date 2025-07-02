import config as cf
import data_containers as dc
import channel_mapping as chmap

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
from .save_plot import *



def draw_pds_ED( option=None, to_be_shown=False, draw_peak=False, draw_cluster=False, draw_roi=False):

    fig = plt.figure()
    n_modules = cf.pds_n_modules
    n_chan = cf.n_pds_channels
    gs = gridspec.GridSpec(nrows=int(n_modules/2), ncols=2)

    axs = []
    axs_pds = []
    k = 0

    xx = np.linspace(0,cf.n_pds_sample-1,cf.n_pds_sample)
    for irow in range(int(n_modules/2)):
        for icol in range(2):
        
            module = 2*irow + icol

            if(cf.pds_modules_type[module] == 'Cathode'):

                axs.append( fig.add_subplot(gs[irow, icol]) )
                gsgs = gridspec.GridSpecFromSubplotSpec(2, 1, hspace=0, subplot_spec=gs[irow, icol])

                axs_pds.append(fig.add_subplot(gsgs[0,0]))
                axs_pds.append(fig.add_subplot(gsgs[1,0]))
                axs_pds[-1].sharex(axs_pds[-2])
                #axs_pds[-2].get_shared_x_axes().join(axs_pds[-2], axs_pds[-1])

                axs[-1].set_xticks([])
                axs[-1].set_yticks([])

            else:
                axs_pds.append(fig.add_subplot(gs[irow, icol]) )

    for ipds in range(n_chan):
                
        label = dc.chmap_pds[ipds].det+' Ch. '+str(dc.chmap_pds[ipds].chan)

        axs_pds[ipds].step(xx, dc.data_pds[ipds, :], where='mid',c="k", label=label)
        axs_pds[ipds].legend(frameon=False, loc='upper right')
        axs_pds[ipds].set_ylabel('ADC')

        rms = dc.evt_list[-1].noise_pds_filt.ped_rms[ipds]

        axs_pds[ipds].axhline(rms, c='orange', lw=0.5, zorder=100)
        axs_pds[ipds].axhline(-rms, c='orange', lw=0.5, zorder=100)
                
        """ draw rois """
        if(draw_roi == True):
            ymin, ymax = axs_pds[ipds].get_ylim()
            ROI = np.r_['-1',0,np.array(~dc.mask_pds[ipds], dtype=int),0]
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

                axs_pds[ipds].add_patch(r)


        if(draw_peak==True):
            for p in dc.pds_peak_list:
                if(p.glob_ch == ipds):
                    axs_pds[ipds].axvline(p.max_t, c='coral', lw=0.4)



        col_size=['#c1e7ff', '#9dc6e0', '#7aa6c2', '#5886a5', '#346888' , '#004c6d']
        if(draw_cluster==True):
            for cl in dc.pds_cluster_list:
                for icl in range(cl.size):
                    if(cl.glob_chans[icl] == ipds):                        
                        axs_pds[ipds].axvline(cl.t_maxs[icl], lw=4, color=col_size[int(cl.size/2)-1], alpha=0.8, zorder=-100)
                            

        axs_pds[ipds].set_xlabel('Time tick')
        #for j in range(2):
        ymin, ymax = axs_pds[ipds].get_ylim()
        axs_pds[ipds].set_ylim(-100, ymax)
        k = k+1
    
    plt.tight_layout()

    save_with_details(fig, option, 'ED_pds')

    if(to_be_shown):
        plt.show()

    plt.close()




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
