import lardon.config as cf
import lardon.data_containers as dc
import lardon.channel_mapping as chmap

import time as time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import collections  as mc
import itertools as itr
import math
import colorcet as cc
import matplotlib.patches as patches
from lardon.plotting.save_plot import *


cmap_ed_coll = cc.cm.linear_tritanopic_krjcw_5_95_c24_r
cmap_ed_ind  = cc.cm.diverging_tritanopic_cwr_75_98_c20



def draw(view, ax, adc_min, adc_max, roi, noise):
    
    #print('draw view ',view)
    ax = plt.gca() if ax is None else ax
    cmap = cmap_ed_coll if(cf.view_type[cf.imod][view] == 'Collection') else cmap_ed_ind
    
    if(roi==True):
        temp = dc.data_daq.copy()
        dc.data_daq *= ~dc.mask_daq
        chmap.arange_in_view_channels()

    if(noise==True):
        temp = dc.data_daq.copy()
        dc.data_daq *= dc.mask_daq
        chmap.arange_in_view_channels()

    min_ch = 0
    max_ch = cf.view_nchan[view]
    
    if(dc.evt_list[-1].det == "pdhd"):
        if(view == 2):
            if(cf.imod < 2):
                min_ch = 480
            else:
                min_ch = 480
                max_ch = 960#480
        if(cf.imod < 2):
            origin = 'lower'
            tmin,tmax=0,cf.n_sample[cf.imod]
        else:
            origin='upper'
            tmin,tmax = cf.n_sample[cf.imod], 0
         
    elif(dc.evt_list[-1].det == "pdvd"):
        #if(cf.imod < 2):
        if(cf.drift_direction[cf.imod] > 0):
            origin='upper'
            tmin,tmax = cf.n_sample[cf.imod], 0
        else:
            origin='lower'
            tmin,tmax=0,cf.n_sample[cf.imod]
                  
    else:
        origin='lower'
        tmin,tmax=0,cf.n_sample[cf.imod]

    min_ch = 0
    max_ch = cf.view_nchan[view]

        
    ax.imshow(dc.data[view, min_ch:max_ch, :].transpose(), 
              origin = origin, 
              aspect = 'auto', 
              interpolation='none',
              cmap   = cmap,    
              vmin   = adc_min, 
              vmax   = adc_max,
              extent = [min_ch, max_ch, tmin,tmax])    
    
    
    if(roi==True or noise == True):
        dc.data_daq = temp


    return ax


def event_display_per_view(adc_ind=[-10,10], adc_coll=[-5,30], option=None, to_be_shown=False, draw_hits=False, roi=False, noise=False):
    chmap.arange_in_view_channels()

    n_signal = len(set(cf.view_type[cf.imod]))
    n_mod = 1 #sum(cf.module_used)


    width = 8 if cf.n_view == 2 else 10
    height = n_mod*width/2 #10


    fig = plt.figure(figsize=(width, height))

    gs = gridspec.GridSpec(nrows=n_mod+1, 
                           ncols=cf.n_view,
                           height_ratios=[1 if i == 0 else 10 for i in range(n_mod+1)])

    if(n_signal == 2):
        axs_col = [fig.add_subplot(gs[0,:2]), fig.add_subplot(gs[0,2:])]
    else:
        axs_col = [fig.add_subplot(gs[0,:])]


    
    axs = [[] for x in range(cf.n_view)]

    irow = 0
    idx_ind, idx_coll = -1, -1
    for iv in range(cf.n_view):            
        axs[iv].append(fig.add_subplot(gs[1+irow,iv]) if iv==0 else fig.add_subplot(gs[1+irow,iv], sharey=axs[0][-1]))
            
        if(cf.view_type[cf.imod][iv] == 'Induction'):
            adc_min, adc_max = adc_ind[0], adc_ind[1]
            vname = "Ind."
            idx_ind = iv
        else:
            adc_min, adc_max = adc_coll[0], adc_coll[1]
            vname = "Coll."
            idx_coll = iv
            
        axs[iv][-1] = draw(iv, axs[iv][-1], adc_min, adc_max, roi, noise)

        if(draw_hits):
            for h in dc.hits_list:
                if(h.view==iv and h.module==cf.imod):
                    color = 'k'
                    '''
                    if(h.signal != cf.view_type[h.view]):
                        color = 'r'
                    if(h.match_sh != -9999):
                        color = 'gold'                        
                    if(h.match_2D != -9999):
                        color = 'gold'
                    if(h.match_3D != -9999):
                        color = 'r'
                    
                    if(h.has_3D == True):
                        color = 'gold'                        
                    '''
                    if(h.match_sh != -9999):
                        color = 'gold'
                    if(h.match_2D != -9999):
                        color = 'green'                        
                    if(h.match_3D != -9999):
                        color = 'r'
                    

                    r = patches.Rectangle((h.channel,h.start),1,h.stop-h.start,linewidth=.5,edgecolor=color,facecolor='none')

                    axs[iv][-1].add_patch(r)
                        


        title = ''
        out_mod = ''

        title += cf.module_name[cf.imod]+' - '
        out_mod += '_'+cf.module_name[cf.imod]
        
        title += 'View '+str(iv)
            
        if(n_signal > 1):
            title += '/'+cf.view_name[iv]+' ('+vname+')'

        axs[iv][-1].set_title(title)        

    irow += 1

    
    for a in axs[0]:
        a.set_ylabel('Time [tick]')
    for a in axs[-1]:
        a.set_ylabel('Time [tick]')
        a.yaxis.tick_right()
        a.yaxis.set_label_position("right")

    if(n_mod > 1):
        for a in axs:
            for b in a[:-1]:
                b.tick_params(labelbottom=False)


    if(n_signal>1):
        for a in axs[1]:
            a.tick_params(labelleft=False)

    for a in axs:
        a[-1].set_xlabel('View Channel')


    """
    for a in axs_col:
        a.set_title('Collected Charge [ADC]')
    """

    axs_col[0].set_title('Induced Charge [ADC]')
    axs_col[1].set_title('Collected Charge [ADC]')

    cb = fig.colorbar(axs[idx_ind][-1].images[-1], cax=axs_col[0], orientation='horizontal')
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')

    if(n_signal>1):
        cb = fig.colorbar(axs[idx_coll][-1].images[-1], cax=axs_col[-1], orientation='horizontal')
        cb.ax.xaxis.set_ticks_position('top')
        cb.ax.xaxis.set_label_position('top')



    #plt.subplots_adjust(top=0.865, bottom=0.131, left=0.095, right=0.905, hspace=0.232, wspace=0.096)
    plt.subplots_adjust(top=0.865, bottom=0.131, left=0.105, right=0.90, hspace=0.232, wspace=0.096)



    save_with_details(fig, option, 'ED_vch_hits_found'+out_mod if draw_hits else 'ED_vch'+out_mod)


    if(to_be_shown):
        plt.show()

    plt.close()




def event_display_per_daqch(adc_range=[-10,10],  option=None, to_be_shown=False):
    fig = plt.figure(figsize=(9,4))

    gs = gridspec.GridSpec(nrows=2, 
                           ncols=1,
                           height_ratios=[1, 10])

    ax_col = fig.add_subplot(gs[0,:])
    ax     = fig.add_subplot(gs[1, :])

    ax.imshow(dc.data_daq.transpose(), 
              origin = 'lower', 
              aspect = 'auto', 
              interpolation='none',
              cmap   = cmap_ed_ind,
              vmin   = adc_range[0], 
              vmax   = adc_range[1])

    
    ax.set_xlabel('DAQ Channel Number')
    
    ax.set_ylabel('Time [tick]')
    title = ''
    out_mod = ''

    title += cf.module_name[cf.imod]+' - '
    out_mod += '_'+cf.module_name[cf.imod]
    
    
    ax_col.set_title(title+'Collected Charge [ADC]')
    
    cb = fig.colorbar(ax.images[-1], cax=ax_col, orientation='horizontal')
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')

    plt.tight_layout()


    save_with_details(fig, option, 'ED_daqch'+out_mod)


    if(to_be_shown):
        plt.show()

    plt.close()





def event_display_per_view_hits_found(adc_ind=[-10,10], adc_coll=[-5,30],  option=None, to_be_shown=False):
    return event_display_per_view(adc_ind=adc_ind, adc_coll=adc_coll,  option=option, to_be_shown=to_be_shown, draw_hits=True)

def event_display_per_view_roi(adc_ind=[-10,10], adc_coll=[-5,30],  option=None, to_be_shown=False):
    return event_display_per_view(adc_ind=adc_ind, adc_coll=adc_coll,  option=option, to_be_shown=to_be_shown, draw_hits=False, roi=True, noise=False)

def event_display_per_view_noise(adc_ind=[-10,10], adc_coll=[-5,30],  option=None, to_be_shown=False):
    return event_display_per_view(adc_ind=adc_ind, adc_coll=adc_coll,  option=option, to_be_shown=to_be_shown, draw_hits=False, roi=False, noise=True)


def event_display_per_view_dp(adc_coll=[-5,30], option=None, to_be_shown=False, draw_hits=False, roi=False, noise=False):

    n_mod = sum(cf.module_used)
    
    fig = plt.figure(figsize=(8,4*n_mod))

    gs = gridspec.GridSpec(nrows=1+n_mod, ncols=2, height_ratios=[1 if i == 0 else 10 for i in range(n_mod+1)])

    ax_col = fig.add_subplot(gs[0,:])
    axs = [[] for iv in range(cf.n_view)]
    ir = 0
    for imod, use in enumerate(cf.module_used):
        print(imod, use)
        if(use==False):
            continue
        ir += 1
        for iv in range(cf.n_view):
            if(iv==0):
                axs[iv].append(fig.add_subplot(gs[ir, iv]))
            else:
                axs[iv].append(fig.add_subplot(gs[ir, iv], sharey=axs[0][-1]))
                
            axs[iv][-1] = draw(imod, iv, axs[iv][-1], adc_coll[0], adc_coll[1], roi, noise)

            if(draw_hits):
                for h in dc.hits_list:
                    if(h.view==iv and h.module==imod):
                        if(h.signal != cf.view_type[cf.imod][h.view]):
                            color = 'r'
                        else:
                            color='k'

                        r = patches.Rectangle((h.channel-0.5,h.start),1,h.stop-h.start,linewidth=.5,edgecolor=color,facecolor='none')
                            
                        axs[iv][-1].add_patch(r)



        axs[iv][-1].set_title('CRP '+str(imod)+' - View '+str(iv))

    for a in axs[0]:
        a.set_ylabel('Time')
    for a in axs[-1]:
        a.set_ylabel('Time')
        a.yaxis.tick_right()
        a.yaxis.set_label_position("right")

    for a,b in zip(axs[0][:-1], axs[1][:-1]):
        a.tick_params(labelbottom=False)
        b.tick_params(labelbottom=False)
        
    axs[0][-1].set_xlabel('View Channel')
    axs[1][-1].set_xlabel('View Channel')


    ax_col.set_title('Collected Charge [ADC]')
                                 
    cb = fig.colorbar(axs[-1][-1].images[-1], cax=ax_col, orientation='horizontal')
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')

    plt.tight_layout()


    save_with_details(fig, option, 'ED_daqch')

    if(to_be_shown):
        plt.show()

    plt.close()



def event_display_coll_pds(adc_coll=[-5,150], option=None, to_be_shown=False, draw_trk_t0 = False):
    chmap.arange_in_view_channels()
    cmap = cmap_ed_coll
    adc_min, adc_max = adc_coll[0], adc_coll[1]
    fig = plt.figure(figsize=(8, 8))

    gs = gridspec.GridSpec(nrows=3, ncols=2, height_ratios = [1, 10, 10])
    axs_region = []

    ax_col = fig.add_subplot(gs[0,:])
    axs_data = []
    axs_pds_1 = []
    axs_pds_2 = []

    ts_pds = dc.evt_list[-1].pds_time_s + dc.evt_list[-1].pds_time_ns*1e-9
    ts_charge = dc.evt_list[-1].charge_time_s + dc.evt_list[-1].charge_time_ns*1e-9
    tns_pds_delay = dc.evt_list[-1].pds_time_ns-dc.evt_list[-1].charge_time_ns
    print('TS PDS = ', ts_pds, ' ----- ', dc.evt_list[-1].pds_time_ns)
    print('TS WIB = ', ts_charge, ' ------ ', dc.evt_list[-1].charge_time_ns)
    print('nanosecond delays : ', tns_pds_delay)
    ts_pds_delay = (ts_pds - ts_charge)*1e6
    tick_pds_delay = int(ts_pds_delay*cf.pds_sampling)
    print('---> ts_pds_delay = ', ts_pds_delay, ' mus')
    print(' === ', ts_pds_delay*cf.pds_sampling, ' pds ticks')
    print('nb of charge sample = ', cf.n_sample, ' nb of light sample = ', cf.n_pds_sample, ' nb of pds samples to have same wib window = ', cf.n_sample*32)
    #n_pds_ticks_window = cf.n_sample*32

    #if(tick_pds_delay >=0):
        #idx_start = tick_pds_delay
    
    k = 0
    for irow in range(2):
        for icol in range(2):
            axs_region.append(fig.add_subplot(gs[irow+1, icol]))
            axs_region[-1].set_xticks([])
            axs_region[-1].set_yticks([])
            
            gsgs = gridspec.GridSpecFromSubplotSpec(1, 2, wspace=0, subplot_spec=gs[irow+1, icol], width_ratios=[2, 1])
            axs_data.append(fig.add_subplot(gsgs[0,0]))
            gsgsgs = gridspec.GridSpecFromSubplotSpec(1, 2, wspace=0, subplot_spec=gsgs[0, 1])
            axs_pds_1.append(fig.add_subplot(gsgsgs[0,0]))
            axs_pds_2.append(fig.add_subplot(gsgsgs[0,1], sharey=axs_pds_1[k]))
            k += 1

    k = 0
    ch_pds = [0, 6, 2, 4]
    for i in [2, 3, 0, 1]:
        im = axs_data[i].imshow(dc.data[0, 2, k*292:(k+1)*292, :].transpose(), 
                                aspect = 'auto', 
                                interpolation='none',
                                cmap   = cmap,
                                vmin   = adc_min, 
                                vmax   = adc_max,
                                extent=[k*292, (k+1)*292-1, cf.n_sample, 0])
        #print(dc.data_pds[k*2].shape, cf.n_pds_sample)

        if(draw_trk_t0 == True):
            for trk in dc.tracks3D_list:
                if(trk.t0_corr > 0):
                    tick_t0_corr = trk.t0_corr*cf.sampling
                    axs_data[i].axhline(tick_t0_corr, c='r', lw=1)

                    
        axs_pds_1[i].plot(dc.data_pds[ch_pds[k]], np.linspace(0, cf.n_pds_sample, cf.n_pds_sample), c='k')
                
        axs_pds_2[i].plot(dc.data_pds[ch_pds[k]+1], np.linspace(0, cf.n_pds_sample, cf.n_pds_sample), c='k')

        axs_region[i].set_title('Around '+dc.chmap_pds[ch_pds[k]].det)
        
        for a in [axs_pds_1[i], axs_pds_2[i]]:

            a.set_ylim(0, cf.n_pds_sample)
            a.invert_yaxis()
            a.set_yticks([])
            a.set_xlim(-1000, 10000)
            a.set_xticks([])
        if(k == 0):
            cb = fig.colorbar(im, cax=ax_col, orientation='horizontal')
            cb.ax.xaxis.set_ticks_position('top')
            cb.ax.xaxis.set_label_position('top')
            ax_col.set_title(r'Charge Collected on View 2 [ADC]')
        
        k = k+1


    plt.tight_layout()


    save_with_details(fig, option, 'ED_coll_pds')

    if(to_be_shown):
        plt.show()

    plt.close()
