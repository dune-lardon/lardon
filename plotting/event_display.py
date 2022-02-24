import config as cf
import data_containers as dc
import time as time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import collections  as mc
import itertools as itr
import math
import colorcet as cc
import channel_mapping as chmap

import matplotlib.patches as patches
from .save_plot import *


cmap_ed_coll = cc.cm.linear_tritanopic_krjcw_5_95_c24_r
cmap_ed_ind  = cc.cm.diverging_tritanopic_cwr_75_98_c20



def draw(view, ax, adc_min, adc_max, roi, noise):
    ax = plt.gca() if ax is None else ax
    cmap = cmap_ed_coll if(cf.view_type[view] == 'Collection') else cmap_ed_ind

    if(roi==True):
        temp = dc.data_daq.copy()
        dc.data_daq *= ~dc.mask_daq
        chmap.arange_in_view_channels()

    if(noise==True):
        temp = dc.data_daq.copy()
        dc.data_daq *= dc.mask_daq
        chmap.arange_in_view_channels()

    ax.imshow(dc.data[view, :cf.view_nchan[view], :].transpose(), 
              origin = 'lower', 
              aspect = 'auto', 
              interpolation='none',
              cmap   = cmap,
              vmin   = adc_min, 
              vmax   = adc_max)

    if(roi==True or noise == True):
        dc.data_daq = temp


    return ax


def event_display_per_view(adc_ind=[-10,10], adc_coll=[-5,30], option=None, to_be_shown=False, draw_hits=False, roi=False, noise=False):
    chmap.arange_in_view_channels()
    fig = plt.figure(figsize=(10,5))

    gs = gridspec.GridSpec(nrows=2, 
                           ncols=3,
                           height_ratios=[1, 10])

    ax_col_ind  = fig.add_subplot(gs[0,:2])
    ax_col_coll = fig.add_subplot(gs[0,2])

    ax_v0  = fig.add_subplot(gs[1, 0])
    ax_v1  = fig.add_subplot(gs[1, 1], sharey=ax_v0)
    ax_v2  = fig.add_subplot(gs[1, 2], sharey=ax_v0)

    axs = [ax_v0, ax_v1, ax_v2]
    for iv in range(cf.n_view):

        if(cf.view_type[iv] == 'Induction'):
            
            axs[iv] = draw(iv, axs[iv], adc_ind[0], adc_ind[1], roi, noise)
            vname = 'Ind.'
        else:
            axs[iv] = draw(iv, axs[iv], adc_coll[0], adc_coll[1], roi, noise)
            vname = 'Coll.'

        if(draw_hits):
            for h in dc.hits_list:
                if(h.view==iv):
                    if(h.signal != cf.view_type[h.view]):
                        color = 'r'
                    else:
                        color='k'

                    r = patches.Rectangle((h.channel-0.5,h.start),1,h.stop-h.start,linewidth=.5,edgecolor=color,facecolor='none')

                    axs[iv].add_patch(r)

        axs[iv].set_title('View '+str(iv)+'/'+cf.view_name[iv]+' ('+vname+')')
        axs[iv].set_xlabel('Channel Number')
        
        if(iv == 0):
            axs[iv].set_ylabel('Time [tick]')

        if(iv == 1):
            #axs[iv].set_yticklabels([])
            axs[iv].tick_params(labelleft=False)
        if(iv == 2):
            axs[iv].set_ylabel('Time [tick]')
            axs[iv].yaxis.tick_right()
            axs[iv].yaxis.set_label_position("right")

    for a in [ax_col_ind, ax_col_coll]:
        a.set_title('Collected Charge [ADC]')

    cb = fig.colorbar(ax_v0.images[-1], cax=ax_col_ind, orientation='horizontal')
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')

    cb = fig.colorbar(ax_v2.images[-1], cax=ax_col_coll, orientation='horizontal')
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')

    #plt.tight_layout()
    plt.subplots_adjust(top=0.869, bottom=0.131, left=0.089, right=0.911, hspace=0.232, wspace=0.096)


    save_with_details(fig, option, 'ED_vch_hits_found' if draw_hits else 'ED_vch')
    #plt.savefig(cf.plot_path+'/ED_vch'+option+'_'+elec+'_run_'+run_nb+'_evt_'+evt_nb+'.png')

    if(to_be_shown):
        plt.show()

    plt.close()



def event_display_per_daqch(adc_range=[-10,10], option=None, to_be_shown=False):
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
    ax_col.set_title('Collected Charge [ADC]')
    cb = fig.colorbar(ax.images[-1], cax=ax_col, orientation='horizontal')
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')

    plt.tight_layout()


    save_with_details(fig, option, 'ED_daqch')


    if(to_be_shown):
        plt.show()

    plt.close()





def event_display_per_view_hits_found(adc_ind=[-10,10], adc_coll=[-5,30], option=None, to_be_shown=False):
    return event_display_per_view(adc_ind=adc_ind, adc_coll=adc_coll, option=option, to_be_shown=to_be_shown, draw_hits=True)

def event_display_per_view_roi(adc_ind=[-10,10], adc_coll=[-5,30], option=None, to_be_shown=False):
    return event_display_per_view(adc_ind=adc_ind, adc_coll=adc_coll, option=option, to_be_shown=to_be_shown, draw_hits=False, roi=True, noise=False)

def event_display_per_view_noise(adc_ind=[-10,10], adc_coll=[-5,30], option=None, to_be_shown=False):
    return event_display_per_view(adc_ind=adc_ind, adc_coll=adc_coll, option=option, to_be_shown=to_be_shown, draw_hits=False, roi=False, noise=True)
