import config as cf
import data_containers as dc

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import collections  as mc
import itertools as itr
import math
import colorcet as cc

cmap_ed_coll = cc.cm.linear_tritanopic_krjcw_5_95_c24_r
cmap_ed_ind  = cc.cm.diverging_tritanopic_cwr_75_98_c20


def draw(view, ax, adc_min, adc_max):
    ax = plt.gca() if ax is None else ax
    cmap = cmap_ed_coll if(cf.view_type[view] == 'Collection') else cmap_ed_ind

    ax.imshow(dc.data[view, :cf.view_nchan[view], :].transpose(), 
              origin = 'lower', 
              aspect = 'auto', 
              interpolation='none',
              cmap   = cmap,
              vmin   = adc_min, 
              vmax   = adc_max)

    return ax

def event_display_per_view(adc_min_ind=-10, adc_max_ind=10, adc_min_coll=-5, adc_max_coll=30):
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
            
            axs[iv] = draw(iv, axs[iv], adc_min_ind, adc_max_ind)
            vname = 'Ind.'
        else:
            axs[iv] = draw(iv, axs[iv], adc_min_coll, adc_max_coll)
            vname = 'Coll.'

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

    plt.tight_layout()
    plt.show()



def event_display_per_daqch(adc_min=-10, adc_max=10):
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
              vmin   = adc_min, 
              vmax   = adc_max)

    
    ax.set_xlabel('DAQ Channel Number')
    
    ax.set_ylabel('Time [tick]')
    ax_col.set_title('Collected Charge [ADC]')
    cb = fig.colorbar(ax.images[-1], cax=ax_col, orientation='horizontal')
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')

    plt.tight_layout()
    plt.show()