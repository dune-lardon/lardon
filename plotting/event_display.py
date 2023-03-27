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



def draw(module, view, ax, adc_min, adc_max, roi, noise):
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

    ax.imshow(dc.data[module, view, :cf.view_nchan[view], :].transpose(), 
              #origin = 'lower', 
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

    n_signal = len(set(cf.view_type))
    n_mod = sum(cf.module_used)


    width = 8 if cf.n_view == 2 else 10
    height = n_mod*width/2


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
    for im, use in enumerate(cf.module_used):
        #print(im, use, irow)

        if(use==False):
            continue            
 
        for iv in range(cf.n_view):            
            axs[iv].append(fig.add_subplot(gs[1+irow,iv]) if iv==0 else fig.add_subplot(gs[1+irow,iv], sharey=axs[0][-1]))
            
            if(cf.view_type[iv] == 'Induction'):
                adc_min, adc_max = adc_ind[0], adc_ind[1]
                vname = "Ind."
            else:
                adc_min, adc_max = adc_coll[0], adc_coll[1]
                vname = "Coll."

            axs[iv][-1] = draw(im, iv, axs[iv][-1], adc_min, adc_max, roi, noise)

            if(draw_hits):
                for h in dc.hits_list:
                    if(h.view==iv and h.module==im):
                        if(h.signal != cf.view_type[h.view]):
                            color = 'r'
                        else:
                            color='k'

                        if(h.matched == -5555):
                            color='orange'
                        r = patches.Rectangle((h.channel-0.5,h.start),1,h.stop-h.start,linewidth=.5,edgecolor=color,facecolor='none')
                        
                        axs[iv][-1].add_patch(r)
        


            title = ''
            if(len(cf.module_used)>1):
                title += 'CRP '+str(im)+' - '
        
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

    for a in axs_col:
        a.set_title('Collected Charge [ADC]')


    cb = fig.colorbar(axs[0][0].images[-1], cax=axs_col[0], orientation='horizontal')
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')

    if(n_signal>1):
        cb = fig.colorbar(axs[-1][0].images[-1], cax=axs_col[-1], orientation='horizontal')
        cb.ax.xaxis.set_ticks_position('top')
        cb.ax.xaxis.set_label_position('top')


    plt.subplots_adjust(top=0.865, bottom=0.131, left=0.095, right=0.905, hspace=0.232, wspace=0.096)


    save_with_details(fig, option, 'ED_vch_hits_found' if draw_hits else 'ED_vch')


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
                        if(h.signal != cf.view_type[h.view]):
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
