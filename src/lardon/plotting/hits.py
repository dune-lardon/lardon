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
from lardon.plotting.waveforms import *
from lardon.plotting.event_display import *

cmap_ed_coll = cc.cm.linear_tritanopic_krjcw_5_95_c24_r
cmap_ed_ind  = cc.cm.diverging_tritanopic_cwr_75_98_c20

def plot_hit_fromID(hitID,adc_min=-1, adc_max=-1, option=None, to_be_shown=False):
    if(hitID < dc.n_tot_hits or hitID>= np.sum(dc.evt_list[-1].n_hits)+dc.n_tot_hits):
        print(hitID, ' hit ID does not exist...')
        return

    
        
    local_nb = hitID - dc.n_tot_hits
    the_hit = dc.hits_list[local_nb]

    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(nrows=3, ncols=2, height_ratios=[1,15,5], width_ratios=[1,4])

    ax_ED = fig.add_subplot(gs[1,0])
    ax_ED_col = fig.add_subplot(gs[0,0])
    ax_wvf = fig.add_subplot(gs[1,1])
    ax_infos = fig.add_subplot(gs[2,1])

    view, vch, daq_ch = the_hit.view, the_hit.channel, the_hit.daq_channel
    start, stop = the_hit.start, the_hit.stop
    tmax, tmin =  the_hit.max_t, the_hit.min_t

    if(adc_min < 0 and adc_max < 0):
        if(the_hit.signal == 'Induction'):
            adc_min, adc_max = -50,50
        else:
            adc_min, adc_max = -10,150


    ped_mean = dc.evt_list[-1].noise_filt.ped_mean[daq_ch]
    legend = "v"+str(view)+" ch"+str(vch)
    ax_wvf = draw_current_waveform(daq_ch, -1, -1, ax=ax_wvf, c='k',lw=.5)

    ax_wvf.fill_between(np.linspace(start,stop,stop-start+1), dc.data_daq[daq_ch, start:stop+1], ped_mean, step='mid',color='r', alpha=0.25,zorder=200)
    ax_wvf.step(np.linspace(start,stop,stop-start+1), dc.data_daq[daq_ch, start:stop+1], linewidth=1, c='r',zorder=250,where='mid')


    ax_wvf.axvline(start, ls='dotted',c='r',lw=.5,zorder=300)
    ax_wvf.axvline(stop, ls='dotted',c='r',lw=.5,zorder=300)


    chmap.arange_in_view_channels()
    ch_off = 5
    ch_start = vch-ch_off if vch > ch_off else 0
    ch_stop  = vch+ch_off if vch < cf.view_nchan[view]+ch_off else cf.view_nchan[view]
    
    t_off = 100
    t_start = start-t_off if start > t_off else 0
    t_stop  = stop+t_off if stop < cf.n_sample+t_off else cf.n_sample

    ax_wvf.set_xlim(t_start, t_stop)
    ax_wvf.set_xlabel('Time [tick]')
    ax_wvf.set_ylabel('ADC')

    cmap = cmap_ed_coll if(cf.view_type[view] == 'Collection') else cmap_ed_ind
    im = ax_ED.imshow(dc.data[view, ch_start:ch_stop, t_start:t_stop].transpose(), 
                      origin = 'lower', 
                      aspect = 'auto', 
                      interpolation='none',
                      cmap   = cmap,
                      vmin   = adc_min, 
                      vmax   = adc_max,
                      extent = [ch_start-.5,ch_stop-.5,t_start,t_stop])
    r = patches.Rectangle((vch-0.5,start),1,stop-start,linewidth=.5,edgecolor='k',facecolor='none')

    ax_ED.add_patch(r)
    ax_ED.set_xlabel('Channel Number')
    ax_ED.set_ylabel('Time [tick]')

    cb = fig.colorbar(im, cax=ax_ED_col, orientation='horizontal')
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')
    ax_ED_col.set_title('ADC')

    ax_wvf.set_title('Hit \#'+str(hitID)+' in View '+str(view)+' Channel '+str(vch))


    infos = 'Signal Type: '+the_hit.signal
    infos += '\n'
    infos += 'From tick '+str(start)+' to tick '+str(stop)
    infos += '\n'
    infos += 'Positive Signal : Peak at tick %i, value of %.1f ADC (%.2f fC), Charge : %.2f fC'%(tmax, the_hit.max_adc, the_hit.max_fC, the_hit.charge_pos)
    if(the_hit.signal=='Induction'):
        infos += '\n'
        infos += 'Zero Crossed at tick %i'%(the_hit.zero_t)
        infos += '\n'
        infos += 'Negative Signal : Peak at tick %i, value of %.1f ADC (%.2f fC), Charge : %.2f fC'%(tmin, the_hit.min_adc, the_hit.min_fC, the_hit.charge_neg)


    ax_infos.text(0,0.5,infos,ha='left',va='center')
    ax_infos.axis('off')

    save_with_details(fig, option, 'hit'+str(hitID)+'_details')

    if(to_be_shown):
        plt.show()

    plt.close()

