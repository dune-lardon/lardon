import config as cf
import data_containers as dc
import lar_param as lar

from .select_hits import *
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import collections  as mc
from matplotlib.legend_handler import HandlerTuple
import itertools as itr
import math
import colorcet as cc


cmap_ed = cc.cm.kbc_r
marker_size = 5

def draw_hits(pos, time, z=[], ax=None, **kwargs):

    ax = plt.gca() if ax is None else ax
    print('there is ', len(pos), ' in this view ')
    if(len(pos) == 0):
        return ax
    
    if(len(z) > 0):
        ax.scatter(pos, time, c=z, **kwargs)
    else:
        ax.scatter(pos, time, **kwargs)
    return ax

def draw_all_hits(axs, sel='True', adc=False, charge=False, **kwargs):


    for iview in range(cf.n_view):
        z = []
        if(adc==True):
            z = np.fabs(get_hits_adc(iview, sel))
        elif(charge==True):
            z = get_hits_charge(iview, sel)

        axs[iview] = draw_hits(pos=get_hits_pos(iview, sel), 
                               time=get_hits_z(iview, sel), 
                               z=z,
                               ax=axs[iview], **kwargs)
    return axs



def draw_tracks(pos, time, ax=None, legend="", **kwargs):

    ax = plt.gca() if ax is None else ax
    
    if(len(pos) == 0):
        return ax

    if(len(legend)>0):
        ax.plot(pos[0], time[0], label=legend, **kwargs)
        
    for tx,tz in zip(pos, time):
        ax.plot(tx,tz, **kwargs)
    
    return ax


def draw_all_tracks(axs, sel='True', legend="", **kwargs):

    

    for iview in range(cf.n_view):
        axs[iview] = draw_tracks(pos=get_2dtracks_pos(iview,sel), 
                                 time=get_2dtracks_z(iview,sel), 
                                 ax=axs[iview], 
                                 legend=legend,
                                 **kwargs)
    return axs





def template_data_view():

    fig = plt.figure(figsize=(10,5))
    gs = gridspec.GridSpec(nrows=2, ncols=cf.n_view, 
                           height_ratios=[1,20], 
                           width_ratios=cf.view_length)
    

                           
    ax_col = fig.add_subplot(gs[0,:])

    axs = [fig.add_subplot(gs[1, iv]) for iv in range(cf.n_view)]
    v = lar.drift_velocity()
    
    axs[0].get_shared_y_axes().join(*axs)

    for iv in range(cf.n_view):
        axs[iv].set_title('View '+str(iv)+"/"+cf.view_name[iv])
        axs[iv].set_xlabel(cf.view_name[iv]+' [cm]')

        print('view ', iv, ' from ', cf.view_offset[iv], ' to ', cf.view_offset[iv]+cf.view_length[iv])
        axs[iv].set_xlim([cf.view_offset[iv], cf.view_offset[iv]+cf.view_length[iv]])
        axs[iv].set_ylim([cf.anode_z - v*cf.n_sample/cf.sampling, cf.anode_z])


        if(iv == 0):
            axs[iv].set_ylabel('Drift [cm]')
        elif(iv == cf.n_view-1):
            axs[iv].set_ylabel('Drift [cm]')
            axs[iv].yaxis.tick_right()
            axs[iv].yaxis.set_label_position("right")
        else:
            axs[iv].tick_params(labelleft=False)


    plt.subplots_adjust(top=0.9,
                        bottom=0.11,
                        left=0.1,
                        right=0.905,
                        hspace=0.3,
                        wspace=0.1)
    
    return fig, ax_col, axs


def plot_2dview_hits(max_adc=100, option=None, to_be_shown=False):
    
    fig, ax_col, axs = template_data_view()
    axs = draw_all_hits(axs, adc=True, cmap=cmap_ed, s=marker_size, vmin=0, vmax=max_adc)


    """ color bar """
    ax_col.set_title('Hit Max ADC')

    cb = fig.colorbar(axs[0].collections[0], cax=ax_col, orientation='horizontal')
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')

    run_nb = str(dc.evt_list[-1].run_nb)
    evt_nb = str(dc.evt_list[-1].trigger_nb)
    elec   = dc.evt_list[-1].elec

    if(option):
        option = "_"+option
    else:
        option = ""


    plt.savefig(cf.plot_path+'/hits_view'+option+'_'+elec+'_run_'+run_nb+'_evt_'+evt_nb+'.png')

    if(to_be_shown):
        plt.show()

    plt.close()



def plot_2dview_2dtracks(option=None, to_be_shown=False):
    fig, ax_leg, axs = template_data_view()
    
    """ all hits """
    axs = draw_all_hits(axs, c="#e6e6e6", s=marker_size, marker='o', label='Hits')

    
    
    """ 2D tracks """
    axs = draw_all_tracks(axs, linewidth=1, legend='2D Track')


    """ legend """
    ax_leg.axis('off')
    ax_leg.legend(*axs[0].get_legend_handles_labels(),loc='center', ncol=2, markerscale=4, markerfirst=True)    



    run_nb = str(dc.evt_list[-1].run_nb)
    evt_nb = str(dc.evt_list[-1].trigger_nb)
    elec   = dc.evt_list[-1].elec

    if(option):
        option = "_"+option
    else:
        option = ""

    plt.savefig(cf.plot_path+'/track2D'+option+'_'+elec+'_run_'+run_nb+'_evt_'+evt_nb+'.png')
    if(to_be_shown):
        plt.show()
    plt.close()

