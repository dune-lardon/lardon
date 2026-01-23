import lardon.config as cf
import lardon.data_containers as dc
import lardon.lar_param as lar

from lardon.plotting.select_hits import *
from lardon.plotting.save_plot import *

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import collections  as mc
from matplotlib.legend_handler import HandlerTuple
import matplotlib.lines as mlines
import itertools as itr
import math
import colorcet as cc


cmap_ed = cc.cm.kbc_r
marker_size = 5

color_noise      = "#c7c7c7"
color_singlehits = "#ffb77d"
color_matched1   = "#28568f"
color_matched2   = "#abdf7f"
color_track2d    = "#de425b"
color_track3d    = "#00a9b2"
color_ghost      = "#bd96d0"


def draw_hits(pos, time, z=[], ax=None, **kwargs):

    ax = plt.gca() if ax is None else ax
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



def draw_tracks(pos, time, ids=[], ax=None, legend="", **kwargs):

    ax = plt.gca() if ax is None else ax

    
    if(len(pos) == 0):
        return ax

    if(len(legend)>0):
        ax.plot(pos[0], time[0], label=legend, **kwargs)

    if(len(ids)==0):
        for tx,tz in zip(pos, time):
            ax.plot(tx,tz, **kwargs)
    else:
        for tx,tz,ii in zip(pos, time,ids):
            ax.plot(tx,tz, **kwargs)
            ax.text(tx[0],tz[0],str(ii))
            
    
    return ax


def draw_all_tracks(axs, sel='True', legend="", **kwargs):

    for iview in range(cf.n_view):
        axs[iview] = draw_tracks(pos=get_2dtracks_pos(iview,sel), 
                                 time=get_2dtracks_z(iview,sel),
                                 ids = get_2dtracks_ID(iview,sel),
                                 ax=axs[iview], 
                                 legend=legend,
                                 **kwargs)
        #axs[iview].axvline(0, c='k', lw=0.5, ls='dotted')
    return axs





def template_data_view(modules):


    fig = plt.figure(figsize=(10,5))
    gs = gridspec.GridSpec(nrows=2, ncols=cf.n_view, 
                           height_ratios=[1,20], 
                           width_ratios=cf.view_length)
    

                           
    ax_col = fig.add_subplot(gs[0,:])

    axs = [fig.add_subplot(gs[1, iv]) for iv in range(cf.n_view)]
    v = lar.drift_velocity()
    [axs[i].sharey(axs[0]) for i in range(1, cf.n_view)]


    mod_name = ', '.join([cf.module_name[x] for x in modules])
    
    ymin = min(min([(cf.anode_z[x], cf.anode_z[x]- cf.drift_direction[x]*v*cf.n_sample[x]/cf.sampling[x]) for x in modules]))
    ymax = max(max([(cf.anode_z[x], cf.anode_z[x]- cf.drift_direction[x]*v*cf.n_sample[x]/cf.sampling[x]) for x in modules]))


        
    for iv in range(cf.n_view):
        xmin = min([cf.view_boundaries_min[x][iv] for x in modules])
        xmax = max([cf.view_boundaries_max[x][iv] for x in modules])


        
        axs[iv].set_title(mod_name+' - View '+str(iv)+"/"+cf.view_name[iv])
        axs[iv].set_xlabel(cf.view_name[iv]+' [cm]')

        axs[iv].set_xlim(xmin, xmax)                      
        axs[iv].set_ylim(ymin, ymax)
        

        if(iv == 0):
            axs[iv].set_ylabel('Drift [cm]')
        elif(iv == cf.n_view-1):
            axs[iv].set_ylabel('Drift [cm]')
            axs[iv].yaxis.tick_right()
            axs[iv].yaxis.set_label_position("right")
        else:
            axs[iv].tick_params(labelleft=False)


    plt.subplots_adjust(top=0.87,
                        bottom=0.11,
                        left=0.1,
                        right=0.905,
                        hspace=0.3,
                        wspace=0.1)
    
    return fig, ax_col, axs
    
def plot_2dview_hits(modules, max_adc=100, option=None, to_be_shown=False):
    
    fig, ax_col, axs = template_data_view(modules)

    selection = 'x.module == '+' or x.module == '.join([str(x) for x in modules])

    axs = draw_all_hits(axs, adc=True, cmap=cmap_ed, s=marker_size, vmin=0, vmax=max_adc, sel=selection)


    """ color bar """
    ax_col.set_title('Hit Max ADC')

    for i in range(cf.n_view):
        try :
            cb = fig.colorbar(axs[i].collections[0], cax=ax_col, orientation='horizontal')
            cb.ax.xaxis.set_ticks_position('top')
            cb.ax.xaxis.set_label_position('top')
            break
        except: 
            print('no hits in view ', i)


    save_with_details(fig, option, 'hits_view')
    #plt.savefig(cf.plot_path+'/hits_view'+option+'_'+elec+'_run_'+run_nb+'_evt_'+evt_nb+'.png')

    if(to_be_shown):
        plt.show()

    plt.close()



def plot_2dview_2dtracks(modules, option=None, to_be_shown=False):
    fig, ax_leg, axs = template_data_view(modules)

    hit_selection = 'x.module == '+' or x.module == '.join([str(x) for x in modules])
    
    """ all hits """
    axs = draw_all_hits(axs, c="#e6e6e6", s=marker_size, marker='o', label='Hits', sel=hit_selection)
    
    trk_selection = 't.module_ini == '+' or t.module_ini == '.join([str(x) for x in modules])
    trk_selection += ' or t.module_end =='+' or t.module_end == '.join([str(x) for x in modules])
    

    
    """ 2D tracks """
    axs = draw_all_tracks(axs, linewidth=1, legend='2D Track', sel=trk_selection)


    """ legend """
    ax_leg.axis('off')
    ax_leg.legend(*axs[0].get_legend_handles_labels(),loc='center', ncol=2, markerscale=4, markerfirst=True)    


    save_with_details(fig, option, 'track2D_hits')

    if(to_be_shown):
        plt.show()
    plt.close()


def plot_2dview_hits_tracks(modules, draw_2D=True, draw_3D=True, option=None, to_be_shown=False):
    fig, ax_leg, axs = template_data_view(modules)
    #save="tracks2D_hits_type"

    leg_handle = []
    leg_label = []
    
    """ unmatched hits """
    sel = 'x.ID >=0 and (x.module == '+' or x.module == '.join([str(x) for x in modules])+')'
    
    axs = draw_all_hits(axs, sel, c=color_noise, s=marker_size, marker='o', label='Noise Hits')
    leg_handle.append(mlines.Line2D([], [], color=color_noise, marker='o', linestyle='None', markersize=10, label='Noise'))


    """ hits attached to track """
    sel = 'x.match_2D >= 0  and (x.module == '+' or x.module == '.join([str(x) for x in modules])+')'
    axs = draw_all_hits(axs, sel, c=color_matched1, s=marker_size, marker='o', label='Hits Attached to Track')
    leg_handle.append(mlines.Line2D([], [], color=color_matched1, marker='o', linestyle='None', markersize=10, label='Track Hits'))

    """ delta_ray hits attached to track """
    sel = 'x.match_dray >=0  and (x.module == '+' or x.module == '.join([str(x) for x in modules])+')'
    axs = draw_all_hits(axs, sel, c=color_matched2, s=marker_size, marker='o', label='Delta Rays')
    leg_handle.append(mlines.Line2D([], [], color=color_matched2, marker='o', linestyle='None', markersize=10, label=r'$\delta_{ray}$'))


    """ single hits """
    sel = 'x.match_sh >=0  and (x.module == '+' or x.module == '.join([str(x) for x in modules])+')'
    axs = draw_all_hits(axs, sel, c=color_singlehits, s=marker_size, marker='o', label='Single Hits')    
    leg_handle.append(mlines.Line2D([], [], color=color_singlehits, marker='o', linestyle='None', markersize=10, label='Single'))  

    if(draw_2D):
        trk_sel = 't.module_ini == '+' or t.module_ini == '.join([str(x) for x in modules])
        trk_sel += ' or t.module_end =='+' or t.module_end == '.join([str(x) for x in modules])

        """ 2D tracks """

        sel = 't.ghost == False and ('+trk_sel+')'

        axs = draw_all_tracks(axs, sel, legend='2D Track', c=color_track2d, linewidth=1)
        leg_handle.append(mlines.Line2D([], [], color=color_track2d, linestyle='solid', lw=3, label='Track 2D'))     


        sel = 'np.fabs(t.ini_slope) > 50 and np.fabs(t.end_slope) > 50 and ('+trk_sel+')'

        axs = draw_all_tracks(axs, sel, legend='Bad 2D Track', c='k', linewidth=1)
        leg_handle.append(mlines.Line2D([], [], color='k', linestyle='solid', lw=3, label='Bad Track 2D'))     


        sel = 't.ghost == True and ('+trk_sel+')'
        axs = draw_all_tracks(axs, sel, legend='Ghost', c=color_ghost, linewidth=1)

        leg_handle.append(mlines.Line2D([], [], color=color_ghost, linestyle='solid', lw=3, label='Ghost'))     

    if(draw_3D):

        trk_sel = 't.module_ini == '+' or t.module_ini == '.join([str(x) for x in modules])
        trk_sel += ' or t.module_end =='+' or t.module_end == '.join([str(x) for x in modules])
        """ 3D tracks """
        sel = 't.match_3D >= 0 and ('+trk_sel+')'
        axs = draw_all_tracks(axs, sel, c=color_track3d, linewidth=2, legend='3D Track')
        save="tracks3D_hits_type"
        leg_handle.append(mlines.Line2D([], [], color=color_track3d, linestyle='solid', lw=3, label='Track 3D'))

    """ legend """
    ax_leg.axis('off')
    
    """ re-arrange the legend (line last), and merge blue and green entries """
    """ might not work anymore if the plotting order is changed """


    if(False):
        leg = ax_leg.legend([h[1], h[2], (h[3], h[4]), h[0]], [l[1], l[2], 'Hits Attached to Track (1,2)', l[0]], loc='center', ncol=4, markerscale=4, handler_map={tuple: HandlerTuple(ndivide=None)})
    else:
        """otherwise this works well """
        leg = ax_leg.legend(handles=leg_handle, ncol=5 , frameon=False, bbox_to_anchor=(0.0, 0., 1., 1.), loc='center') #int(len(leg_handle)/2)


            #*axs[0].get_legend_handles_labels(),loc='center', ncol=5, markerscale=4, markerfirst=True)
    
    """ make line in the legend bigger """
    #for line in leg.get_lines():
        #line.set_linewidth(3)

    save_with_details(fig, option, "reco2D_module_"+str(modules[0]))

    if(to_be_shown):
        plt.show()
    plt.close()


def plot_2dview_hits_2dtracks(modules, option=None, to_be_shown=False):
    return plot_2dview_hits_tracks(modules, draw_2D=True, draw_3D=False, option=option, to_be_shown=to_be_shown) 

def plot_2dview_hits_3dtracks(modules, option=None, to_be_shown=False):
    return plot_2dview_hits_tracks(modules, draw_2D=True, draw_3D=True, option=option, to_be_shown=to_be_shown) 
