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
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cbook import flatten

color = ['#FBA120', '#435497', '#df5286']

def plot_3d(option=None, to_be_shown=True):
    
    fig = plt.figure(figsize=(6, 6))
    ax  = fig.add_subplot(111, projection='3d')

    for iv in range(cf.n_view):
        x, y, z = list(flatten(get_3dtracks_x(iv))), list(flatten(get_3dtracks_y(iv))), list(flatten(get_3dtracks_z(iv)))
        
        if(len(x) == 0):
            continue
        
        """ shadow on the walls """
        """
        ax.scatter(x,
                   [cf.view_offset[1] for i in y], 
                   z, 
                   c="#f2f2f2", s=4)


        ax.scatter([300 for i in x1], 
                   y1,
                   z1,
                   c="#f2f2f2", s=4)

        """
        ax.scatter(x, 
                   y, 
                   z, 
                   c=color[iv], s=4)



    v = lar.drift_velocity()
    ax.set_xlim3d(cf.x_boundaries)
    ax.set_ylim3d(cf.y_boundaries)
    ax.set_zlim3d(cf.anode_z - v*cf.n_sample/cf.sampling, cf.anode_z)


    ax.set_xlabel('x [cm]')
    ax.set_ylabel('y [cm]')
    ax.set_zlabel('Drift/z [cm]')


    ax.grid(False)
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.zaxis.pane.set_edgecolor('black')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.view_init(elev=10, azim=135)


    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.zaxis.set_major_locator(plt.MaxNLocator(7))
    

    ax.xaxis._axinfo['tick']['inward_factor'] = 0.4
    ax.xaxis._axinfo['tick']['outward_factor'] = 0.4
    ax.yaxis._axinfo['tick']['inward_factor'] = 0.4
    ax.yaxis._axinfo['tick']['outward_factor'] = 0.4
    ax.zaxis._axinfo['tick']['outward_factor'] = 0.3
    ax.zaxis._axinfo['tick']['inward_factor'] = 0.3
    

    plt.subplots_adjust(top=0.95,
                        bottom=0.05,
                        left=0.05,
                        right=0.95,
                        hspace=0.2,
                        wspace=0.2)



    run_nb = str(dc.evt_list[-1].run_nb)
    evt_nb = str(dc.evt_list[-1].trigger_nb)
    elec   = dc.evt_list[-1].elec

    if(option):
        option = "_"+option
    else:
        option = ""


    plt.savefig(cf.plot_path+'/track3D'+option+'_elec_'+elec+'_run_'+run_nb+'_evt_'+evt_nb+'.png')
    if(to_be_shown):
        plt.show()
    plt.close()
