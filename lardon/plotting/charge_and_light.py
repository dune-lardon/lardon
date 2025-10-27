import config as cf
import data_containers as dc
import lar_param as lar

from .select_hits import *
from .save_plot import *

import numpy as np
import matplotlib as mpl
from matplotlib import colors

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import collections  as mc
from matplotlib.legend_handler import HandlerTuple
import itertools as itr
import math
import colorcet as cc
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cbook import flatten

color = ['#FBA120', '#435497', '#df5286']



def plot_track_pds_matched(trackID, option=None, to_be_shown=True):
    if(cf.tpc_orientation == 'Horizontal'):
        return
    
    v = lar.drift_velocity()


    cmap = cc.cm.linear_tritanopic_krw_5_95_c46_r
    vmin, vmax = 0, 1e6
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    ID_trk_shift =dc.n_tot_trk3d
    print('shift is ', ID_trk_shift)
    trk = dc.tracks3D_list[trackID-ID_trk_shift]
    trk.dump()

    clusID = trk.match_pds_cluster
    ID_clus_shift = dc.n_tot_pds_clusters
    clus = dc.pds_cluster_list[clusID-ID_clus_shift]
    
    clus.dump()

    clus_ch = clus.glob_chans
    
    
    
    
    xmin, xmax = min(min(cf.x_boundaries))-10, max(max(cf.x_boundaries))+10
    ymin, ymax = min(min(cf.y_boundaries))-10, max(max(cf.y_boundaries))+10
    zmin, zmax = -500, 500#min(cf.anode_z) - v*max(cf.n_sample)/cf.sampling[0], max(cf.anode_z)
    
    xlabel, ylabel, zlabel = 'x', 'y', 'Drift/z'

    #corr = list(flatten(get_3dtracks_corr(iv),'t.ID_3D=='+str(trackID)))
    

    print('track ', trackID)
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(nrows = 2, ncols = 1, height_ratios=[1, 20])
    ax  = fig.add_subplot(gs[1,0], projection='3d')

    z0_corr = trk.z0_corr

    print("--> ", z0_corr)
    if(z0_corr >= 9999):
        z0_corr = 0.0


    for iv in range(3):
        pts = [p for p in trk.path[iv]]
        print(pts)
        
        x,y,z = zip(*pts)
        z = [i+z0_corr for i in z]
    
        #x, y, z = list(flatten(get_3dtracks_x(iv),f't.ID_3D=={trackID}')), list(flatten(get_3dtracks_y(iv),f't.ID_3D=={trackID}')), list(flatten(get_3dtracks_z_corr(iv),f't.ID_3D=={trackID}'))
        #print('printing ', z0_corr)
        #print(z)
        ax.scatter(x, y, z, c='k', s=4)


    
     
    
    
    for ic in range(0, clus.size,2):

        glob_ch = clus.glob_chans[ic]
        charge = clus.charges[ic]
        color = cmap(norm(charge))

        ip = dc.chmap_pds[glob_ch].module
        print(ic, '=', glob_ch, ' charge ', charge, ' module ', ip)                    
        x0,y0,z0 = cf.pds_x_centers[ip], cf.pds_y_centers[ip], cf.pds_z_centers[ip]
        L = cf.pds_length
        h = L/2
        square = [
            [x0 - h, y0 - h, z0],
            [x0 + h, y0 - h, z0],
            [x0 + h, y0 + h, z0],
            [x0 - h, y0 + h, z0]]

        ax.add_collection3d(Poly3DCollection([square], color=color, alpha=0.5))


    for ip in range(cf.pds_n_modules):
        x0,y0,z0 = cf.pds_x_centers[ip], cf.pds_y_centers[ip], cf.pds_z_centers[ip]
        L = cf.pds_length
        h = L/2
        square = [
            [x0 - h, y0 - h, z0],
            [x0 + h, y0 - h, z0],
            [x0 + h, y0 + h, z0],
            [x0 - h, y0 + h, z0]]
        print(ip, square)
        ax.add_collection3d(Poly3DCollection([square], color='k', edgecolors='k', linewidths=1.5, alpha=0.001))




    ax.set_xlim3d(xmin, xmax)
    ax.set_ylim3d(ymin, ymax)
    ax.set_zlim3d(zmin, zmax)

    ax.set_xlabel(xlabel+' [cm]')
    ax.set_ylabel(ylabel+' [cm]')
    ax.set_zlabel(zlabel+' [cm]')






    plt.show()
