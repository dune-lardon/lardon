import lardon.config as cf
import lardon.data_containers as dc
import lardon.lar_param as lar

from lardon.plotting.select_hits import *
from lardon.plotting.save_plot import *

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
import matplotlib.patches as patches
from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.lines import Line2D

color = ['#FBA120', '#435497', '#df5286']



def plot_timeline(option=None, to_be_shown=True):
    time_starts = [dc.evt_list[-1].pds_stream_time, dc.evt_list[-1].pds_trig_time, min(dc.evt_list[-1].charge_time)]
    time_starts = [(t-dc.evt_list[-1].event_time)*1e6 for t in time_starts]

    durations   = [cf.n_pds_stream_sample/cf.pds_sampling, cf.n_pds_trig_sample/cf.pds_sampling, max([n/s for n, s in zip(cf.n_sample,cf.sampling)])]
    start = min(time_starts)
    stop = max([s+d for s,d in zip(time_starts,durations)])

    
    pds_clusters_time = [p.timestamp for p in dc.pds_cluster_list]

    pds_clusters_size = [p.size for p in dc.pds_cluster_list]
    max_size = max(pds_clusters_size)
    
    trk_time = [t.timestamp for t in dc.tracks3D_list]
    trk_time_r = [t.timestamp_r for t in dc.tracks3D_list]
    trk_len = [max(t.len_straight) for t in  dc.tracks3D_list]
    trk_ano = [t.is_anode_crosser for t in   dc.tracks3D_list]
    trk_cat = [t.is_cathode_crosser for t in   dc.tracks3D_list]
    
    sh_time = [t.timestamp for t in dc.single_hits_list]


    fig = plt.figure(figsize=(14,4))
    ax = fig.add_subplot(111)

    for i in range(len(pds_clusters_time)):
        ax.plot([pds_clusters_time[i], pds_clusters_time[i]], [0, pds_clusters_size[i]], lw=2,c='k', alpha=0.6)
    ax.set_ylabel('PDS Cluster Size')
    ax.set_ylim(0, max(pds_clusters_size))
    
    axt = ax.twinx()
    colors = cc.glasbey_bw  # 
    for i in range(len(trk_time)):
        tstart = trk_time[i]
        dstop  = trk_time_r[i] - tstart
        c = colors[i % len(colors)]
        axt.plot([tstart, tstart],[0, trk_len[i]], lw=2, c=c, alpha=0.6)
        if(dstop > 0):
            rect = patches.Rectangle((tstart, 0), dstop, trk_len[i], alpha=.2, facecolor=c)
            axt.add_patch(rect)
        
        if(trk_ano[i]):
            axt.scatter(tstart, trk_len[i], marker='o', color = c)
        if(trk_cat[i]):
            axt.scatter(tstart, trk_len[i], marker='x', color = c)
            
    axt.set_ylim(0, max(trk_len))
    axt.set_ylabel('Track Length [cm]')
    
    axt.scatter(sh_time, [0.5 for x in range(len(dc.single_hits_list))], marker='*', color='yellow', label='Single Hits')

    legend_elements = [Line2D([0], [0], color='k', lw=2, alpha=0.6, label='PDS clusters'),
                       Line2D([0], [0], marker='o', color='tab:cyan', label='Anode crossers', markerfacecolor='tab:cyan', markersize=15),
                       Line2D([0], [0], marker='x', color='tab:cyan', label='Cathode crossers', markerfacecolor='tab:cyan', markersize=15),                    
                       patches.Patch(facecolor='tab:cyan', edgecolor='tab:cyan', alpha=0.5, label='Unresolved T0'),
                       Line2D([0], [0], marker='*', color='w', label='Blips', markerfacecolor='yellow', markersize=15)]

    
    ax.legend(handles=legend_elements, loc=(0.02, 1.02), ncols=5, frameon=False)
    ax.set_xlabel('Time wrt to trigger [mus]')
    
    plt.tight_layout()

    save_with_details(fig, option, 'Timeline')

    if(to_be_shown):
        plt.show()

    plt.close()
    




def plot_track_pds_matched(trk, option=None, to_be_shown=True):
    if(cf.tpc_orientation == 'Horizontal'):
        return
    
    v = lar.drift_velocity()


    cmap = cc.cm.linear_tritanopic_krw_5_95_c46_r
    vmin, vmax = 0, 1e6
    norm = colors.Normalize(vmin=vmin, vmax=vmax)


    clusID = trk.match_pds_cluster
    ID_clus_shift = dc.n_tot_pds_clusters
    clus = dc.pds_cluster_list[clusID-ID_clus_shift]
    
    trk.dump()
    #clus.dump()
    
    
    clus_ch = clus.glob_chans
    
       
    
    xmin, xmax = min(min(cf.x_boundaries)), max(max(cf.x_boundaries))
    ymin, ymax = min(min(cf.y_boundaries)), max(max(cf.y_boundaries))
    zmin, zmax = min(cf.anode_z), max(cf.anode_z)
    
    xlabel, ylabel, zlabel = 'x', 'y', 'Drift/z'


    fig = plt.figure(figsize=(6, 8))
    gs = gridspec.GridSpec(nrows = 3, ncols = 1, height_ratios=[1, 20, 5])
    ax  = fig.add_subplot(gs[1,0], projection='3d')
    ax_infos = fig.add_subplot(gs[0,0])
    ax_dist =  fig.add_subplot(gs[2,0])
    
    """ cathode plane """
    rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, alpha=.2, facecolor='gray')
    ax.add_patch(rect)
    art3d.pathpatch_2d_to_3d(rect, z=0, zdir="z")

    """ crp separation """
    ax.plot([0,0],[ymin,ymax], zs=zmax, zdir="z", c='k',ls='dashed')
    ax.plot([0,0],[ymin,ymax], zs=0., zdir="z", c='k',ls='dashed')
    ax.plot([0,0],[ymin,ymax], zs=zmin, zdir="z", c='k',ls='dashed')
    
    z0_corr = trk.z0_light
    if(z0_corr >= 9999):
        z0_corr = 0.0

    color = ['#FBA120', '#435497', '#df5286']
    for iv in range(3):
        pts = [p for p in trk.path[iv]]

        
        x,y,z = zip(*pts)
        z = [i+z0_corr for i in z]
    
        ax.scatter(x, y, z, c=color[iv], s=4)

    dist_charge = []

    trk_vol = int(trk.module_ini/cf.n_drift_volumes)
    
    for pt_trk, pt_pds, is_extr, dist, adc, gchan in zip(clus.point_closest[trk_vol], clus.point_impact[trk_vol], clus.point_closest_is_extrapolated[trk_vol], clus.dist_closest[trk_vol], clus.max_adcs, clus.glob_chans):

        tx, ty, tz = pt_trk

        pds_type = dc.chmap_pds[gchan].data_type
        
        if(is_extr == False):
            col, col_l='r', 'k'
        else:
            col, col_l='gray', 'gray'

        if(pds_type == "stream"):
            fc_col = col
        else:
            fc_col="none"
            
        ax.scatter(tx, ty, tz, c=col, s=3)
        px, py, pz = pt_pds
        ax.scatter(px, py, pz, c=col_l, s=3)

        ax.plot([tx,px], [ty,py],[tz,pz], c='k',lw=0.5, ls='dashed')
        
        ax_dist.scatter(dist, adc*1e-3, fc=fc_col, ec=col)

    ax_dist.set_xlabel('Track-PDS distance [cm]')
    ax_dist.set_ylabel('Peak max [kADC]')
    ax_dist.set_xlim(0, 1000)

    if(trk.cathode_crosser_ID >=0):
        trk_id_shift = dc.n_tot_trk3d
        other_trk = dc.tracks3D_list[trk.cathode_crosser_ID-trk_id_shift]
        other_z0_corr = other_trk.z0_light #other_trk.z0_corr
        for iv in range(3):
            pts = [p for p in other_trk.path[iv]]

            
            x,y,z = zip(*pts)
            z = [i+other_z0_corr for i in z]
            
            ax.scatter(x, y, z, c='gray', s=4, alpha=0.5)


    if(trk.is_anode_crosser and trk.exit_trk_end >=0):
        truth_from = [trk.ini_x, trk.ini_y, trk.ini_z+z0_corr] if trk.exit_trk_end == 1 else [trk.end_x, trk.end_y, trk.end_z+z0_corr]
        truth_to = trk.exit_point
        ax.plot([truth_from[0], truth_to[0]], [truth_from[1], truth_to[1]],[truth_from[2], truth_to[2]], c='r', ls='dotted')

        
    pds_max = np.argmax(clus.charges)

    pds_q_max = clus.charges[pds_max] 
    pds_ch_max = clus.glob_chans[pds_max]
    
    size_max = 20
    for pds_ch,pds_q in zip(clus.glob_chans, clus.charges):
        pds_module = dc.chmap_pds[pds_ch].module
        x_center = cf.pds_x_center[pds_module]
        y_center = cf.pds_y_center[pds_module]
        z_center = cf.pds_z_center[pds_module]

        x_length = cf.pds_x_length[pds_module]
        y_length = cf.pds_y_length[pds_module]
        z_length = cf.pds_z_length[pds_module]

        if(x_length==0):
            p = Circle((y_center, z_center), size_max*(pds_q/pds_q_max), color="tab:purple",alpha=0.5)
            ax.add_patch(p)
            art3d.pathpatch_2d_to_3d(p, z=x_center, zdir="x")
        elif(y_length==0):
            p = Circle((x_center, z_center), size_max*(pds_q/pds_q_max), color="tab:purple",alpha=0.5)
            ax.add_patch(p)
            art3d.pathpatch_2d_to_3d(p, z=y_center, zdir="y")
        elif(z_length==0):
            p = Circle((x_center, y_center), size_max*(pds_q/pds_q_max), color="tab:purple",alpha=0.5)
            ax.add_patch(p)
            art3d.pathpatch_2d_to_3d(p, z=z_center, zdir="z")


    for mod in range(cf.pds_n_modules):
        if(cf.pds_x_length[mod] == 0):
            a_low = cf.pds_y_center[mod]-cf.pds_y_length[mod]/2.
            b_low = cf.pds_z_center[mod]-cf.pds_z_length[mod]/2.
            a_width = cf.pds_y_length[mod]
            b_width = cf.pds_z_length[mod]
            center  = cf.pds_x_center[mod]
            zdir = "x"
        elif(cf.pds_y_length[mod] == 0):
            a_low = cf.pds_x_center[mod]-cf.pds_x_length[mod]/2.
            b_low = cf.pds_z_center[mod]-cf.pds_z_length[mod]/2.
            a_width = cf.pds_x_length[mod]
            b_width = cf.pds_z_length[mod]
            center  = cf.pds_y_center[mod]
            zdir = "y"

        elif(cf.pds_z_length[mod] == 0):
            a_low = cf.pds_x_center[mod]-cf.pds_x_length[mod]/2.
            b_low = cf.pds_y_center[mod]-cf.pds_y_length[mod]/2.
            a_width = cf.pds_x_length[mod]
            b_width = cf.pds_y_length[mod]
            center  = cf.pds_z_center[mod]
            zdir = "z"

        rect = patches.Rectangle((a_low, b_low), a_width, b_width, edgecolor='k', facecolor='none')
        
        ax.add_patch(rect)
        art3d.patch_2d_to_3d(rect, z=center, zdir=zdir)  # place in z=0 plane
        


    ax.set_xlim3d(xmin, xmax)
    ax.set_ylim3d(ymin, ymax)
    ax.set_zlim3d(zmin, zmax)

    ax.set_xlabel(xlabel+' [cm]')
    ax.set_ylabel(ylabel+' [cm]')
    ax.set_zlabel(zlabel+' [cm]')

    
    ax.grid(False)
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.zaxis.pane.set_edgecolor('black')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.view_init(elev=10, azim=-45)

    ax_infos.set_axis_off()
    ax_infos.text(0., 2., f'Track {trk.ID_3D} length {max(trk.len_straight):.1f} cm at {trk.timestamp:.3f} mus', ha='left')
    ax_infos.text(0., 1., f'PDS cluster {clus.ID} with {clus.size} peaks at {clus.timestamp:.3f} mus', ha='left')
    ax_infos.text(0., 0.0, f'-> Delay (PDS-TPC) = {clus.timestamp-trk.timestamp:3f} mus', ha='left')
        
    plt.show()
