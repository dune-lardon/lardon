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


from scipy.special import gammaln


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

    if(len(dc.tracks3D_list) >0):
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
    


def draw_detector(ax):
    xmin, xmax = min(min(cf.x_boundaries)), max(max(cf.x_boundaries))
    ymin, ymax = min(min(cf.y_boundaries)), max(max(cf.y_boundaries))
    zmin, zmax = min(cf.anode_z), max(cf.anode_z)
    
    xlabel, ylabel, zlabel = 'x', 'y', 'Drift/z'

    """ cathode plane """
    rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, alpha=.2, facecolor='gray')
    ax.add_patch(rect)
    art3d.pathpatch_2d_to_3d(rect, z=0, zdir="z")

    """ crp separation """
    ax.plot([0,0],[ymin,ymax], zs=zmax, zdir="z", c='k',ls='dashed')
    ax.plot([0,0],[ymin,ymax], zs=0., zdir="z", c='k',ls='dashed')
    ax.plot([0,0],[ymin,ymax], zs=zmin, zdir="z", c='k',ls='dashed')


    """ plot limits """
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
    


def draw_pds_charge(ax, pds_module, pds_q, pds_q_max):
    size_max = 20
    
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

def draw_pds_geo(ax):
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


def draw_extrapolation(trk, ax):

    import lardon.track_timing as tmg
    P0, P1, tdir = tmg.track_point_direction(trk, trk.z0_light)
    tdir = tdir / np.linalg.norm(tdir)
    """
    print(trk.ID_3D, " z0 = ", trk.z0_light)
    print("cross check ", P0, " TO ", P1)

    print("full: from ", trk.extrap_ta_AV, " to ", trk.extrap_tb_AV)
    print("decay: from ", trk.extrap_ta_dk_AV, " to ", trk.extrap_tb_dk_AV)
    """
    
    full_ini = P0 + trk.extrap_ta_AV*tdir
    full_end = P0 + trk.extrap_tb_AV*tdir

    dk_ini = P0 + trk.extrap_ta_dk_AV*tdir
    dk_end = P0 + trk.extrap_tb_dk_AV*tdir


    print('Full extrapolation: ', full_ini, ' to ', full_end)
    print('Decay extrapolation: ', dk_ini, ' to ', dk_end)

    
    ax.plot([full_ini[0], full_end[0]], [full_ini[1], full_end[1]], [full_ini[2], full_end[2]], c='k', lw=1)
    ax.plot([dk_ini[0], dk_end[0]], [dk_ini[1], dk_end[1]], [dk_ini[2], dk_end[2]], c='r', lw=1)


    
def plot_track_pds_matched(trk, option=None, to_be_shown=True):
    if(cf.tpc_orientation == 'Horizontal'):
        return
    
    v = lar.drift_velocity()

    clusID = trk.match_pds_cluster
    ID_clus_shift = dc.n_tot_pds_clusters
    clus = dc.pds_cluster_list[clusID-ID_clus_shift]
    
    #trk.dump()
    #clus.dump()
        
    clus_ch = clus.glob_chans
    
    fig = plt.figure(figsize=(6, 8))
    gs = gridspec.GridSpec(nrows = 3, ncols = 1, height_ratios=[1, 20, 5])
    ax  = fig.add_subplot(gs[1,0], projection='3d')
    ax_infos = fig.add_subplot(gs[0,0])
    ax_dist =  fig.add_subplot(gs[2,0])

    
    draw_detector(ax)
    draw_pds_geo(ax)

    draw_extrapolation(trk, ax)
    
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
    
    for pt_trk, pt_pds, dist, pred, adc, q, gchan in zip(clus.point_closest[trk_vol], clus.point_impact[trk_vol], clus.dist_closest[trk_vol], clus.n_predicted[trk_vol], clus.max_npes,clus.npes, clus.glob_chans):

        tx, ty, tz = pt_trk

        pds_module = dc.chmap_pds[gchan].module
        pds_det = cf.pds_modules_type[pds_module]

        fc_col_p = 'none'
        
        if(pds_det == "Cathode"):
            col = 'r'
            fc_col = col
            col_p =  'tab:pink'

        elif(pds_det == "Membrane"):
            col = 'tab:blue'
            fc_col = col
            col_p = 'tab:cyan'

        else:
            col = 'tab:green'
            fc_col = col
            col_p = 'tab:olive'

        col_l = 'k'

        
        ax.scatter(tx, ty, tz, c=col, s=3)
        px, py, pz = pt_pds

        ax.scatter(px, py, pz, c=col_l, s=3)

        ax.plot([tx,px], [ty,py], [tz,pz], c='k',lw=0.5, ls='dashed')

        ax_dist.scatter(dist, q*1e-3, fc=fc_col, ec=col)
        ax_dist.scatter(dist, pred*1e-3, fc="none", ec=col_p)

        #print(gchan, ' pred ', pred, ' meas ', q, "=",q* np.log(pred) - pred - gammaln(q + 1))
    ax_dist.set_xlabel('Track-PDS distance [cm]')
    ax_dist.set_ylabel('Light Charge [kNPE]')

    ax_dist.set_xlim(0, 1000)


    
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Data',
                              mfc='k', markersize=10),
                       Line2D([0], [0], marker='o', color='w', label='Predicted',
                              mfc='w', mec='k', markersize=10),
                       patches.Patch(facecolor='r', edgecolor='tab:pink',
                                     label='C-PD'),
                       patches.Patch(facecolor='tab:blue', edgecolor='tab:cyan',
                                     label='M-PD'),
                       patches.Patch(facecolor='tab:green', edgecolor='tab:olive',
                                     label='PMT')#,
                       #patches.Patch(facecolor='gray', edgecolor='gray',
                       #              label='Saturates')
                       ]
    ax_dist.legend(handles=legend_elements, loc=(0.02, 1.02), frameon=False, ncols=5,fontsize="x-small")    


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
    
    for pds_ch,pds_q in zip(clus.glob_chans, clus.charges):
        pds_module = dc.chmap_pds[pds_ch].module
        draw_pds_charge(ax, pds_module, pds_q, pds_q_max)            



    
    ax_infos.set_axis_off()
    if(trk.is_decay_from_light):
        decay_mess = '[decay hyp]'
    else:
        decay_mess = ''
    if(trk.t0_corr >= 9999.):
        ax_infos.text(0., 2., f'Track {trk.ID_3D} length {max(trk.len_straight):.1f} cm had no t0', ha='left')
        ax_infos.text(0., 1., f'PDS cluster {clus.ID} with {clus.size} peaks at {clus.timestamp:.3f} mus', ha='left')
        ax_infos.text(0., 0.0, f'-> -logL = {trk.logL_pds_cluster:.1f} '+decay_mess, ha='left')
    else:
        ax_infos.text(0., 2., f'Track {trk.ID_3D} length {max(trk.len_straight):.1f} cm at {trk.timestamp:.3f} mus', ha='left')
        ax_infos.text(0., 1., f'PDS cluster {clus.ID} with {clus.size} peaks at {clus.timestamp:.3f} mus', ha='left')
        ax_infos.text(0., 0.0, f'-> Delay (PDS-TPC) = {clus.timestamp-trk.timestamp:.3f} mus, -logL = {trk.logL_pds_cluster:.1f} '+decay_mess, ha='left')
    
    plt.show()






def plot_track_pds_test(trk, z0_light, clus, dists, points, preds, logL, option=None, to_be_shown=True):
    if(cf.tpc_orientation == 'Horizontal'):
        return
    
    v = lar.drift_velocity()

    
    trk.dump()
    clus.dump()
    
    print('\nNEW TRACK END POINTS with z0=', z0_light)
    print("ini ", [trk.ini_x, trk.ini_y, trk.ini_z+z0_light])
    print("to ", [trk.end_x, trk.end_y, trk.end_z+z0_light],"\n")
    
    clus_ch = clus.glob_chans


    fig = plt.figure(figsize=(6, 8))
    gs = gridspec.GridSpec(nrows = 3, ncols = 1, height_ratios=[1, 20, 5])
    ax  = fig.add_subplot(gs[1,0], projection='3d')
    ax_infos = fig.add_subplot(gs[0,0])
    ax_dist =  fig.add_subplot(gs[2,0])


    
    draw_detector(ax)
    draw_pds_geo(ax)
    draw_extrapolation(trk, ax)    
    z0_corr = z0_light
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
    pds_ped = [dc.evt_list[-1].noise_pds_raw.ped_mean[dc.chmap_pds[gch].daqch] for gch in range(cf.n_pds_tot_channels)]

    for pt_trk, dist, pred, adc, q, gchan in zip(points, dists, preds, clus.max_adcs, clus.npes, clus.glob_chans):

        tx, ty, tz = pt_trk


        pds_module = dc.chmap_pds[gchan].module
        pds_det = cf.pds_modules_type[pds_module]

        pt_pds = np.array([cf.pds_x_center[pds_module], cf.pds_y_center[pds_module], cf.pds_z_center[pds_module]])

        fc_col_p = 'none'
        
        if(pds_det == "Cathode"):
            col = 'r'
            fc_col = col
            col_p =  'tab:pink'

            print(gchan, ' pred ', pred, ' meas ', q, "=",q* np.log(pred) - pred - gammaln(q + 1))
        elif(pds_det == "Membrane"):
            col = 'tab:blue'
            fc_col = col
            col_p = 'tab:cyan'
        else:
            col = 'tab:green'
            fc_col = col
            col_p = 'tab:olive'

        col_l = 'k'

        ax.scatter(tx, ty, tz, c=col, s=3)
        px, py, pz = pt_pds
        ax.scatter(px, py, pz, c=col_l, s=3)

        ax.plot([tx,px], [ty,py],[tz,pz], c='k',lw=0.5, ls='dashed')

        if(adc >= pow(2, 14)-pds_ped[gchan]-10):
            #print(gchan, 'at ', dist, ' saturates: ', adc, " => ", q, "npe")
            fc_col = 'gray'
            
        ax_dist.scatter(dist, q*1e-3, fc=fc_col, ec="none", alpha=0.6)
        ax_dist.scatter(dist, pred*1e-3, fc=fc_col_p, ec=col_p)
        
    ax_dist.set_xlabel('Track-PDS distance [cm]')
    ax_dist.set_ylabel('Light Charge [kNPE]')
    ax_dist.set_xlim(0, 1000)

    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Data',
                              mfc='k', markersize=10),
                       Line2D([0], [0], marker='o', color='w', label='Predicted',
                              mfc='w', mec='k', markersize=10),
                       patches.Patch(facecolor='r', edgecolor='tab:pink',
                                     label='C-PD'),
                       patches.Patch(facecolor='b', edgecolor='tab:cyan',
                                     label='M-PD'),
                       patches.Patch(facecolor='tab:green', edgecolor='tab:olive',
                                     label='PMT'),
                       patches.Patch(facecolor='gray', edgecolor='gray',
                                     label='Saturates')
                       ]
    ax_dist.legend(handles=legend_elements, loc=(0.02, 1.02), frameon=False, ncols=6,fontsize="x-small")    
    
    pds_max = np.argmax(clus.charges)
    pds_q_max = clus.charges[pds_max] 

    
    size_max = 20
    for pds_ch,pds_q in zip(clus.glob_chans, clus.charges):
        pds_module = dc.chmap_pds[pds_ch].module
        draw_pds_charge(ax, pds_module, pds_q, pds_q_max)
        
    ax_infos.set_axis_off()
    ax_infos.text(0., 2., f'Track {trk.ID_3D} length {max(trk.len_straight):.1f} cm', ha='left')
    ax_infos.text(0., 1., f'PDS cluster {clus.ID} with {clus.size} peaks at {clus.timestamp:.3f} mus', ha='left')
    ax_infos.text(0., 0.0, f'-> logL = {logL}', ha='left')
        
    plt.show()


def plot_multiple_track_pds_test(trks, z0_lights, clus, logLs, option=None, to_be_shown=True):
    if(cf.tpc_orientation == 'Horizontal'):
        return
    
    v = lar.drift_velocity()
    
    clus_ch = clus.glob_chans


    fig = plt.figure(figsize=(6, 6))
    gs = gridspec.GridSpec(nrows = 2, ncols = 1, height_ratios=[1, 20])
    ax  = fig.add_subplot(gs[1,0], projection='3d')
    ax_infos = fig.add_subplot(gs[0,0])



    
    draw_detector(ax)
    draw_pds_geo(ax)
    
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=cc.glasbey_category10)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    legend_elements=[]
    for trk , z0, c, score in zip(trks,z0_lights, colors, logLs):
        for iv in range(3):
            pts = [p for p in trk.path[iv]]
        
            x,y,z = zip(*pts)
            z = [i+z0 for i in z]            
            ax.scatter(x, y, z, c=c, s=4)

    
        legend_elements.append(Line2D([0], [0], color=c, lw=2, label=f'Track {trk.ID_3D} logL={score:.1f}'))

    if(len(trks)>6):
        ncols=2
    else:
        ncols=1
    ax_infos.legend(handles=legend_elements, loc='upper left', frameon=False, fontsize="medium",ncols=ncols)

    pds_max = np.argmax(clus.charges)
    pds_q_max = clus.charges[pds_max] 

    size_max = 20
    for pds_ch,pds_q in zip(clus.glob_chans, clus.charges):
        pds_module = dc.chmap_pds[pds_ch].module
        draw_pds_charge(ax, pds_module, pds_q, pds_q_max)
        
    ax_infos.set_axis_off()

    plt.show()
