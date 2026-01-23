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
import itertools as itr
import math
import colorcet as cc
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.cbook import flatten

color = ['#FBA120', '#435497', '#df5286']

def correct_path_orientation(path):
    if(cf.tpc_orientation == 'Vertical'):
        return path
    else:
        #print(path)
        #newpath = [(x[0], x[2], x[1]) for x in path]
        #newpath = [(x[1], x[2], x[0]) for x in path]
        newpath = [(x[2], x[1], x[0]) for x in path]
        
        return newpath

def plot_3d(option=None, to_be_shown=True):


    v = lar.drift_velocity()
    if(cf.tpc_orientation == 'Vertical'):        
        xmin, xmax = min(min(cf.x_boundaries))-10, max(max(cf.x_boundaries))+10
        ymin, ymax = min(min(cf.y_boundaries))-10, max(max(cf.y_boundaries))+10
        zmin, zmax = -500, 500#min(cf.anode_z) - v*max(cf.n_sample)/cf.sampling[0], max(cf.anode_z)
        
        xlabel, ylabel, zlabel = 'x', 'y', 'Drift/z'

        bmin, bmax = ymin, ymax
        cmin, cmax = zmin, zmax
        amin, amax = xmin, xmax
        
        xlabel, ylabel, zlabel = 'x','y','Drift/z'
        alabel, blabel, clabel = xlabel, ylabel, ylabel



        
    elif(cf.tpc_orientation == 'Horizontal'):


        xmin, xmax = min(min(cf.x_boundaries)), max(max(cf.x_boundaries))
        ymin, ymax = min(min(cf.y_boundaries)), max(max(cf.y_boundaries))
        zmin, zmax = min(cf.anode_z), max(cf.anode_z)
        

        bmin, bmax = ymin, ymax
        cmin, cmax = xmin, xmax
        amin, amax = zmin, zmax
        
        xlabel, ylabel, zlabel = 'x','y','Drift/z'
        alabel, blabel, clabel = zlabel, ylabel, xlabel
        
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(nrows = 2, ncols = 2)
    ax  = fig.add_subplot(gs[:,0], projection='3d')
    ax_xz = fig.add_subplot(gs[0, 1])
    if(cf.tpc_orientation == 'Vertical'):
        ax_yz = fig.add_subplot(gs[1,1], sharey=ax_xz)
    elif(cf.tpc_orientation == 'Horizontal'):
        ax_yz = fig.add_subplot(gs[1,1], sharex=ax_xz)


    
    for iv in range(cf.n_view):
        #a, b, c = list(flatten(get_3dtracks_x(iv))), list(flatten(get_3dtracks_y(iv))), list(flatten(get_3dtracks_z_corr(iv)))
        #x, y, z = list(flatten(get_3dtracks_x(iv))), list(flatten(get_3dtracks_y(iv))), list(flatten(get_3dtracks_z_corr(iv)))
        x, y, z = list(flatten(get_3dtracks_x(iv))), list(flatten(get_3dtracks_y(iv))), list(flatten(get_3dtracks_z(iv)))
        corr = list(flatten(get_3dtracks_corr(iv)))
        corr_color = [color[iv] if cc else 'grey' for cc in corr]
        
        if(cf.tpc_orientation == 'Vertical'):
            #x, y, z = a, b, c
            a, b, c = x, y, z
            
        elif(cf.tpc_orientation == 'Horizontal'):
            ###x, y, z = a, c, b            
            #x, y, z = b, c, a
            a, b, c = z, y, x

            
        if(len(x) == 0):
            continue
        #xini, yini,zini = [t.ini_x for t in dc.tracks3D_list], [t.ini_y for t in dc.tracks3D_list], [t.ini_z+t.z0_corr for t in dc.tracks3D_list]
        #xend, yend,zend = [t.end_x for t in dc.tracks3D_list], [t.end_y for t in dc.tracks3D_list], [t.end_z+t.z0_corr for t in dc.tracks3D_list]

        xini, yini, zini = [t.ini_x for t in dc.tracks3D_list], [t.ini_y for t in dc.tracks3D_list], [t.ini_z for t in dc.tracks3D_list]
        xend, yend, zend = [t.end_x for t in dc.tracks3D_list], [t.end_y for t in dc.tracks3D_list], [t.end_z for t in dc.tracks3D_list]
        mod_ini = [t.module_ini for t in dc.tracks3D_list]
        mod_end = [t.module_end for t in dc.tracks3D_list]
        color_module = ['blue','tab:cyan','red','tab:orange']
        
        t_ID = [t.ID_3D for t in dc.tracks3D_list]
        
        ax.scatter(a, b, c, c=corr_color, s=4)
        if(cf.tpc_orientation == 'Vertical'):
            ax.scatter(xini, yini, zini, marker="*", c='r', s=20)
            ax.scatter(xend, yend, zend, marker="p", c='k', s=20)    

        elif(cf.tpc_orientation == 'Horizontal'):
                ax.scatter(zini, yini, xini, marker="*", c='r', s=20)
                ax.scatter(zend, yend, xend, marker="*", c='k', s=20)    

        if(cf.tpc_orientation == 'Vertical'):
            ax_xz.scatter(x, z, c=corr_color, s=4)  
            ax_yz.scatter(y, z, c=corr_color, s=4)  
            ax_xz.axvline(0,ls='dotted',c='k',lw=1)

            ax_xz.scatter(xini, zini, c=[color_module[m] for m in mod_ini], marker='*', s=20)
            ax_xz.scatter(xend, zend, c=[color_module[m] for m in mod_ini], marker='p', s=20)  
            ax_yz.scatter(yini, zini, c=[color_module[m] for m in mod_ini], marker='*', s=20)
            ax_yz.scatter(yend, zend, c=[color_module[m] for m in mod_ini], marker='p', s=20)  

            
            for xi,yi,zi,xe,ye,ze,m  in zip(xini, yini, zini, xend, yend, zend, mod_ini):
                col = 'b' if m < 2 else 'k'
                ax_xz.plot([xi,xe],[zi,ze],c=col,ls='dashed',lw=1)
                ax_yz.plot([yi,ye],[zi,ze],c=col,ls='dashed',lw=1)
                #ax_xz.plot([xini, xend], [zini, zend], c=['tab:olive' if m < 2 else 'tab:pink' for m in mod_ini], ls='dashed', lw=1)


                #ax_yz.plot([yini, yend], [zini, zend], c=['tab:olive' if m < 2 else 'tab:pink' for m in mod_ini], ls='dashed', lw=1)

            
            for zi, xi, yi, ti in zip(zini, xini, yini, t_ID):
                ax_xz.text(xi, zi, str(ti))
                ax_yz.text(yi, zi, str(ti))

            
        elif(cf.tpc_orientation == 'Horizontal'):
            ax_xz.scatter(z, x, c=corr_color, s=4)  
            ax_yz.scatter(z, y, c=corr_color, s=4)
            
            ax_xz.axvline(0,ls='dotted',c='k',lw=1)
            ax_yz.axvline(0,ls='dotted',c='k',lw=1)
            ax_xz.scatter(zini, xini, c='r', marker='*', s=20)
            ax_xz.scatter(zend, xend, c='k', marker='*', s=20)  
            ax_xz.plot([zini, zend], [xini, xend], c='tab:cyan', ls='dashed', lw=1)

            ax_yz.scatter(zini, yini, c='r', marker='*', s=20)
            ax_yz.scatter(zend, yend, c='k', marker='*', s=20)
            ax_yz.plot([zini, zend], [yini, yend], c='tab:cyan', ls='dashed', lw=1)        


            for zi, xi, yi, ti in zip(zini, xini, yini, t_ID):
                ax_xz.text(zi,xi,str(ti))
                ax_yz.text(zi,yi,str(ti))



                
        
    """ single hits"""
    sh = get_3dsingle_hits()
    
    if(len(sh)>0):
        if(cf.tpc_orientation == 'Vertical'):
            ax.scatter(*zip(*correct_path_orientation(sh)), c='k', s=6)
            ax_xz.scatter([x[0] for x in sh], [x[2] for x in sh], c='k', s=6)
            ax_yz.scatter([x[1] for x in sh], [x[2] for x in sh], c='k', s=6)
        
        elif(cf.tpc_orientation == 'Horizontal'):
            ax.scatter(*zip(*correct_path_orientation(sh)), c='k', s=6)
            ax_xz.scatter([x[2] for x in sh], [x[0] for x in sh], c='k', s=6)
            ax_yz.scatter([x[2] for x in sh], [x[1] for x in sh], c='k', s=6)
        
    """ghosts"""
    ghost = correct_path_orientation(get_3dghost())
    if(len(ghost)>0):
        ax.scatter(*zip(*ghost), c='silver', s=5)


    ax.set_xlim3d(amin, amax)
    ax.set_ylim3d(bmin, bmax)
    ax.set_zlim3d(cmin, cmax)

    ax.set_xlabel(alabel+' [cm]')
    ax.set_ylabel(blabel+' [cm]')
    ax.set_zlabel(clabel+' [cm]')





    if(cf.tpc_orientation == 'Vertical'):
        ax_xz.set_ylim(zmin, zmax)
        ax_yz.set_ylim(zmin, zmax)
        ax_xz.set_xlim(xmin, xmax)
        ax_yz.set_xlim(ymin, ymax)

    
        ax_xz.set_ylabel(zlabel+' [cm]')
        ax_yz.set_ylabel(zlabel+' [cm]')
        ax_xz.set_xlabel(xlabel+' [cm]')
        ax_yz.set_xlabel(ylabel+' [cm]')


    elif(cf.tpc_orientation == 'Horizontal'):
        ax_xz.set_xlim(zmin, zmax)
        ax_yz.set_xlim(zmin, zmax)
        ax_xz.set_ylim(xmin, xmax)
        ax_yz.set_ylim(ymin, ymax)

    
        ax_xz.set_xlabel(zlabel+' [cm]')
        ax_yz.set_xlabel(zlabel+' [cm]')
        ax_xz.set_ylabel(xlabel+' [cm]')
        ax_yz.set_ylabel(ylabel+' [cm]')

    

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
    

    plt.subplots_adjust(top=0.99,
                        bottom=0.1,
                        left=0.05,
                        right=0.95,
                        hspace=0.235,
                        wspace=0.2)



    save_with_details(fig, option, 'track3D', is3D=False)#True)

    if(to_be_shown):
        plt.show()
    plt.close()



def compare_3D():

    fig = plt.figure()
    ax_xy = fig.add_subplot(131)
    ax_xz = fig.add_subplot(132)
    ax_yz = fig.add_subplot(133)

    """ off. method """
    color='k'
    for iv in range(3):
        x_off, y_off, z_off = list(flatten(get_3dtracks_x(iv))), list(flatten(get_3dtracks_y(iv))), list(flatten(get_3dtracks_z(iv)))
            
        ax_xy.scatter(x_off, y_off, c=color, s=3)
        ax_xz.scatter(x_off, z_off, c=color, s=3)
        ax_yz.scatter(y_off, z_off, c=color, s=3)

    sh_off = get_3dsingle_hits()
    x_sh_off = [x[0] for x in sh_off]
    y_sh_off = [x[1] for x in sh_off]
    z_sh_off = [x[2] for x in sh_off]

    ax_xy.scatter(x_sh_off, y_sh_off, c=color, s=3)
    ax_xz.scatter(x_sh_off, z_sh_off, c=color, s=3)
    ax_yz.scatter(y_sh_off, z_sh_off, c=color, s=3)
    

    color='tab:cyan'
    x_new = [h.x_3D for h in dc.hits_list if h.has_3D]
    y_new = [h.y_3D for h in dc.hits_list if h.has_3D]
    z_new = [h.Z for h in dc.hits_list if h.has_3D]

    ax_xy.scatter(x_new, y_new, c=color, s=1, alpha=0.6)
    ax_xz.scatter(x_new, z_new, c=color, s=1, alpha=0.6)
    ax_yz.scatter(y_new, z_new, c=color, s=1, alpha=0.6)

    v = lar.drift_velocity()
    xmin, xmax = min(min(cf.x_boundaries)), max(max(cf.x_boundaries))
    ymin, ymax = min(min(cf.y_boundaries)), max(max(cf.y_boundaries))
    print('test', cf.n_sample)
    zmin, zmax = max(cf.anode_z) - v*cf.n_sample/cf.sampling, max(cf.anode_z)
    print("z range-->",zmin,zmax)
    ax_xy.set_xlim(xmin/2,xmax/2)
    ax_xy.set_ylim(ymin,ymax)

    ax_xz.set_xlim(xmin/2,xmax/2)
    ax_xz.set_ylim(zmin,zmax)

    ax_yz.set_xlim(ymin,ymax)
    ax_yz.set_ylim(zmin,zmax)

    plt.show()
    
def  show_shadows():  
    """ shadow on the walls """

    """ kept for memory, not used atm """
    ax.scatter(x,
               [cf.view_offset[1] for i in y], 
               z, 
               c="#f2f2f2", s=4)
    
    
    ax.scatter([300 for i in x1], 
               y1,
               z1,
               c="#f2f2f2", s=4)
    

