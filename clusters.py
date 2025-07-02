import config as cf
import data_containers as dc
import lar_param as lar

from sklearn.cluster import DBSCAN
from collections import Counter
import numpy as np
from rtree import index

import time




def build_pds_cluster(peaks, idx):
    cluster_ID = idx
    IDs = [p.ID for p in peaks]
    glob_chans = [p.glob_ch for p in peaks]
    channels = [p.channel for p in peaks]
    t_starts = [p.start for p in peaks]
    t_maxs = [p.max_t for p in peaks]
    t_stops = [p.stop for p in peaks]
    max_adcs = [p.max_adc for p in peaks]
    charges = [p.charge for p in peaks]
    
    cluster = dc.pds_cluster(cluster_ID, IDs, glob_chans, channels, t_starts, t_maxs, t_stops, max_adcs, charges)

    [p.set_cluster_ID(idx) for p in peaks]
    
    return cluster

def light_clustering():

    cluster_list = []
    
    if(len(dc.pds_peak_list) <2 ):
        return

    time_tol = dc.reco['pds']['cluster']['time_tol'] #in ticks

    id_peak_shift = dc.n_tot_pds_peaks
    id_cluster_shift = dc.n_tot_pds_clusters

    
    pties = index.Property()
    pties.dimension = 3

    ''' create an rtree index (channel, pds peak times)'''
    rtree_idx = index.Index(properties=pties)

    ''' filling the R-tree '''
    for p in dc.pds_peak_list:

        start = p.start
        chan  = p.glob_ch
        readout = chan%2
        module  = p.module
        ID    = p.ID

        """ NB : this is for the coldbox only, as the membrane PDS had only one readout """
        if(cf.pds_modules_type[module] == 'Membrane'):
            ID = len(cluster_list) + id_cluster_shift
            clus = build_pds_cluster([p, p], ID)
            cluster_list.append( clus )
            continue

        rtree_idx.insert(ID, (readout, module, start, readout, module, start))
        

    ''' Now searching for overlaps in the same PDS module'''
    for pi in dc.pds_peak_list:
        if(pi.cluster_ID >= 0):
            continue
        
        i_start = pi.start
        i_stop  = pi.stop
        i_chan  = pi.glob_ch
        i_readout = i_chan%2

        if(i_readout != 0):
            continue
        
        i_module  = int(i_chan/2)
        i_ID    = pi.ID 
    
        j_readout = 1-i_readout
        
        overlaps = list(rtree_idx.intersection((j_readout, i_module, i_start - time_tol, j_readout, i_module, i_start + time_tol)))        
        
        if(len(overlaps) > 0):
            peaks = [pi]
            for ov in overlaps:
                ov_idx = ov - id_peak_shift
                pj = dc.pds_peak_list[ov_idx]
                                
                if (pj.cluster_ID >=0 or pj.ID == i_ID):
                    continue
            
                peaks.append(pj)
                    
            if(len(peaks) > 1):
                ID = len(cluster_list) + id_cluster_shift
                clus = build_pds_cluster(peaks, ID)
                cluster_list.append( clus )

    

    """ try to merge the clusters across the PDS """
    """ create a new Rtree, now filled with the found clusters """
    pties = index.Property()
    pties.dimension = 2

    ''' create an rtree index (times and 0)'''
    rtree_mod = index.Index(properties=pties)

    
    ''' filling the R-tree '''
    for c in cluster_list:        
        start = c.t_start
        idx   = c.ID
        rtree_mod.insert(idx, (start, 0, start, 0))

        
    ''' now search for overlaps '''
    for ci in cluster_list:        
        if(ci.ID < 0):
            """ this cluster has already been merged """
            continue
        
        i_start = ci.t_start
        i_idx   = ci.ID

        overlaps = list(rtree_mod.intersection((i_start-time_tol, 0, i_start+time_tol, 0)))
        #print(i_idx, " :: ", overlaps, len(cluster_list))
        
        if(len(overlaps)>0):
            for ov in overlaps:
                if(ov <= i_idx):
                    continue
                co = cluster_list[ov-id_cluster_shift]
            
                ci.merge(co)
                co.set_ID(-1)
                
                [dc.pds_peak_list[p-id_peak_shift].set_cluster_ID(i_idx) for p in ci.peak_IDs]

        if(ci.ID >=0):
            if(ci.ID == dc.evt_list[-1].n_pds_clusters+id_cluster_shift):
               dc.pds_cluster_list.append(ci)
               dc.evt_list[-1].n_pds_clusters += 1
            else:
               new_ID = dc.evt_list[-1].n_pds_clusters + id_cluster_shift
               ci.set_ID(new_ID)
               [dc.pds_peak_list[p-id_peak_shift].set_cluster_ID(new_ID) for p in ci.peak_IDs]
               dc.pds_cluster_list.append(ci)
               dc.evt_list[-1].n_pds_clusters += 1
    



def hits_rtree(modules = [cf.imod]):

    
    dc.rtree_hit_idx = index.Index(properties=dc.pties)
    [dc.rtree_hit_idx.insert(h.ID, (h.module, h.view, h.X, min([h.Z_start, h.Z_stop]), h.module, h.view, h.X, max([h.Z_start, h.Z_stop]))) for h in dc.hits_list if h.module in modules]

    n_hits = 0
    for m in modules:
        n_hits += sum(dc.evt_list[-1].n_hits[:,m])
    #print(dc.rtree_hit_idx.count(dc.rtree_hit_idx.bounds), ' in rtree, vs ', n_hits)

def charge_clustering():

    y_squeez = 0.5
    eps = 3.
    min_samp = 1
    debug = True
    
    if(debug):
        import matplotlib.pyplot as plt
        import colorcet as cc
        fig = plt.figure()
        ax = [fig.add_subplot(131+i) for i in range(3)]
        [ax[i].sharey(ax[0]) for i in [1,2]]
        fig_a = plt.figure()
        axa = [fig_a.add_subplot(131+i) for i in range(3)]
        
    n_clusters, n_tot_clusters = 0,0
    ID_shift = dc.n_tot_hits_clusters

    
    for iview in range(cf.n_view):
        hits = [x for x in dc.hits_list if x.view==iview and x.module == cf.imod]#and x.has_3D==False]
        if(len(hits)==0): continue

        """ squeeze y axis instead of defining a new metric """
        data = [[x.X,x.Z*y_squeez] for x in hits]
        X = np.asarray(data)
        db = DBSCAN(eps=eps,min_samples=min_samp).fit(X)
        labels = db.labels_

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        count = Counter([l+n_tot_clusters+ID_shift for l in labels])

        #print('view ', iview, ' has ', n_clusters, ' clusters, tot now ', n_tot_clusters, '--', ID_shift)
        #print(count)
        
        [dc.hits_cluster_list.append(dc.hits_clusters(l+n_tot_clusters+ID_shift, iview)) for l in range(n_clusters)]
        [h.set_cluster(l+n_tot_clusters+ID_shift) for h,l in zip(hits,labels)]
        [dc.hits_cluster_list[h.cluster-ID_shift].add_hit(h.ID) for h in hits]
        
        
        n_tot_clusters += n_clusters        

        #biggest_label = max(count, key=count.get)
        #pts = [(h.X, h.Z) for h in hits if h.cluster == biggest_label]
        #print('test : ', count[biggest_label], ' vs ', len(pts))

        #pts = [(h.X, h.Z) for h in hits if count[h.cluster] > 10]# == biggest_label]
        #print(len(pts),' points to test the code ')
        
        z_anode = cf.anode_z[cf.imod]
        vdrift = lar.drift_velocity()
        max_drift = cf.anode_z[cf.imod] - cf.n_sample[cf.imod]*cf.drift_direction[cf.imod] * vdrift /cf.sampling[cf.imod]
        zmin, zmax = min([z_anode, max_drift]), max([z_anode, max_drift])
        xmin, xmax = cf.view_boundaries_min[cf.imod][iview], cf.view_boundaries_max[cf.imod][iview]
        l_max = np.sqrt(pow(zmax-zmin, 2)+pow(xmax-xmin, 2))
        print(zmin, zmax, ' and ', xmin, xmax, ' -> ', l_max)

        """
        lines = houghLines(pts, 0.5, l_max, 0.5, 10)
        print('nb of lines : ', len(lines))
        print(lines)
        """
        
        #pts = [[h.X, h.Z] for h in hits if h.cluster == biggest_label]

        pts = [[h.X, h.Z] for h in hits if count[h.cluster] > 5]
        print('\n View ', iview, ' --> ', len(pts), ' points vs ', len(hits))
        tt = time.time()
        lines, acc, peaks = hough_transform(np.asarray(pts), 0.5 ,0.5, 20, 1)        
        print('HOUGH took ', time.time()-tt)
        print('found ', len(lines), ' lines')
        tt = time.time()
        best_lines, dists = assign_points_to_lines(np.asarray(pts), np.asarray(peaks))
        print('assign lines to points took ', time.time()-tt)
        print(best_lines.shape, ' and ', dists.shape)
        for i in range(10):
            print(pts[i], ' with ', best_lines[i], ' d = ', dists[i])


        print('dists min ', min(dists), 'max', max(dists))
        
        points = np.asarray(pts)
        if(peaks != None):
            print('nb of peaks : ', len(lines))
            print('best line ', lines[0])
        if(debug):
            cmap = plt.get_cmap(cc.cm.glasbey)
            colors = [cmap(i % cmap.N) for i in range(len(lines))]
            
            """
            for i in range(len(lines)):
                c = colors[i]
                pts = points[best_lines == i]
                print('lines ',i,'::',pts.shape)
                ax[iview].scatter(pts[:, 0], pts[:, 1], s=1, c=c)
                l = lines[0]
                ax[iview].plot([xmin, xmax], [l[0]*xmin+l[1], l[0]*xmax+l[1]], lw=0.5, c=c)
            """


            """
            hx = [h.X for h,l in zip(hits, labels) if count[h.cluster] > 5]#l>=0]
            hy =  [h.Z for h,l in zip(hits, labels) if count[h.cluster] > 5]#
            clus =  [l for h,l in zip(hits, labels) if count[h.cluster] > 5]#

            #clus = [l for l in labels if count[l+n_tot_clusters+ID_shift] >5]#=0]
            print(len(hx), len(hy), len(clus))
            """
            if(peaks != None):
                for l in lines:
                    ax[iview].plot([xmin, xmax], [l[0]*xmin+l[1], l[0]*xmax+l[1]], lw=0.2, c='gray')
                l = lines[0]
                ax[iview].plot([xmin, xmax], [l[0]*xmin+l[1], l[0]*xmax+l[1]], lw=0.5, c='r')
            ax[iview].scatter(points[:,0], points[:,1], c=best_lines, cmap = cc.cm.glasbey, s=1)
            """
            """
            #[ax[iview].scatter(h.X, h.Z, s=0.5,c='k') for h,l in zip(hits, labels) ]#if l==-1]
            #ax[iview].scatter(hx, hy, s=1.5, c=clus, cmap=cc.cm.glasbey, zorder=200)            

            ax[iview].set_ylim(zmin, zmax)

            
            im = axa[iview].imshow(acc, aspect = 'auto', interpolation='none',cmap=cc.cm.linear_tritanopic_krjcw_5_95_c24_r)
            fig_a.colorbar(im, ax=axa[iview], location='right')
    if(debug):
        fig.tight_layout()
        fig_a.tight_layout()
        plt.show()
    dc.n_tot_hits_clusters += n_tot_clusters
    dc.evt_list[-1].n_hits_clusters += n_tot_clusters


