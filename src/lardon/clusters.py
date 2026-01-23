import lardon.config as cf
import lardon.data_containers as dc
import lardon.lar_param as lar

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
    timestamp = min([p.timestamp for p in peaks])
    
    cluster = dc.pds_cluster(cluster_ID, IDs, glob_chans, channels, t_starts, t_maxs, t_stops, max_adcs, charges, timestamp)    
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
    pties.dimension = 2

    ''' create an rtree index (channel, pds peak times)'''
    rtree_idx = index.Index(properties=pties)

    #all_starts = []
    
    ''' filling the R-tree '''
    for p in dc.pds_peak_list:

        start = p.timestamp
        chan  = p.glob_ch
        ID    = p.ID
        #all_starts.append((start, chan))
        
        rtree_idx.insert(ID, (chan, start, chan, start))        

    #sorted_starts = sorted(all_starts, key=lambda tup: tup[0])
    #print(len(sorted_starts),"\n\n")
    #print(sorted_starts[:100])
    
    ''' Now searching for overlaps in other PDS channels'''
    for pi in dc.pds_peak_list:
        if(pi.cluster_ID >= 0):
            continue
        
        i_start = pi.timestamp#start
        i_chan  = pi.glob_ch
        i_ID    = pi.ID 
            
        overlaps = list(rtree_idx.intersection((0, i_start - time_tol, 9999, i_start + time_tol)))        
            
        if(len(overlaps) > 0):
            peaks = [pi]
            for ov in overlaps:
                ov_idx = ov - id_peak_shift
                pj = dc.pds_peak_list[ov_idx]
                                
                if (pj.cluster_ID >=0 or pj.ID == i_ID or pj.glob_ch == chan):
                    continue
            
                peaks.append(pj)
                    
            if(len(peaks) > 1):
                ID = len(dc.pds_cluster_list) + id_cluster_shift
                clus = build_pds_cluster(peaks, ID)
                dc.pds_cluster_list.append( clus )
                dc.evt_list[-1].n_pds_clusters += 1
        







def old_light_clustering():

    cluster_list = []
    
    if(len(dc.pds_peak_list) <2 ):
        return

    time_tol = dc.reco['pds']['cluster']['time_tol'] #in ticks
    print('TiME TOLERANCE CLUSTERING ', time_tol)
    
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

        rtree_idx.insert(ID, (readout, module, start, readout, module, start))        
        
        """ NB : this is for the coldbox only, as the membrane PDS had only one readout """
        if(cf.pds_modules_type[module] == 'Membrane'):
            ID = len(cluster_list) + id_cluster_shift
            clus = build_pds_cluster([p, p], ID)
            cluster_list.append( clus )
            continue


        

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
