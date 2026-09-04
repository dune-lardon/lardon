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
    """ sort peaks by global channel number """
    peaks.sort()
    
    IDs = [p.ID for p in peaks]
    glob_chans = [p.glob_ch for p in peaks]
    channels = [p.channel for p in peaks]

    #t_starts = [p.start for p in peaks]
    #t_maxs = [p.max_t for p in peaks]
    #t_stops = [p.stop for p in peaks]

    
    max_adcs = [p.max_adc for p in peaks]
    charges = [p.charge for p in peaks]

    max_npes = [p.max_npe for p in peaks]
    npes = [p.npe for p in peaks]


    timestamps = [p.timestamp for p in peaks]
    durations = [p.timestamp + (p.stop-p.start)/cf.pds_sampling for p in peaks]
    
    timestamp = min(timestamps)
    timestamp_end = max(durations)
    
    #cluster = dc.pds_cluster(cluster_ID, IDs, glob_chans, channels, t_starts, t_maxs, t_stops, max_adcs, charges, timestamp)

    cluster = dc.pds_cluster(cluster_ID, IDs, glob_chans, channels, max_adcs, charges, max_npes, npes, timestamp, timestamp_end)
    [p.set_cluster_ID(idx) for p in peaks]

    return cluster



def sparse_crosscorr(peaks_ref, peaks_ch, max_lag, debug=False):
    """
    peaks_ref and peaks_ch are sorted lists/arrays of integer time bins.
    max_lag is the search window (in bins).
    Returns: best lag (in bins)
    """

    lag_scores = np.zeros(2 * max_lag + 1, dtype=int)
    lag_range = np.arange(-max_lag, max_lag + 1)

    
    # For each lag, count overlaps
    for i, lag in enumerate(lag_range):
        shifted = peaks_ch + lag/cf.pds_sampling
        if(debug):
            print(i, lag/cf.pds_sampling, shifted[:30])
        lag_scores[i] = np.sum(np.isin(shifted, peaks_ref))

    best_lag = lag_range[np.argmax(lag_scores)]
    return best_lag, max(lag_scores)



def align_waveforms(max_lag):

    peaks_time = [[] for x in range(cf.n_pds_tot_channels)]
    
    for p in dc.pds_peak_list:
        start = int(round(p.timestamp*cf.pds_sampling))/cf.pds_sampling
        chan  = p.glob_ch
        peaks_time[chan].append(start)


    [peaks_time[ch].sort() for ch in range(cf.n_pds_tot_channels)]
    delays = [0 for x in range(cf.n_pds_tot_channels)]

    reference = peaks_time[0]
    if(len(reference) == 0):
        return
    dc.evt_list[-1].pds_time_offset[0] = 0.
    

    for ch in range(cf.n_pds_tot_channels):
        if ch == 0:
            delays[ch] = 0
            continue

        lag, score = sparse_crosscorr(reference, peaks_time[ch], max_lag)#,debug=ch==1)
        delays[ch] = lag

        if(score > 0):
            dc.evt_list[-1].pds_time_offset[ch] = lag/cf.pds_sampling
    #print("DELAY: ", dc.evt_list[-1].pds_time_offset)

def check_unique_channels(peaks):
    """ remove all peaks in a given channel if they appear multiple time in the cluster """
    gchan = [p.glob_ch for p in peaks]
    counts = Counter(gchan)
    unique_chan = [i for i, x in enumerate(gchan) if counts[x] == 1]
    return [peaks[i] for i in unique_chan]

def light_clustering():

    max_lag = dc.reco['pds']['cluster']['max_lag'] #in tick nb
    time_tol = dc.reco['pds']['cluster']['time_tol'] #in mus

    align_waveforms(max_lag)
    
    cluster_list = []
    
    if(len(dc.pds_peak_list) <2 ):
        return
    
    
    id_peak_shift = dc.n_tot_pds_peaks
    id_cluster_shift = dc.n_tot_pds_clusters

    
    pties = index.Property()
    pties.dimension = 2

    ''' create an rtree index (channel, pds peak times)'''
    rtree_idx = index.Index(properties=pties)

    
    ''' filling the R-tree '''
    for p in dc.pds_peak_list:

        chan  = p.glob_ch
        start = p.timestamp + dc.evt_list[-1].pds_time_offset[chan]
        ID    = p.ID
        mod   = p.module        
        rtree_idx.insert(ID, (chan, start, chan, start))        

    
    ''' Now searching for overlaps in other PDS channels'''
    for pi in dc.pds_peak_list:
        if(pi.cluster_ID >= 0):
            continue
        i_chan  = pi.glob_ch        
        i_start = pi.timestamp + dc.evt_list[-1].pds_time_offset[i_chan]
        i_ID    = pi.ID 
        i_mod   = pi.module
            
        overlaps = list(rtree_idx.intersection((0, i_start - time_tol, 9999, i_start + time_tol)))        
            
        if(len(overlaps) > 0):
            peaks = [pi]
            for ov in overlaps:
                ov_idx = ov - id_peak_shift
                pj = dc.pds_peak_list[ov_idx]
                                
                if (pj.cluster_ID >=0 or pj.ID == i_ID):# or pj.glob_ch == i_chan):
                    continue
            
                peaks.append(pj)


            peaks = check_unique_channels(peaks)
            if(len(peaks) > 1):
                ID = len(dc.pds_cluster_list) + id_cluster_shift
                clus = build_pds_cluster(peaks, ID)
                dc.pds_cluster_list.append( clus )
                dc.evt_list[-1].n_pds_clusters += 1
        

    



def hits_rtree(modules = [cf.imod]):

    
    dc.rtree_hit_idx = index.Index(properties=dc.pties)
    [dc.rtree_hit_idx.insert(h.ID, (h.module, h.view, h.X, min([h.Z_start, h.Z_stop]), h.module, h.view, h.X, max([h.Z_start, h.Z_stop]))) for h in dc.hits_list if h.module in modules]

    n_hits = 0
    for m in modules:
        n_hits += sum(dc.evt_list[-1].n_hits[:,m])
