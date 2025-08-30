import data_containers as dc
import config as cf

import numpy as np


import pedestals as ped
import noise_filter as noise
import hit_finder as hf
import track_2d as trk2d
import stitch_tracks as stitch
import track_3d as trk3d
import single_hits as sh
import ghost as ghost
import clusters as clu
import matching as mat
import hits_3d as h3d

import time
from psutil import Process



import plotting as plot

def pds_signal_proc():
    if(cf.n_pds_sample <= 0):
        return
    
    """ compute the pedestal """
    ped.compute_pedestal_pds(first=True)
    noise.median_filter_pds()
    ped.compute_pedestal_pds(first=False)


def pds_reco():
    if(cf.n_pds_sample <= 0):
        return
    
    #plot.event_display_coll_pds(draw_trk_t0 = True, to_be_shown=True)
    hf.find_pds_peak()
    #plot.draw_pds_ED(to_be_shown=True, draw_peak=True)#, draw_peak=False, draw_cluster=False, draw_roi=False)

    #[p.dump() for p in dc.pds_peak_list]
    
    clu.light_clustering()
    #plot.draw_pds_ED(to_be_shown=True, draw_cluster=True, draw_peak=True)#, draw_peak=False, draw_cluster=False, draw_roi=False)
    print('--> Found ', dc.evt_list[-1].n_pds_clusters, ' clusters ')
    #[c.dump() for c in dc.pds_cluster_list]
    #plot.draw_pds_ED(to_be_shown=True, draw_peak=True, draw_cluster=True, draw_roi=False)
    


def charge_pulsing():
    if(cf.n_sample[cf.imod] <= 0):
        return
    
    """ mask the unused channels """
    dc.mask_daq = np.logical_and(dc.mask_daq, dc.alive_chan[:,None])
    

    t1 =time.time()
    """ compute the raw pedestal to get a rough mask estimate """
    ped.compute_pedestal(noise_type='raw')
                
    """ update the pedestal and ROI """
    for n_iter in range(2):
        ped.compute_pedestal(noise_type='filt')
        ped.refine_mask(n_pass=1)
    deb.ped_1[cf.imod] = time.time()-t1

    """ pulse analysis does not need noise filtering """
    pulse.find_pulses()


def charge_signal_proc(deb, is_online):
    if(cf.n_sample[cf.imod] <= 0):
        ped.set_dummy_pedestals()
        return
    
    """ mask the unused channels """
    dc.mask_daq = np.logical_and(dc.mask_daq, dc.alive_chan[:,None])
    

    t1 =time.time()
    """ compute the raw pedestal to get a rough mask estimate """
    ped.compute_pedestal(noise_type='raw')
                
    """ update the pedestal and ROI """
    for n_iter in range(2):
        ped.compute_pedestal(noise_type='filt')
        ped.refine_mask(n_pass=1)
    deb.ped_1[cf.imod] = time.time()-t1

    if(is_online):
        plot.event_display_per_view([-100, 100],[-50, 300], option='raw', to_be_shown=True)

    
    t1 = time.time()
    """ low pass FFT cut """    
    _ = noise.FFT_low_pass(False)    
    deb.fft[cf.imod] = time.time()-t1

    
    if(dc.data_daq.shape[-1] != cf.n_sample[cf.imod]):
        """ 
        when the nb of sample is odd, the FFT returns 
        an even nb of sample. 
        Need to append an extra value (0) at the end 
        of each waveform to make it work """
        
        dc.data_daq = np.insert(dc.data_daq, dc.data_daq.shape[-1], 0, axis=-1)


    """ re-compute pedestal and update mask """        
    t1 = time.time()
    for n_iter in range(2):
        ped.compute_pedestal(noise_type='filt')
        ped.refine_mask(n_pass=2)
    deb.ped_2[cf.imod] = time.time()-t1


    
    """ special microphonic noise study """
    ped.study_noise()

    t1 = time.time()
    """ CNR """
    
    '''
    noise.shield_coupling()
    plot.event_display_per_view([-200, 200],[-100, 300], option='shield', to_be_shown=True)
    '''
    noise.coherent_noise()
    deb.cnr[cf.imod] = time.time()-t1

    
    """ microphonic noise removal """
    noise.median_filter()
            
    t1 = time.time()
    """ finalize pedestal RMS and ROI """
    ped.compute_pedestal(noise_type='filt')
    ped.refine_mask(n_pass=2)
    ped.compute_pedestal(noise_type='filt')
    deb.ped_3[cf.imod] = time.time()-t1
        

    """ extract hits """
    t1 = time.time()
    hf.find_hits()    
    deb.hit_f[cf.imod] = time.time()-t1
    print("----- Number Of Hits found : ", dc.evt_list[-1].n_hits[:,cf.imod])

    #plot.event_display_per_view_hits_found([-20, 20],[-10, 50], option='filt', to_be_shown=False)
    #plot.plot_2dview_hits([cf.imod], to_be_shown=True)
            
    #return ps


def charge_reco_pdvd(deb):
    if(cf.n_sample[cf.imod] <= 0):
        return
    
    h3d.build_3D_hits_with_cluster()
            
    
def charge_reco(deb, is_online):
    if(cf.n_sample[cf.imod] <= 0):
        return
    

    """ build hits R-tree used in track2D and single hit searches """
    clu.hits_rtree([cf.imod])
    
    
    """ search for 2D tracks """
    tt = time.time()
    
    t1 = time.time()

    trk2d.find_tracks_hough([cf.imod])
    deb.trk2D_1[cf.imod] = time.time()-t1
    
    #print("---- Number Of 2D tracks found : ", dc.evt_list[-1].n_tracks2D)
    
    
    """ stitch together pieces of 2D tracks """
    t1 = time.time()

    stitch.stitch2D_in_module([cf.imod])
    
    deb.stitch2D[cf.imod] = time.time()-t1            

    print("---- Number Of 2D tracks found : ", dc.evt_list[-1].n_tracks2D)

    #plot.plot_2dview_2dtracks([cf.imod], to_be_shown=True, option='mod_'+str(cf.imod))
    
    """ tag potential ghosts """
    ghost.ghost_finder()
    

    """ build 3D tracks from 3 views"""
    t1 = time.time()
    trk3d.find_track_3D_rtree_new([cf.imod])


    deb.trk3D[cf.imod] = time.time()-t1

     
    """ build 3D tracks if a view is missing """
    trk3d.find_3D_tracks_with_missing_view([cf.imod])
    
    print("--- Number of 3D tracks found : ", len(dc.tracks3D_list))
    
        
    """ reconstruct the ghosts """
    ghost.ghost_trajectory()

    """ search for single hits in free hits """
    t1 = time.time()
    
    sh.single_hit_finder([cf.imod])
    deb.single[cf.imod] = time.time()-t1
    print('-- Found ', len(dc.single_hits_list), ' Single Hits!')

    #plot.plot_2dview_hits_3dtracks([cf.imod], to_be_shown=False)    
    #plot.event_display_per_view_hits_found([-50, 50],[-10, 100], option='filt_new', to_be_shown=True)

def charge_reco_whole(is_online):
    
    stitch.stitch3D_across_modules([0,1])
    stitch.stitch3D_across_modules([2,3])


    if(dc.evt_list[-1].det == 'pdhd'):        
        stitch.stitch3D_across_cathode([[0,1],[2,3]])
    else:
        stitch.stitch3D_across_cathode([[2,3], [0,1]])

    #[t.dump() for t in dc.tracks3D_list]

    if(is_online):
        plot.plot_3d(to_be_shown=False)
        plot.plot_noise_all_crps(to_be_shown=False)
    
    
def match_charge_and_pds():
    if(cf.n_sample[cf.imod] <= 0 or cf.n_pds_sample <= 0):
        return
    
    mat.matching_charge_pds()               
