import lardon.data_containers as dc
import lardon.config as cf

import numpy as np

import lardon.trigger as trig

import lardon.pedestals as ped
import lardon.noise_filter as noise
import lardon.hit_finder as hf
import lardon.track_2d as trk2d
import lardon.stitch_tracks as stitch
import lardon.track_3d as trk3d
import lardon.single_hits as sh
import lardon.ghost as ghost
import lardon.clusters as clu
#import lardon.matching as mat
import lardon.pattern_matching as pmat
import lardon.track_timing as tmg

#import lardon.hits_3d as h3d
#import lardon.vertexing as vtx

import time
from psutil import Process



import lardon.plotting as plot

def pds_signal_proc():
    if(cf.n_pds_stream_sample <=0 and cf.n_pds_trig_sample <= 0):
        return
    
    """ compute the pedestal """
    ped.compute_pedestal_pds(first=True)
    noise.median_filter_pds()
    ped.compute_pedestal_pds(first=False)
def pds_reco():
    if(cf.n_pds_stream_sample <=0 and cf.n_pds_trig_sample <= 0):
        return
    

    hf.find_all_pds_peak()
    
    #plot.draw_all_pds_ED(data_type='stream', to_be_shown=True, draw_roi=True, draw_peak=True)
    #plot.draw_all_pds_ED(data_type='trigger', to_be_shown=True, draw_roi=True, draw_peak=True)

    clu.light_clustering()


    print('-- Found ', sum(dc.evt_list[-1].n_pds_peaks), ' PDS peaks')
    print('   per PDS channels:')
    print(dc.evt_list[-1].n_pds_peaks)
    print('- Found ', dc.evt_list[-1].n_pds_clusters, ' clusters ')


    from collections import Counter
    print('Cluster sizes')
    print(Counter([c.size for c in dc.pds_cluster_list]))

    #plot.draw_pds_ED([0, 2, 4, 6, 8, 10, 12, 14], to_be_shown=True)
    #plot.plot_timeline(option=None, to_be_shown=True)
    #plot.draw_pds_peaks(option=None, to_be_shown=True)
    
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
        print("setting dummy pedestals")
        return
    """
    if(cf.imod >1):
        return
    """
    
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
        plot.event_display_per_view([-100, 100],[-50, 300], option='raw', to_be_shown=False)
        

    #my_channels = [(0, 62), (1,343), (2, 20)]
    #plot.plot_wvf_current_vch(my_channels, to_be_shown=True)
    #plot.plot_wvf_current_hits_roi_vch(my_channels))
    
    t1 = time.time()
    """ low pass FFT cut """    
    #ps = noise.FFT_low_pass(True)
    
    _ = noise.FFT_low_pass(False)    

    deb.fft[cf.imod] = time.time()-t1

    if(dc.data_daq.shape[-1] != cf.n_sample[cf.imod]):
        """ 
        when the nb of sample is odd, the FFT returns 
        an even nb of sample. 
        Need to append an extra value (0) at the end 
        of each waveform to make it work """
        
        dc.data_daq = np.insert(dc.data_daq, dc.data_daq.shape[-1], 0, axis=-1)

    if(dc.mask_daq.shape[-1] != cf.n_sample[cf.imod]):        
        dc.mask_daq = np.insert(dc.mask_daq, dc.mask_daq.shape[-1], 0, axis=-1)

        

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
    

    """ only for PDVD TDE View 0 """
    noise.shield_coupling()    


    noise.coherent_noise()

    #print('cnr took', time.time()-t1)
    deb.cnr[cf.imod] = time.time()-t1


    """ microphonic noise removal """
    noise.median_filter()

    
    t1 = time.time()
    """ finalize pedestal RMS and ROI """
    ped.compute_pedestal(noise_type='filt')
    ped.refine_mask(n_pass=2)
    ped.compute_pedestal(noise_type='filt')
    deb.ped_3[cf.imod] = time.time()-t1


    #return ps, corr


    #plot.event_display_per_view_roi([-50, 50],[-50, 150], option='raw', to_be_shown=True)
    #plot.event_display_per_view_noise([-50, 50],[-50, 150], option='raw', to_be_shown=True)


    
def charge_reco_pdvd(deb):
    if(cf.n_sample[cf.imod] <= 0):
        return
    
    h3d.build_3D_hits_with_cluster()
            
    
def charge_reco(deb, is_online):
    if(cf.n_sample[cf.imod] <= 0):
        return
    """
    if(cf.imod >1):
        return
    """
    
    """ extract hits """
    t1 = time.time()
    hf.find_hits()    
    deb.hit_f[cf.imod] = time.time()-t1
    print("----- Number of Hits found per view: ", dc.evt_list[-1].n_hits[:,cf.imod])



    #plot.event_display_per_view_hits_found([-50, 50],[-50, 150], option='raw', to_be_shown=True)
    #my_channels = [(0, 62), (1,343), (2, 20)]
    #plot.plot_wvf_current_hits_roi_vch(my_channels, to_be_shown=True)
    
    
    """ build hits R-tree used in track2D and single hit searches """
    clu.hits_rtree([cf.imod])
    
    
    """ search for 2D tracks """
    tt = time.time()
    
    t1 = time.time()

    trk2d.find_tracks_hough([cf.imod])
    deb.trk2D_1[cf.imod] = time.time()-t1

    
    """ stitch together pieces of 2D tracks """
    t1 = time.time()
    #plot.plot_2dview_hits_3dtracks([cf.imod], option=None, to_be_shown=True)
    stitch.stitch2D_in_module([cf.imod])
    stitch.stitch2D_in_module([cf.imod])
    print('stitching took ', time.time()-t1)
    #plot.plot_2dview_hits_3dtracks([cf.imod], option=None, to_be_shown=True)
    
    deb.stitch2D[cf.imod] = time.time()-t1            

    print("---- Number of 2D tracks found per view: ", dc.evt_list[-1].n_tracks2D)


    """ tag potential ghosts """
    #ghost.ghost_finder()
    
    ntrk3D_prev = len(dc.tracks3D_list)
    """ build 3D tracks from 3 views"""
    t1 = time.time()
    trk3d.find_track_3D_rtree_new([cf.imod])


    deb.trk3D[cf.imod] = time.time()-t1

     
    """ build 3D tracks if a view is missing """
    trk3d.find_3D_tracks_with_missing_view([cf.imod])
    
    print("--- Number of 3D tracks found: ", len(dc.tracks3D_list)-ntrk3D_prev, " total: ", len(dc.tracks3D_list))
    

    #print('\n\n==================================\n\n')
    """ reconstruct the ghosts """
    #ghost.ghost_trajectory()

    #vtx.vertexing()

    """ search for single hits in free hits """
    t1 = time.time()

    nSH_prev = len(dc.single_hits_list)
    sh.single_hit_finder([cf.imod])
    deb.single[cf.imod] = time.time()-t1
    print('-- Number of blips found: ', len(dc.single_hits_list)-nSH_prev, " total: ", len(dc.single_hits_list))


    if(cf.imod > 4):
        plot.event_display_per_view_hits_found([-50, 50],[-10, 150], option='filt', to_be_shown=True)

    #plot.plot_2dview_hits_3dtracks([cf.imod], option=None, to_be_shown=True)
    """
    if(cf.imod>=1):
        plot.plot_2dview_hits_3dtracks([cf.imod], option=None, to_be_shown=True)
        plot.event_display_per_view_hits_found([-50, 50],[-10, 150], option='reco', to_be_shown=True) 
    """
    #if(cf.imod >= 2):
    #    plot.event_display_per_view_hits_found([-50, 50],[-10, 150], option='reco', to_be_shown=True) 
    #if(cf.imod>=2):
    #    

def charge_reco_whole(is_online):
    print('\n## All Detector ##')

    
    if(dc.evt_list[-1].det == 'pdhd'):
        stitch.stitch3D_across_modules([0,1])
        stitch.stitch3D_across_modules([2,3])
        stitch.stitch3D_across_cathode([[0,1],[2,3]])
    elif(dc.evt_list[-1].det == 'pdvd'):
        stitch.stitch3D_across_modules([0,1])
        stitch.stitch3D_across_modules([2,3])
        stitch.stitch3D_across_cathode([[2,3], [0,1]])


    trig.search_trigger_track()
    tmg.compute_all_track_timing()
    
    
    if(is_online):
        plot.plot_3d(to_be_shown=True)
        plot.plot_noise_all_crps(to_be_shown=True)
    #plot.plot_noise_all_crps(to_be_shown=True)
    #plot.plot_3d(to_be_shown=True)

    #[t.dump() for t in dc.tracks3D_list]
    #plot.plot_3d(to_be_shown=True)    
    #plot.plot_one_track_3D(dc.tracks3D_list[30], option=None, to_be_shown=True)
    #plot.plot_one_track_3D(dc.tracks3D_list[22], option=None, to_be_shown=True)

    
def match_charge_and_pds():
    #[t.dump() for t in dc.tracks3D_list]
    #plot.plot_timeline(option=None, to_be_shown=True)

    if(cf.n_sample[cf.imod] <= 0 or (cf.n_pds_stream_sample <=0 and cf.n_pds_trig_sample <= 0)):
        return

    t1 = time.time()
    #mat.matching_trk_pds()
    pmat.matching_trk_pds()
    print('light matching took ', time.time()-t1)
    """
    plot.plot_timeline(option=None, to_be_shown=True)    
    """

    """
    for t in dc.tracks3D_list:
        #plot.plot_track_dqds(t, option=None, to_be_shown=True)
        #continue
        #if(t.is_decay_from_light):
        #    plot.plot_track_dqds(t, option=None, to_be_shown=True)
            #plot.plot_track_pds_matched(t, option=None, to_be_shown=True)
        #if(t.is_cathode_crosser and t.cathode_crosser_ID>=0):
        #plot.plot_one_track_3D(t, option=None, to_be_shown=True)
            
        if(t.match_pds_cluster >= 0):
            t.dump()
            plot.plot_track_pds_matched(t, option=None, to_be_shown=True)
        else:
            print("NOT MATCHED")
            #t.dump()
            plot.plot_one_track_3D(t, option=None, to_be_shown=True)
    """
    
    #NOT READY YET ! 
    #mat.matching_sh_pds()               
    
