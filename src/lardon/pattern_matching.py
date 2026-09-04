import lardon.config as cf
import lardon.data_containers as dc
import lardon.lar_param as lar
import lardon.track_timing as tmg

import lardon.plotting as plot

import math
import numpy as np
from rtree import index
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from collections import Counter
from scipy.special import gammaln

import lardon.light_prediction as lp


def set_track_pds_matching(trk, clus):

    trk.match_pds_cluster = clus.ID
    trk_vol   = int(trk.module_ini/cf.n_drift_volumes)
    clus.match_trk3D[trk_vol] = trk.ID_3D

def get_trk_clus_param(clus, trk, vdrift):
    t0_light = clus.timestamp - dc.evt_list[-1].delay_charge_time[trk.module_ini]
    z0_light =  cf.drift_direction[trk.module_ini]*t0_light*vdrift

    P0, P1, tdir = tmg.track_point_direction(trk, z0_light)
    return z0_light, P0, P1, tdir

def matching_trk_pds():
    if(len(dc.tracks3D_list)== 0 or len(dc.pds_cluster_list) == 0):
        return

    v_drift = [lar.drift_velocity(m) for m in range(cf.n_module)]

    anode_time_tol_bef = dc.reco['pds']['tpc_matching']['anode_crosser']['time_tol_bef'] #in mus
    anode_time_tol_aft = dc.reco['pds']['tpc_matching']['anode_crosser']['time_tol_aft'] #in mus
    
    cathode_time_tol_bef = dc.reco['pds']['tpc_matching']['cathode_crosser']['time_tol_bef'] #in mus    
    cathode_time_tol_aft = dc.reco['pds']['tpc_matching']['cathode_crosser']['time_tol_aft'] #in mus

    
    unknown_time_tol_aft = dc.reco['pds']['tpc_matching']['unknown']['time_tol_aft'] #in mus
    unknown_time_tol_bef = dc.reco['pds']['tpc_matching']['unknown']['time_tol_bef'] #in mus

    min_cluster_size = dc.reco['pds']['tpc_matching']['min_cluster_size']
    
    
    light_pred = lp.light_prediction(100000)
    logmax = 10000
    n_matched_track = 0 
    
    n_trk = 0
    
    """ create  Rtree, filled with the found clusters """
    pties = index.Property()
    pties.dimension = 2

    rtree = index.Index(properties=pties)
        


    id_trk3d_shift = dc.n_tot_trk3d
    id_cluster_shift = dc.n_tot_pds_clusters
    
    ''' filling the R-tree with the light clusters on y axis = 0'''
    for c in dc.pds_cluster_list:
        if(np.any(np.array(c.match_trk3D)>=0) or c.match_single >=0 ):
            continue
        if(c.size < min_cluster_size):
            continue
        start = c.timestamp
        idx   = c.ID
        rtree.insert(idx, (start, 0, start, 0))

    tracks = [t for t in dc.tracks3D_list]
    ntracks = len(tracks)
    nclusters = len(dc.pds_cluster_list)

    time_tol_bef = [0 for x in range(ntracks)]
    time_tol_aft = [0 for x in range(ntracks)]
    
    ''' filling the R-tree with the 3D tracks on y axis =1,2,3 '''
    for k,t in enumerate(dc.tracks3D_list):
        if(t.match_pds_cluster >= 0):
            continue

        start = t.timestamp
        stop  = t.timestamp_r
        volume = int(t.module_ini/cf.n_drift_volumes)
        idx   = t.ID_3D


        is_anode_crosser = t.is_anode_crosser
        is_cathode_crosser = t.is_cathode_crosser
        is_trigger = t.is_trigger

        if(is_trigger):
            time_tol_bef[k] = anode_time_tol_bef[volume]
            time_tol_aft[k] = anode_time_tol_aft[volume]
        elif(is_anode_crosser and is_cathode_crosser):
            time_tol_bef[k] = anode_time_tol_bef[volume]
            time_tol_aft[k] = anode_time_tol_aft[volume]
        elif(is_anode_crosser and not is_cathode_crosser):
            time_tol_bef[k] = anode_time_tol_bef[volume]
            time_tol_aft[k] = anode_time_tol_aft[volume]
        elif(not is_anode_crosser and is_cathode_crosser):#???
            time_tol_bef[k] = cathode_time_tol_bef[volume]
            time_tol_aft[k] = cathode_time_tol_aft[volume]
        else:
            time_tol_bef[k] = unknown_time_tol_bef[volume]
            time_tol_aft[k] = unknown_time_tol_aft[volume]
            
        rtree.insert(idx, (start, 1, stop, 1))

    #anode_cathode_tracks = [t for t in dc.tracks3D_list if t.is_anode_crosser and t.is_cathode_crosser]
    #anode_tracks = [t for t in dc.tracks3D_list if t.is_anode_crosser and not t.is_cathode_crosser]
    #cathode_tracks = [t for t in dc.tracks3D_list if not t.is_anode_crosser and t.is_cathode_crosser]# and t.cathode_crosser_ID>=0]
    #unknown_tracks = [t for t in dc.tracks3D_list if not t.is_cathode_crosser and not t.is_anode_crosser]
        

    #sparse matrix should be (N,N) shape
    sparse = np.zeros((ntracks+nclusters, ntracks+nclusters))
    
    match_record = [(np.inf, None) for x in range(ntracks)]
    
    for trk, t_bef, t_aft in zip(dc.tracks3D_list, time_tol_bef, time_tol_aft):
        if(trk.match_pds_cluster >= 0):
            continue

        
        #if(trk.is_cathode_crosser == True and trk.cathode_crosser_ID >=0):
        #    tmin, tmax = trk.timestamp, trk.timestamp_r
        #else:
        #    tmin, tmax = tmg.possible_timestamp_range(trk)
        
            
        module_ini = trk.module_ini
        vdrift = lar.drift_velocity(module_ini)

        trk_start = trk.timestamp
        trk_stop  = trk.timestamp_r
        trk_vol   = int(trk.module_ini/cf.n_drift_volumes)
            
        pds_overlaps = list(rtree.intersection((trk_start - t_bef, 0, trk_stop + t_aft, 0)))
        pds_overlaps = [ov for ov in pds_overlaps if all(x < 0 for x in dc.pds_cluster_list[ov-id_cluster_shift].match_trk3D)]
        
        if(len(pds_overlaps) == 0):
            #match_record[trk.ID_3D-id_trk3d_shift]=(np.inf, None)
            #print('\n TRACK ', trk.ID_3D, " has NO PDS OVERLAP ! in ts range ", trk_start - t_bef, 'to',trk_stop + t_aft)
            #print('test with a wider range: ')
            pds_overlaps = list(rtree.intersection((trk_start - t_bef-200, 0, trk_stop + t_aft+60, 0)))
            pds_overlaps = [ov for ov in pds_overlaps if all(x < 0 for x in dc.pds_cluster_list[ov-id_cluster_shift].match_trk3D)]
            #print('---->>> Now ', len(pds_overlaps))
            if(len(pds_overlaps) == 0):
                continue                       

        best_logL, best_cluster, best_is_dk = np.inf, None, False
        best_debuglogL = np.inf
        
        #print('\nTRACK ', trk.ID_3D, 'testing ', len(pds_overlaps),' ts range ', trk_start - t_bef, 'to',trk_stop + t_aft, " has ", len(pds_overlaps))
        
        for pds_ov in pds_overlaps:
            clus = dc.pds_cluster_list[pds_ov-id_cluster_shift]

            
            z0_light, P0, P1, trk_dir = get_trk_clus_param(clus, trk, vdrift)
            logL, is_decay, debuglogL = track_light_log_likelihood(clus, P0, P1, trk_dir, light_pred)

            if(abs(logL) < logmax and abs(logL) < best_logL):
                best_logL = logL
                best_cluster = clus
                best_is_dk = is_decay
                best_debuglogL = debuglogL
            #print('   test cluster ', clus.ID, ' at ', clus.timestamp, "size", clus.size, "Z0light = ", z0_light, " --->>> ", logL, ' decay?', is_decay)

        if(np.isfinite(best_logL)):            
            #print('Track ', trk.ID_3D,'--> BEST MATCH with  ', best_cluster.ID, "LOGL = ", best_logL, ' at ', best_cluster.timestamp, 'is decay?', best_is_dk, "-- other log ", best_debuglogL)
            match_record[trk.ID_3D-id_trk3d_shift]=(best_logL, best_is_dk)                
            sparse[trk.ID_3D-id_trk3d_shift, ntracks+best_cluster.ID-id_cluster_shift] = 1
        #else:
        #    print(' ... NO GOOD CLUSTER FOUND :/ ')


            
    """ sort the track -- cluster matched """
    graph = csr_matrix(sparse)
    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
    count = Counter(labels)


    #print("CHECKING ", len(match_record))
    """ count == 1 are unmatched clusters and tracks """
    for lab, nelem in count.items():
        if(nelem == 2):
            #print('\nUnique track - light match !! ', lab)
            match = np.where(labels == lab)[0]
            #print(match)
            
            m_trk = dc.tracks3D_list[match[0]]#+id_trk3d_shift]
            if(m_trk.is_cathode_crosser == True and m_trk.cathode_crosser_ID >=0):
                #print('Uh oh ... cathode crosser track', m_trk.ID_3D, 'matched but not the other one', m_trk.cathode_crosser_ID)
                continue
               
            m_clus = dc.pds_cluster_list[match[1] - ntracks]
            set_track_pds_matching(m_trk, m_clus)

            n_matched_track += 1
            m_trk.set_times_from_light(m_clus.timestamp, v_drift[t.module_ini])
            vol   = int(m_trk.module_ini/cf.n_drift_volumes)
            
            module_ini = m_trk.module_ini
            vdrift = lar.drift_velocity(module_ini)
            
            z0_light, P0, P1, trk_dir = get_trk_clus_param(m_clus, m_trk, vdrift)
            #print('P0= ',P0, ' direction = ', trk_dir, "Z0= ", z0_light)
            light_pred.set_track(P0, P1, trk_dir, m_clus.glob_chans)    

            logL, is_decay = match_record[match[0]]#+id_trk3d_shift]
            #print('logL = ', logL, ' decay?', is_decay)
               

            m_trk.set_AV_extrapolation(light_pred.t0, light_pred.t1, light_pred.ta, light_pred.tb)
            m_trk.is_decay_from_light = is_decay
            m_trk.set_pds_logL(logL)                   
            m_clus.point_impact[vol].extend(light_pred.pds_impact_point_per_channels)

            if(is_decay):
                m_clus.dist_closest[vol].extend(light_pred.track_dk_closest_distance_per_channels)
                m_clus.point_closest[vol].extend(light_pred.track_dk_closest_point_per_channels)
                m_clus.costheta_closest[vol].extend(light_pred.track_dk_closest_costheta_per_channels)
                m_clus.dist_maxval[vol].extend(light_pred.track_dk_maxval_distance_per_channels)
                m_clus.point_maxval[vol].extend(light_pred.track_dk_maxval_point_per_channels)

                
                m_clus.n_predicted[vol].extend(light_pred.dk_prediction_per_channels)
                m_clus.n_geo_predicted[vol].extend(light_pred.geo_dk_prediction_per_channels)
                
            else:
                m_clus.dist_closest[vol].extend(light_pred.track_closest_distance_per_channels)
                m_clus.point_closest[vol].extend(light_pred.track_closest_point_per_channels)
                m_clus.costheta_closest[vol].extend(light_pred.track_closest_costheta_per_channels)
                m_clus.dist_maxval[vol].extend(light_pred.track_maxval_distance_per_channels)
                m_clus.point_maxval[vol].extend(light_pred.track_maxval_point_per_channels)
                
                m_clus.n_predicted[vol].extend(light_pred.prediction_per_channels)
                m_clus.n_geo_predicted[vol].extend(light_pred.geo_prediction_per_channels)
                
            
                
            """
            if(is_dk == False):
                dists = light_pred.track_closest_distance_per_channels
                pts =  light_pred.track_closest_point_per_channels
                preds = light_pred.prediction_per_channels

            else:
                dists = light_pred.track_dk_closest_distance_per_channels
                pts =  light_pred.track_dk_closest_point_per_channels
                preds = light_pred.dk_prediction_per_channels
            
            plot.plot_track_pds_test(m_trk, z0_light, m_clus, dists, pts, preds, logL, option=None, to_be_shown=True)
            """
            
        elif(nelem > 2):
            #print('\nMORE MATCH: ', nelem, ' : ', lab)
            matches = np.where(labels == lab)[0]
            #print(matches)
            
            m_trks = [dc.tracks3D_list[m] for m in matches if m < ntracks]
            m_clusters = [dc.pds_cluster_list[m-ntracks] for m in matches if m >= ntracks]
            #print('NB of tracks ', len(m_trks), ' NB of clusters ', len(m_clusters))
            #print('tracks ', [m.ID_3D for m in m_trks])
            #print('cluster', [c.ID for c in m_clusters])
            if(len(m_clusters)>1):
                continue
            if(len(m_trks) > 2):
                #print('too many tracks')
                continue

            if(m_trks[0].is_cathode_crosser and m_trks[1].is_cathode_crosser):
                if(m_trks[0].cathode_crosser_ID == m_trks[1].ID_3D):
                    
                    m_clus = m_clusters[0]

                    for m_trk, match in zip(m_trks, matches):
                        if(match >=ntracks):
                            continue
                        set_track_pds_matching(m_trk, m_clus)
                        n_matched_track += 1
                        m_trk.set_times_from_light(m_clus.timestamp, v_drift[t.module_ini])
                        vol   = int(m_trk.module_ini/cf.n_drift_volumes)
            
                        module_ini = m_trk.module_ini
                        vdrift = lar.drift_velocity(module_ini)
            
                        z0_light, P0, P1, trk_dir = get_trk_clus_param(m_clus, m_trk, vdrift)
                        #print('P0= ',P0, ' direction = ', trk_dir, "Z0= ", z0_light)
                        light_pred.set_track(P0, P1, trk_dir, m_clus.glob_chans)    

                        logL, is_decay = match_record[match]
                        #print('logL = ', logL, ' decay?', is_decay)
                        m_trk.set_pds_logL(logL)                   
                        m_clus.point_impact[vol].extend(light_pred.pds_impact_point_per_channels)
            

                        m_trk.set_AV_extrapolation(light_pred.t0, light_pred.t1, light_pred.ta, light_pred.tb)
                        m_trk.is_decay_from_light = is_decay

                        if(is_decay):
                            m_clus.dist_closest[vol].extend(light_pred.track_dk_closest_distance_per_channels)
                            m_clus.point_closest[vol].extend(light_pred.track_dk_closest_point_per_channels)
                            m_clus.costheta_closest[vol].extend(light_pred.track_dk_closest_costheta_per_channels)
                            
                            m_clus.n_predicted[vol].extend(light_pred.dk_prediction_per_channels)
                            m_clus.n_geo_predicted[vol].extend(light_pred.geo_dk_prediction_per_channels)
                
                        else:
                            m_clus.dist_closest[vol].extend(light_pred.track_closest_distance_per_channels)
                            m_clus.point_closest[vol].extend(light_pred.track_closest_point_per_channels)
                            m_clus.costheta_closest[vol].extend(light_pred.track_closest_costheta_per_channels)
                
                            m_clus.n_predicted[vol].extend(light_pred.prediction_per_channels)
                            m_clus.n_geo_predicted[vol].extend(light_pred.geo_prediction_per_channels)

                
            """
            print("CLUSTER OF LIGHT")
            [c.dump() for c  in m_clusters]

            print('TRACKS')
            [t.dump() for t in m_trks]

            modules = [t.module_ini for t in m_trks]
            vdrifts = [lar.drift_velocity(m) for m in modules]
            t0s = [m_clusters[0].timestamp - dc.evt_list[-1].delay_charge_time[m] for m in modules]
            z0s = [cf.drift_direction[m]*t*v for m,t,v in zip(modules, t0s, vdrifts)]
            #plot.plot_test_track_3D(m_trks, z0s, option=None, to_be_shown=True)
            logLs = [match_record[m+id_trk3d_shift][0]  for m in matches if m < ntracks]

            
            plot.plot_multiple_track_pds_test(m_trks, z0s, m_clusters[0], logLs, option=None, to_be_shown=True)
            """
    print('\n\nNumber of matched Tracks = ', n_matched_track, '!!!')


def track_light_log_likelihood(clus, P0, P1, trk_dir, light_pred):
    cath_channels = [c for c in clus.glob_chans if cf.pds_modules_type[dc.chmap_pds[c].module]=="Cathode"]


    data_meas = [0 for x in range(cf.n_pds_tot_channels) if cf.pds_modules_type[dc.chmap_pds[x].module]=="Cathode"]

    for c, q in zip(clus.glob_chans, clus.npes):
        if cf.pds_modules_type[dc.chmap_pds[c].module]=="Cathode":
            data_meas[c] = q

    
    light_pred.set_track(P0, P1, trk_dir, [x for x in range(cf.n_pds_tot_channels) if cf.pds_modules_type[dc.chmap_pds[x].module]=="Cathode"])

    predictions = light_pred.prediction_per_channels
    dk_predictions = light_pred.dk_prediction_per_channels
    
    logL, dk_logL = 0, 0
    n_chan = 0
    debug, debug_dk = [], []
    for npe, pred, dk_pred in zip(data_meas, predictions, dk_predictions):
        debug.append(npe * np.log(pred) - pred - gammaln(npe + 1))
        debug_dk.append(npe * np.log(dk_pred) - dk_pred - gammaln(npe + 1))
        logL += npe * np.log(pred) - pred - gammaln(npe + 1)
        dk_logL += npe * np.log(dk_pred) - dk_pred - gammaln(npe + 1)
        
    logL *= -1.
    dk_logL *= -1.

    
    #print(f'Full track {logL:.3f} Decay track {dk_logL:.3f}')
    #print('ratio===', dk_logL/logL)
    
    if(dk_logL < logL):
        if(dk_logL/logL < 0.9):
            #print([f"{x:.1f}" for x in debug_dk])
            return dk_logL, True, logL
    #print([f"{x:.1f}" for x in debug])
    return logL, False, dk_logL

    

