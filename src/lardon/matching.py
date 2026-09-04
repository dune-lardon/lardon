import lardon.config as cf
import lardon.data_containers as dc
import lardon.lar_param as lar
import lardon.track_timing as tmg

import lardon.light_prediction as lp

import numpy as np
from rtree import index
from scipy.special import gammaln

def set_track_pds_matching(trk, clus, trk_id_shift):

    trk.match_pds_cluster = clus.ID
    trk_vol   = int(trk.module_ini/cf.n_drift_volumes)
    clus.match_trk3D[trk_vol] = trk.ID_3D

    if(trk.is_cathode_crosser == True and trk.cathode_crosser_ID >=0 ):
        other_trk = dc.tracks3D_list[trk.cathode_crosser_ID-trk_id_shift]
        other_trk.match_pds_cluster = clus.ID
        other_trk_vol = int(other_trk.module_ini/cf.n_drift_volumes)
        other_trk_idx   = other_trk.ID_3D
        clus.match_trk3D[other_trk_vol] = other_trk_idx

    

def matching_trk_pds():
    if(len(dc.tracks3D_list)== 0 or len(dc.pds_cluster_list) == 0):
        return

    light_pred = lp.light_prediction(1000)

    n_trk = 0

    anode_time_tol_bef = dc.reco['pds']['tpc_matching']['anode_crosser']['time_tol_bef'] #in mus
    anode_time_tol_aft = dc.reco['pds']['tpc_matching']['anode_crosser']['time_tol_aft'] #in mus
    
    cathode_time_tol_bef = dc.reco['pds']['tpc_matching']['cathode_crosser']['time_tol_bef'] #in mus
    
    cathode_time_tol_aft = dc.reco['pds']['tpc_matching']['cathode_crosser']['time_tol_aft'] #in mus

    
    unknown_time_tol_aft = dc.reco['pds']['tpc_matching']['unknown']['time_tol_aft'] #in mus
    unknown_time_tol_bef = dc.reco['pds']['tpc_matching']['unknown']['time_tol_bef'] #in mus

    min_cluster_size = dc.reco['pds']['tpc_matching']['min_cluster_size']
    v_drift = [lar.drift_velocity(m) for m in range(cf.n_module)]

    pds_ped = [dc.evt_list[-1].noise_pds_raw.ped_mean[dc.chmap_pds[gch].daqch] for gch in range(cf.n_pds_tot_channels)]
    
    """ create  Rtree, now filled with the found clusters """
    pties = index.Property()
    pties.dimension = 2

    rtree = index.Index(properties=pties)
        

    ''' filling the R-tree with the light clusters on y axis = 0'''
    for c in dc.pds_cluster_list:
        if(np.any(np.array(c.match_trk3D)>=0) or c.match_single >=0 ):
            continue
        if(c.size < min_cluster_size):
            continue
        start = c.timestamp
        idx   = c.ID
        rtree.insert(idx, (start, 0, start, 0))

        
    ''' filling the R-tree with the 3D tracks on y axis =1,2,3 '''
    for t in dc.tracks3D_list:
        if(t.match_pds_cluster >= 0):
            continue
        start = t.timestamp
        stop  = t.timestamp_r
        volume = int(t.module_ini/cf.n_drift_volumes)
        idx   = t.ID_3D
        is_anode_crosser = t.is_anode_crosser
        is_cathode_crosser = t.is_cathode_crosser

        """ to add later: is track trigger trk_type = 0"""        
        if(is_anode_crosser and is_cathode_crosser):
            trk_type = 1
        elif(is_anode_crosser and not is_cathode_crosser):
            trk_type = 2
        elif(not is_anode_crosser and is_cathode_crosser):
            trk_type = 3
        else: #unresolved case
            trk_type = 4



        rtree.insert(idx, (start, trk_type, stop, trk_type))
    
    
    id_trk3d_shift = dc.n_tot_trk3d
    id_cluster_shift = dc.n_tot_pds_clusters

    
    """ search for the 3D track - light clusters """
    anode_cathode_tracks = [t for t in dc.tracks3D_list if t.is_anode_crosser and t.is_cathode_crosser]
    anode_tracks = [t for t in dc.tracks3D_list if t.is_anode_crosser and not t.is_cathode_crosser]
    cathode_tracks = [t for t in dc.tracks3D_list if not t.is_anode_crosser and t.is_cathode_crosser]
    unknown_tracks = [t for t in dc.tracks3D_list if not t.is_cathode_crosser and not t.is_anode_crosser]


    """
    print('Nb of anode-cathode Xs', len(anode_cathode_tracks))
    print('Nb of anode Xs', len(anode_tracks))
    print('Nb of cathode Xs', len(cathode_tracks))
    print('Nb of unknowns ', len(unknown_tracks))
    """

    
    
    for trk_sel, time_tol_before, time_tol_after, name in zip([anode_cathode_tracks, anode_tracks, cathode_tracks, unknown_tracks], [anode_time_tol_bef, anode_time_tol_bef, cathode_time_tol_bef, unknown_time_tol_bef], [anode_time_tol_aft, anode_time_tol_aft, cathode_time_tol_aft, unknown_time_tol_aft], ['anode-cathode', 'anode', 'cathode','unknown']):
        #print("\n=======\ntesting ", name)
        for trk in trk_sel:
            if(trk.match_pds_cluster >= 0):
                continue
        
            trk_start = trk.timestamp
            trk_stop  = trk.timestamp_r
            trk_vol   = int(trk.module_ini/cf.n_drift_volumes)
            
            pds_overlaps = list(rtree.intersection((trk_start - time_tol_before[trk_vol], 0, trk_stop + time_tol_after[trk_vol], 0)))
            pds_overlaps = [ov for ov in pds_overlaps if all(x < 0 for x in dc.pds_cluster_list[ov-id_cluster_shift].match_trk3D)]

                
            """ only one match possible """
            if(len(pds_overlaps) != 1):
                continue

            clus = dc.pds_cluster_list[pds_overlaps[0]-id_cluster_shift]
            
            if(clus.match_trk3D[trk_vol] >= 0 or clus.match_single >=0 ):
                continue
            set_track_pds_matching(trk, clus, id_trk3d_shift)

            delay = clus.timestamp - trk_start
            
            """ debug """            
            #print('[',name,'] Potential Track-light match ! track ID ', trk.ID_3D)
            #print('---> with delay ', delay)

            
            n_trk += 1

            tracks = [trk]
            if(trk.is_cathode_crosser == True and trk.cathode_crosser_ID >=0 ):
                tracks.append(dc.tracks3D_list[trk.cathode_crosser_ID-id_trk3d_shift])
                
                
            
            for t in tracks:
                vol = int(t.module_ini/cf.n_drift_volumes)
                t.set_times_from_light(clus.timestamp, v_drift[t.module_ini])
                #t.dump()
                

                #print('\nTrack ', t.ID_3D)                
                P0, P1, tdir = tmg.track_point_direction(t, t.z0_light)
                logL, is_decay = track_light_log_likelihood(clus, P0, P1, tdir, light_pred)

                """
                print('track extrapolation: ', light_pred.t0, light_pred.t1)
                print('extrapolated track goes from ',(light_pred.P0 + light_pred.t0*light_pred.tdir),'to', (light_pred.P0 + light_pred.t1*light_pred.tdir))

                print('from muon decay test: ', light_pred.ta, light_pred.tb)
                print('extrapolated dk track goes from ',(light_pred.P0 + light_pred.ta*light_pred.tdir),'to', (light_pred.P0 + light_pred.tb*light_pred.tdir))
                """
                #print(logL,'---> compatible with decay track ? ', is_decay)
                
                t.set_pds_logL(logL)
                
                light_pred.set_track(P0, P1, tdir, clus.glob_chans) 
                clus.point_impact[vol].extend(light_pred.pds_impact_point_per_channels)
                t.set_AV_extrapolation(light_pred.t0, light_pred.t1, light_pred.ta, light_pred.tb)

                """
                if(trk.is_cathode_crosser == True and trk.cathode_crosser_ID >=0 ):
                    other_trk = dc.tracks3D_list[trk.cathode_crosser_ID-id_trk3d_shift]
                    other_trk.set_pds_logL(logL)
                    other_trk.set_AV_extrapolation(light_pred.t0, light_pred.t1, light_pred.ta, light_pred.tb)
                """
                
                if(is_decay):
                    clus.dist_closest[vol].extend(light_pred.track_dk_closest_distance_per_channels)
                    clus.point_closest[vol].extend(light_pred.track_dk_closest_point_per_channels)
                    clus.costheta_closest[vol].extend(light_pred.track_dk_closest_costheta_per_channels)

                    clus.n_predicted[vol].extend(light_pred.dk_prediction_per_channels)
                    clus.n_geo_predicted[vol].extend(light_pred.geo_dk_prediction_per_channels)
                
                    #trk.set_AV_extrapolation(light_pred.ta, light_pred.tb)
                    t.is_decay_from_light = True
                else:
                    clus.dist_closest[vol].extend(light_pred.track_closest_distance_per_channels)
                    clus.point_closest[vol].extend(light_pred.track_closest_point_per_channels)
                    clus.costheta_closest[vol].extend(light_pred.track_closest_costheta_per_channels)

                    clus.n_predicted[vol].extend(light_pred.prediction_per_channels)
                    clus.n_geo_predicted[vol].extend(light_pred.geo_prediction_per_channels)
                
                
                    

                """
                for pds_ch, pds_q, pds_max_adc in zip(clus.glob_chans, clus.npes, clus.max_adcs):
                    pds_module = dc.chmap_pds[pds_ch].module

                    if(pds_max_adc > pds_ped[pds_ch]-10):
                        saturate = True
                        sigma = 0.5
                    else:
                        saturate = False
                        sigma = 0.1
                        
                    if(prev_module != pds_module):
                        dist, pt_track, pt_pds, belongToTrack, npe_predicted = dist_trk_to_pds(t, pds_module)
                        if(cf.pds_modules_type[pds_module] == 'Cathode'):
                            chi2 += (npe_predicted-pds_q)**2/sigma**2
                            nchan += 1
                            
                    clus.dist_closest[vol].append(dist)
                    clus.point_closest[vol].append(pt_track)
                    clus.point_impact[vol].append(pt_pds)
                    #clus.solid_angle_closest[vol].append(solid_angle_close)
                    #clus.solid_angle_integrated[vol].append(solid_angle_tot)
                    clus.n_predicted[vol].append(npe_predicted)
                    clus.point_closest_is_extrapolated[vol].append(~belongToTrack)
                    #print(pds_ch, "::::", dist, pt_track, pt_pds, belongToTrack, 'predicted : ', npe_predicted, ' vs ', pds_q)
                    if(cf.pds_modules_type[pds_module] == 'Cathode'):
                        chi2 += (npe_predicted-pds_q)**2/sigma**2
                        nchan += 1
                    prev_module = pds_module
                #print('Chi2 for tracks = ', chi2, " chi2/NDF = ", chi2/nchan)
                if(nchan > 0):
                    chi2_ndf = chi2/nchan
                else:
                    chi2_ndf = -1
                """
                #trk.set_pds_chi2(logL)
    print('Number of pds-matched tracks: ', n_trk)        




def track_light_log_likelihood(clus, P0, P1, trk_dir, light_pred):

    data_meas = [0 for x in range(cf.n_pds_tot_channels) if cf.pds_modules_type[dc.chmap_pds[x].module]=="Cathode"]

    for c, q in zip(clus.glob_chans, clus.npes):
        if cf.pds_modules_type[dc.chmap_pds[c].module]=="Cathode":
            data_meas[c] = q

    light_pred.set_track(P0, P1, trk_dir,  [x for x in range(cf.n_pds_tot_channels) if cf.pds_modules_type[dc.chmap_pds[x].module]=="Cathode"])
    predictions = light_pred.prediction_per_channels
    dk_predictions = light_pred.dk_prediction_per_channels
    logL, dk_logL = 0, 0
    n_chan = 0
    
    for npe, pred, dk_pred in zip(data_meas, predictions, dk_predictions):
        logL += npe * np.log(pred) - pred - gammaln(npe + 1)
        dk_logL += npe * np.log(dk_pred) - dk_pred - gammaln(npe + 1)

    logL *= -1.
    dk_logL *= -1.

    #print(f'Full track {logL:.3f} Decay track {dk_logL:.3f}')
    #print('ratio===', dk_logL/logL)
    
    if(dk_logL < logL):
        if(dk_logL/logL < 0.9):
            return dk_logL, True
    return logL, False


    
def dist_trk_to_pds(trk, pds_mod):
    t0 = np.array([trk.ini_x, trk.ini_y, trk.ini_z+trk.z0_light])
    t1 = np.array([trk.end_x, trk.end_y, trk.end_z+trk.z0_light])
    tdir = t1-t0

    eff = cf.pds_eff[pds_mod]
    
    x_center = cf.pds_x_center[pds_mod]
    y_center = cf.pds_y_center[pds_mod]
    z_center = cf.pds_z_center[pds_mod]
    x_length = cf.pds_x_length[pds_mod]/2.
    y_length = cf.pds_y_length[pds_mod]/2.
    z_length = cf.pds_z_length[pds_mod]/2.

    pds_bounds = (x_center-x_length, x_center+x_length,
                  y_center-y_length, y_center+y_length,
                  z_center-z_length, z_center+z_length)

    tpc_bounds = (min([cf.x_boundaries[i][0] for i in range(cf.n_module)]), max([cf.x_boundaries[i][1] for i in range(cf.n_module)]),
                 min([cf.y_boundaries[i][0] for i in range(cf.n_module)]), max([cf.y_boundaries[i][1] for i in range(cf.n_module)]),
                 min(cf.anode_z), max(cf.anode_z))

    

    dist, pt_track, pt_pds, is_extrap = track_closest_point_to_pds(t0, tdir, pds_bounds, tpc_bounds)

    npe_predicted = predicted_npe_geometric(t0, tdir, eff, pds_bounds, tpc_bounds, debug=False)
    return dist, pt_track, pt_pds, is_extrap, npe_predicted



#  clip line inside the TPC volume
def clip_line_to_box(P0, d, box):
    bxmin, bxmax, bymin, bymax, bzmin, bzmax = box
    t0, t1 = -np.inf, np.inf

    for i, (p, di, mn, mx) in enumerate([
            (P0[0], d[0], bxmin, bxmax),
            (P0[1], d[1], bymin, bymax),
            (P0[2], d[2], bzmin, bzmax)]):
        if abs(di) < 1e-12:
            # Line parallel: must lie inside slab
            if p < mn or p > mx:
                return None, None
        else:
            tmin = (mn - p) / di
            tmax = (mx - p) / di
            if tmin > tmax:
                tmin, tmax = tmax, tmin
            t0 = max(t0, tmin)
            t1 = min(t1, tmax)

        if t0 > t1:
            return None, None

    return t0, t1

def track_closest_point_to_pds(P0, d, bounds, tpc_bound):
    """
    Compute closest distance between an infinite line (restricted to TPC volume)
    and an axis-aligned rectangular panel (possibly flat).

    P0, d: infinite line (param t)
    bounds: panel bounds (xmin, xmax, ymin, ymax, zmin, zmax)
    tpc_bound: volume bounds inside which the line point must lie
    """

    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    txmin, txmax, tymin, tymax, tzmin, tzmax = tpc_bound


    # Normalize direction
    d = d / np.linalg.norm(d)



    # Clip infinite line to TPC
    t0, t1 = clip_line_to_box(P0, d, tpc_bound)

    if t0 is None:
        # Line does not enter the TPC
        return 9999, [9999,9999,9999], [9999, 9999, 9999], False, 0.


    # build pds panel face
    faces = []
    center = []
    if xmin == xmax:  # YZ plane
        faces.append(('x', xmin, ymin, ymax, zmin, zmax))
        center = [xmin, (ymax-ymin)/2, (zmax-zmin)/2]
    if ymin == ymax:  # XZ plane
        faces.append(('y', ymin, xmin, xmax, zmin, zmax))
        center = [(xmax-xmin)/2, ymin, (zmax-zmin)/2]
    if zmin == zmax:  # XY plane
        faces.append(('z', zmin, xmin, xmax, ymin, ymax))
        center = [(xmax-xmin)/2, (ymax-ymin)/2, zmin]

    # clamp line to rectangle
    def clamp_to_rect(P, xmin, xmax, ymin, ymax, zmin, zmax):
        return np.array([
            np.clip(P[0], xmin, xmax),
            np.clip(P[1], ymin, ymax),
            np.clip(P[2], zmin, zmax)
        ])


    # clamp line point to TPC volume
    def clamp_line_to_tpc_point(P0, d, t_hit):
        if t_hit < t0:
            t_hit = t0
        if t_hit > t1:
            t_hit = t1
        return t_hit, P0 + t_hit * d


    # closest point between line and segment
    def closest_point_line_segment(P0, d, A, B):
        AB = B - A
        AP = P0 - A
        dAB = np.dot(d, AB)
        ABAB = np.dot(AB, AB)
        dd = np.dot(d, d)
        denom = dd * ABAB - dAB * dAB

        # Line parallel to segment
        if abs(denom) < 1e-12:
            t = np.dot(d, A - P0) / dd
            C = P0 + t * d
            s = np.dot(AB, C - A) / ABAB
            s = np.clip(s, 0, 1)
            Dp = A + s * AB
            return C, Dp

        t = (np.dot(d, AP) * ABAB - np.dot(AB, AP) * dAB) / denom
        C = P0 + t * d

        # Clamp s to segment
        s = (np.dot(d, AP) + t * dAB) / ABAB
        s = np.clip(s, 0, 1)
        Dp = A + s * AB

        return C, Dp


    # Compute closest distance over all faces + edges
    best_dist = np.inf
    best_cl = None
    best_cp = None


    
    for axis, c, a1min, a1max, a2min, a2max in faces:
        t_hit = 0
        # ----- Line-plane projection -----
        if axis == 'x':
            if abs(d[0]) < 1e-12:
                Pproj = P0.copy()
                Pproj[0] = c
            else:
                t_hit = (c - P0[0]) / d[0]
                Pproj = P0 + t_hit * d
            Cp = clamp_to_rect(Pproj, c, c, a1min, a1max, a2min, a2max)

        elif axis == 'y':
            if abs(d[1]) < 1e-12:
                Pproj = P0.copy()
                Pproj[1] = c
            else:
                t_hit = (c - P0[1]) / d[1]
                Pproj = P0 + t_hit * d
            Cp = clamp_to_rect(Pproj, a1min, a1max, c, c, a2min, a2max)

        else:  # axis == 'z'
            if abs(d[2]) < 1e-12:
                Pproj = P0.copy()
                Pproj[2] = c
            else:
                t_hit = (c - P0[2]) / d[2]
                Pproj = P0 + t_hit * d
            Cp = clamp_to_rect(Pproj, a1min, a1max, a2min, a2max, c, c)

        # Closest point on line
        t_line = np.dot(d, Cp - P0)

        
        is_inside_track = (0 <= t_hit <= 1)

        t_line, Cl = clamp_line_to_tpc_point(P0, d, t_line)

        dist = np.linalg.norm(Cl - Cp)
        if dist < best_dist:
            best_dist = dist
            best_cl = Cl
            best_cp = Cp

        # ---------- Also check the 4 edges ----------
        if axis == 'x':
            A = np.array([c, a1min, a2min])
            B = np.array([c, a1max, a2min])
            C = np.array([c, a1max, a2max])
            D = np.array([c, a1min, a2max])
        elif axis == 'y':
            A = np.array([a1min, c, a2min])
            B = np.array([a1max, c, a2min])
            C = np.array([a1max, c, a2max])
            D = np.array([a1min, c, a2max])
        else:
            A = np.array([a1min, a2min, c])
            B = np.array([a1max, a2min, c])
            C = np.array([a1max, a2max, c])
            D = np.array([a1min, a2max, c])

        edges = [(A, B), (B, C), (C, D), (D, A)]

        for e0, e1 in edges:
            Cl_e, Cp_e = closest_point_line_segment(P0, d, e0, e1)

            # clamp line point to TPC
            t_e = np.dot(d, Cl_e - P0)
            t_e, Cl_e = clamp_line_to_tpc_point(P0, d, t_e)

            dist_e = np.linalg.norm(Cl_e - Cp_e)
            if dist_e < best_dist:
                best_dist = dist_e
                best_cl = Cl_e
                best_cp = Cp_e


    #solid_angle_closest = compute_solid_angle(bounds, best_cl)

    return best_dist, best_cl, best_cp, is_inside_track#, solid_angle_closest


def solid_angle_triangle(a, b, c):
        la = np.linalg.norm(a)
        lb = np.linalg.norm(b)
        lc = np.linalg.norm(c)
        num = np.dot(a, np.cross(b, c))
        den = la*lb*lc + np.dot(a,b)*lc + np.dot(a,c)*lb + np.dot(b,c)*la
        return 2*np.arctan2(num, den)


def compute_solid_angle(bounds, point, l_abs, eff, nphoton):
    xmin, xmax, ymin, ymax, zmin, zmax = bounds

    if xmin == xmax:  # YZ panel
        A = np.array([xmin, ymin, zmin])
        B = np.array([xmin, ymax, zmin])
        C = np.array([xmin, ymax, zmax])
        D = np.array([xmin, ymin, zmax])
        M = np.array([xmin, (ymax+ymin)/2, (zmax+zmin)/2])
    elif ymin == ymax:  # XZ panel
        A = np.array([xmin, ymin, zmin])
        B = np.array([xmax, ymin, zmin])
        C = np.array([xmax, ymin, zmax])
        D = np.array([xmin, ymin, zmax])
        M = np.array([(xmax+xmin)/2, ymin, (zmax+zmin)/2])
        
    else:  # XY panel
        A = np.array([xmin, ymin, zmin])
        B = np.array([xmax, ymin, zmin])
        C = np.array([xmax, ymax, zmin])
        D = np.array([xmin, ymax, zmin])
        M = np.array([(xmax+xmin)/2, (ymax+ymin)/2, zmin])
        
    rA = A - point
    rB = B - point
    rC = C - point
    rD = D - point
    rM = M - point

    dist = np.linalg.norm(rM)

    solid_angle = abs(solid_angle_triangle(rA, rB, rC)) + abs(solid_angle_triangle(rA, rC, rD))

    return nphoton*eff*np.exp(-dist/l_abs)*abs(solid_angle)/4./np.pi

def predicted_npe_geometric(P0, d, eff, pds_bounds, tpc_bounds, debug=False):

    """
    Compute the solid angle integrated along the track segment inside the TPC
    with respect to all PDS panels.

    Returns
    -------
    total_integrated_solid_angle
        Integral of Omega(s) ds along the track
    """

    l_abs = 1000. #cm
    nphoton = 3.2e4 #/cm for muons at mip
    
    d = d / np.linalg.norm(d)
    t0, t1 = clip_line_to_box(P0, d, tpc_bounds)
    if t0 is None:
        return 0.0, 0.0


    # track total length inside TPC
    L = np.linalg.norm((P0 + t1*d) - (P0 + t0*d))
    n_points = int(L)
    ds = L / n_points
    ts = np.linspace(t0, t1, n_points)

    total_integral = 0.0

    npe = [compute_solid_angle(pds_bounds, P0+t*d, l_abs, eff, nphoton)*ds for t in ts]

    npe_tot = sum(npe)
    if(debug):
        print('track: t0 t1', t0, t1, ' L=', L, 'n_points=', n_points, ' ds= ', ds)
        print('would be from ', P0+t0*d, 'to',P0+t1*d)
        print([compute_solid_angle(pds_bounds, P0+t*d, l_abs, eff, nphoton)*ds for t in ts[:10]])
    
    return npe_tot
        


def  matching_sh_pds():
    n_sh = 0

    time_tol = dc.reco['pds']['matching']['time_tol'] #in mus
    
    """ create Rtree filled with the clusters and SH time """
    pties = index.Property()
    pties.dimension = 3

    rtree = index.Index(properties=pties)

        
    """ create SH Rtree to ensure the SH is isolated in space and time """
    sh_pties = index.Property()
    sh_pties.dimension = 3
    sh_rtree = index.Index(properties=sh_pties)

    id_cluster_shift = dc.n_tot_pds_clusters
    id_sh_shift = dc.n_tot_sh

    
    ''' filling the R-tree with the single hits on y axis 0'''
    for sh in dc.single_hits_list:
        if(sh.match_pds_cluster >= 0):
            continue
        start = sh.timestamp
        volume = int(sh.module/cf.n_drift_volumes)
        x,y,z = sh.X, sh.Y, sh.Z
        idx   = sh.ID_SH
        rtree.insert(idx, (start, 0, volume, start, 0, volume))
        sh_rtree.insert(idx, (x,y,start,x,y,start))
        

    ''' filling the R-tree with the light clusters on y axis 1'''
    for c in dc.pds_cluster_list:
        if(np.any(np.array(c.match_trk3D)>=0) or c.match_single >=0 ):
            continue

        if(c.size > 2): #TO BE CHANGED FOR PDHD !!            
            continue

        start = c.timestamp
        mod = dc.chmap_pds[c.glob_chans[0]].module
        idx   = c.ID
        rtree.insert(idx, (start, 1, mod, start, 1, mod))



    
    """ search for the single hits - light clusters """
    for sh in dc.single_hits_list:
        if(sh.match_pds_cluster >= 0):
            continue
        
        sh_start = sh.timestamp
        #sh_vol   = int(sh.module/cf.n_drift_volumes)
        sh_idx   = sh.ID_SH

        z_anode = cf.anode_z[sh.module]
        vdrift = lar.drift_velocity(sh.module)
        
        ''' maximum drift distance given the time window '''
        max_drift = cf.drift_length[sh.module]/ vdrift
        overlaps = list(rtree.intersection((sh_start-time_tol-max_drift, 1,0, sh_start+time_tol, 1,999)))
        
        free_overlaps = []

        #if(len(overlaps)):
            #print('\n max drift: ', max_drift, "nb of overlaps", len(overlaps))

            
        for ov in overlaps:
            clus = dc.pds_cluster_list[ov-id_cluster_shift]

            if(np.any(np.array(c.match_trk3D)>=0) or clus.match_single >=0):
                continue

            d = sh_pds_dist(sh, ov - id_cluster_shift)

            if(d < 70):
            
                mod = dc.chmap_pds[clus.glob_chans[0]].module
                start = clus.timestamp
                
                ov_pds = list(rtree.intersection((start-time_tol, 1,mod, start+max_drift+time_tol, 1,mod)))


                if(len(ov_pds) < 2):
                    pds_x, pds_y = cf.pds_x_centers[mod], cf.pds_y_centers[mod]
                    ov_sh = list(sh_rtree.intersection((pds_x-70, pds_y-70, start-time_tol, pds_x+70, pds_y+70, start+max_drift+time_tol)))

                    if(len(ov_sh) < 2):

                        print('\n\n\nPOTENTIAL MATCH!')
                        print("NB OF SH around : ", len(ov_sh))
                    
                        sh.dump()


                        print('\ncould be with ')
                        print('That module saw ', len(ov_pds), ' around that cluster')
                        print("distance: ", d)
                        clus.dump()
                
                        free_overlaps.append(ov)
            

        best_overlap = -1
        
        if(len(free_overlaps) == 1):
            best_overlap = free_overlaps[0]
        else:
            continue

        
        if(best_overlap >=0):
            
            n_sh += 1
            clus = dc.pds_cluster_list[best_overlap-id_cluster_shift]
            delay = sh_start - clus.timestamp        

            z_estimate = cf.anode_z[sh.module] - cf.drift_direction[sh.module]*(vdrift*delay)

            sh.match_pds_cluster = clus.ID
            sh.Z_from_light = z_estimate            
            clus.match_single = sh_idx

            for pds_ch in clus.glob_chans[::2]:
                module_chan = int(pds_ch/2)
                
                res = dist_sh_to_pds_side(sh, module_chan)
            
                clus.dist_closest_strip.extend([r[0] for r in res])
                clus.id_closest_strip.extend([r[1] for r in res])
                clus.point_impact.extend([r[2] for r in res])
                clus.point_closest_above.extend([r[3] for r in res])
                clus.point_closest.extend([r[4] for r in res])



    print('\n---->>>> Number of pds-matched single hits ', n_sh)

def sh_pds_dist(sh, cluster_idx):
    clus = dc.pds_cluster_list[cluster_idx]
    distances = []

    for pds_ch in clus.glob_chans:
        mod = dc.chmap_pds[pds_ch].module
        d = np.sqrt(pow(sh.X-cf.pds_x_centers[mod], 2) + pow(sh.Y-cf.pds_y_centers[mod], 2))
        distances.append(d)

    return min(distances)

def sh_closest_cluster(sh, cluster_idx, id_shift):
    clus_dist = []

    for idx in cluster_idx:
        d = sh_pds_dist(sh, idx-id_shift)
        
        clus_dist.append((idx, d))
        
    clus_dist_sorted = sorted(clus_dist, key=lambda tup: tup[1])

    return clus_dist_sorted[0][0]

def dist_sh_to_pds_side(sh, pds_chan):
    a = np.asarray([sh.X, sh.Y, sh.Z_from_light])
    all_b0, all_b1 = xarapucas_siPM_strips(pds_chan)    


    res = []
    for idx, (b0, b1) in enumerate(zip(all_b0, all_b1)):
         p, dist = closest_distance_point_to_line(a, np.asarray(b0), np.asarray(b1))
         a = a#.tolist() #sh
         p = p.tolist() #closest point on the xarapuxa side
         res.append((dist, idx+10*(pds_chan+1), p, above_xarapuca(pds_chan, a), a))


    sort_res = sorted(res, key=lambda tup: tup[0])
    return sort_res[:2]



def closest_distance_point_to_line(a, b0, b1):
    B = b1 - b0
    magB = np.linalg.norm(B)
    AB = b0 - a
    magAB = np.linalg.norm(AB)
    
    dot = np.dot(AB, B)
    denom = magB**2
    if(denom == 0):
        return None, 9999
    else:
        t = -dot/denom
        if(t<0):t=0
        if(t>1):t=1
        point = b0 + B*t
        l = np.linalg.norm(a-point)
        return point, l
 
