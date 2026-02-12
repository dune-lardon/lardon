import lardon.config as cf
import lardon.data_containers as dc
import lardon.lar_param as lar
import lardon.track_timing as tmg


import numpy as np
from rtree import index

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

    n_trk = 0

    anode_time_tol_bef = dc.reco['pds']['tpc_matching']['anode_crosser']['time_tol_bef'] #in mus
    anode_time_tol_aft = dc.reco['pds']['tpc_matching']['anode_crosser']['time_tol_aft'] #in mus
    
    cathode_time_tol_bef = dc.reco['pds']['tpc_matching']['cathode_crosser']['time_tol_bef'] #in mus
    
    cathode_time_tol_aft = dc.reco['pds']['tpc_matching']['cathode_crosser']['time_tol_aft'] #in mus

    
    unknown_time_tol_aft = dc.reco['pds']['tpc_matching']['unknown']['time_tol_aft'] #in mus
    unknown_time_tol_bef = dc.reco['pds']['tpc_matching']['unknown']['time_tol_bef'] #in mus

    min_cluster_size = dc.reco['pds']['tpc_matching']['min_cluster_size']
    v_drift = [lar.drift_velocity(m) for m in range(cf.n_module)]
    
    """
    print('LIGHT - TRK MATCHING time tolerances: ')
    print('anode crossers:', anode_time_tol)
    print('cathode crossers:', cathode_time_tol)
    print('unknown:', unknown_time_tol)
    """
    
    """ create  Rtree, now filled with the found clusters """
    pties = index.Property()
    pties.dimension = 2

    rtree = index.Index(properties=pties)
        
    #debug = []
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
        if(is_anode_crosser and is_anode_crosser):
            trk_type = 1
        elif(is_anode_crosser and not is_anode_crosser):
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

            #print(trk.ID_3D, " at ", trk_start, "has ", len(pds_overlaps))
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

                for pds_ch, pds_q in zip(clus.glob_chans, clus.charges):
                    pds_module = dc.chmap_pds[pds_ch].module
                
                    dist, pt_track, pt_pds, belongToTrack = dist_trk_to_pds(t, pds_module)
                    
                    clus.dist_closest[vol].append(dist)
                    clus.point_closest[vol].append(pt_track)
                    clus.point_impact[vol].append(pt_pds)
                    clus.point_closest_is_extrapolated[vol].append(~belongToTrack)
         

    print('\n---->>>> Number of pds-matched tracks: ', n_trk)        


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


def dist_trk_to_pds(trk, pds_mod):
    t0 = np.array([trk.ini_x, trk.ini_y, trk.ini_z+trk.z0_light])
    t1 = np.array([trk.end_x, trk.end_y, trk.end_z+trk.z0_light])
    tdir = t1-t0
    
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
    
    dist, pt_track, pt_pds, is_extrap = closest_line_to_pds(t0, tdir, pds_bounds, tpc_bounds)
    return dist, pt_track, pt_pds, is_extrap

def closest_point_on_line(p0, d, point):
    t = np.dot(point - p0, d) / np.dot(d, d)
    return p0 + t * d, t

def closest_distance_line_segment(p0, d, a, b):
    ab = b - a
    dab = np.dot(d, ab)
    dda = np.dot(d, d)
    aba = np.dot(ab, ab)
    ap = a - p0

    denom = dda * aba - dab * dab

    # Near-parallel case: check endpoints
    if abs(denom) < 1e-12:
        P1, t1 = closest_point_on_line(p0, d, a)
        d1 = np.linalg.norm(P1 - a)
        P2, t2 = closest_point_on_line(p0, d, b)
        d2 = np.linalg.norm(P2 - b)
        if d1 < d2:
            return d1, P1, a, t1
        else:
            return d2, P2, b, t2

    # General case
    t = (dab*np.dot(ap,ab) - aba*np.dot(ap,d)) / denom
    u = (dab*t + np.dot(ap,ab)) / aba

    if u < 0:
        P_seg = a
        P_line, t = closest_point_on_line(p0, d, a)
    elif u > 1:
        P_seg = b
        P_line, t = closest_point_on_line(p0, d, b)
    else:
        P_seg = a + u * ab
        P_line = p0 + t * d

    dist = np.linalg.norm(P_line - P_seg)
    return dist, P_line, P_seg, t

def clip_line_to_box(p0, d, box):
    xmin, xmax, ymin, ymax, zmin, zmax = box
    bounds = [(xmin,xmax), (ymin,ymax), (zmin,zmax)]
    tmin = -np.inf
    tmax = np.inf

    for coord in range(3):
        p = p0[coord]
        di = d[coord]
        mn, mx = bounds[coord]

        if abs(di) < 1e-12:
            # parallel: must be inside slab
            if p < mn or p > mx:
                return None, None
            continue

        # compute intersection parameters
        t1 = (mn - p) / di
        t2 = (mx - p) / di
        t_low, t_high = min(t1, t2), max(t1, t2)

        tmin = max(tmin, t_low)
        tmax = max(min(tmax, t_high), tmin)

        if tmin > tmax:
            return None, None

    return tmin, tmax


def closest_line_to_pds(p0, d, rect_bounds, volume_bounds):
    """
    p0, d: line origin and direction
    rect_bounds = PDS: (xmin, xmax, ymin, ymax, zmin, zmax) rectangle with one collapsed axis
    volume_bounds = TPC: 3D bounding box inside which the closest point must lie

    Returns:
        distance
        closest point on line (restricted to volume)
        closest point on rectangle
        whether the closest point lies on the track segment (t in [0,1])
    """

    t0, t1 = clip_line_to_box(p0, d, volume_bounds)
    if t0 is None:
        return None, None, None, False  # line never enters volume

    xmin, xmax, ymin, ymax, zmin, zmax = rect_bounds
    flat_x = xmin == xmax
    flat_y = ymin == ymax
    flat_z = zmin == zmax


    # Check intersection with rectangle interior
    def check_plane_intersection(coord_idx, coord_val, lo1, hi1, lo2, hi2):
        di = d[coord_idx]
        if abs(di) < 1e-12:
            return None
        t = (coord_val - p0[coord_idx]) / di
        I = p0 + t * d
        if lo1 <= I[(coord_idx+1)%3] <= hi1 and lo2 <= I[(coord_idx+2)%3] <= hi2:
            return t, I
        return None

    intersect = None

    if flat_x:
        intersect = check_plane_intersection(0, xmin, ymin, ymax, zmin, zmax)
    elif flat_y:
        intersect = check_plane_intersection(1, ymin, xmin, xmax, zmin, zmax)
    else:
        intersect = check_plane_intersection(2, zmin, xmin, xmax, ymin, ymax)

    if intersect:
        t_hit, I = intersect
        # enforce volume bounds
        if t_hit < t0:
            I = p0 + t0*d
        elif t_hit > t1:
            I = p0 + t1*d
        return 0.0, I, I, (0 <= t_hit <= 1)


    # Build rectangle corners and normal
    if flat_x:
        x = xmin
        corners = [
            np.array([x, ymin, zmin]),
            np.array([x, ymin, zmax]),
            np.array([x, ymax, zmin]),
            np.array([x, ymax, zmax]),
        ]
        normal = np.array([1,0,0])
        plane_val = x

    elif flat_y:
        y = ymin
        corners = [
            np.array([xmin, y, zmin]),
            np.array([xmax, y, zmin]),
            np.array([xmin, y, zmax]),
            np.array([xmax, y, zmax]),
        ]
        normal = np.array([0,1,0])
        plane_val = y

    else:
        z = zmin
        corners = [
            np.array([xmin, ymin, z]),
            np.array([xmax, ymin, z]),
            np.array([xmin, ymax, z]),
            np.array([xmax, ymax, z]),
        ]
        normal = np.array([0,0,1])
        plane_val = z

    edges = [(0,1),(0,2),(3,1),(3,2)]


    # Project line to plane (interior projection test)
    denom = np.dot(d, normal)
    if abs(denom) > 1e-12:
        t_plane = (plane_val - np.dot(normal, p0)) / denom
        P_plane = p0 + t_plane*d

        inside = False
        if flat_x:
            inside = (ymin <= P_plane[1] <= ymax and zmin <= P_plane[2] <= zmax)
        elif flat_y:
            inside = (xmin <= P_plane[0] <= xmax and zmin <= P_plane[2] <= zmax)
        else:
            inside = (xmin <= P_plane[0] <= xmax and ymin <= P_plane[1] <= ymax)

        if inside:
            # Clamp t to volume
            t_clamp = min(max(t_plane, t0), t1)
            P_line = p0 + t_clamp*d
            P_rect = P_plane
            dist = np.linalg.norm(P_line - P_rect)
            return dist, P_line, P_rect, (0 <= t_clamp <= 1)


    # Edge distances
    best = (float('inf'), None, None, None)

    for i, j in edges:
        a, b = corners[i], corners[j]
        dist, Pl, Pr, t = closest_distance_line_segment(p0, d, a, b)

        # clamp line point to volume
        if t < t0:
            Pl = p0 + t0*d
            dist = np.linalg.norm(Pl - Pr)
            t = t0
        elif t > t1:
            Pl = p0 + t1*d
            dist = np.linalg.norm(Pl - Pr)
            t = t1

        if dist < best[0]:
            best = (dist, Pl, Pr, t)


    # Corner distances
    for c in corners:
        Pl, t = closest_point_on_line(p0, d, c)
        # clamp
        if t < t0:
            Pl = p0 + t0*d
            dist = np.linalg.norm(Pl - c)
            t = t0
        elif t > t1:
            Pl = p0 + t1*d
            dist = np.linalg.norm(Pl - c)
            t = t1
        else:
            dist = np.linalg.norm(Pl - c)

        if dist < best[0]:
            best = (dist, Pl, c, t)

    dist, Pl, Pr, t = best
    return dist, Pl, Pr, (0 <= t <= 1)


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
 
