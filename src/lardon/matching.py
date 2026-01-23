import lardon.config as cf
import lardon.data_containers as dc
import lardon.lar_param as lar

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

    anode_time_tol = dc.reco['pds']['tpc_matching']['anode_crosser']['time_tol'] #in mus
    cathode_time_tol = dc.reco['pds']['tpc_matching']['cathode_crosser']['time_tol'] #in mus
    unknown_time_tol = dc.reco['pds']['tpc_matching']['unknown']['time_tol'] #in mus

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

        """ to add later: is track trigger """
        if(is_anode_crosser):
            trk_type = 1
        elif(is_cathode_crosser):
            trk_type = 2
        else:
            trk_type = 3
        rtree.insert(idx, (start, trk_type, stop, trk_type))
    
    
    id_trk3d_shift = dc.n_tot_trk3d
    id_cluster_shift = dc.n_tot_pds_clusters

    
    """ search for the 3D track - light clusters """
    anode_tracks = [t for t in dc.tracks3D_list if t.is_anode_crosser]
    cathode_tracks = [t for t in dc.tracks3D_list if t.is_cathode_crosser and not t.is_anode_crosser]
    unknown_tracks = [t for t in dc.tracks3D_list if not t.is_cathode_crosser and not t.is_anode_crosser]

    for trk_sel, time_tol, name in zip([anode_tracks, cathode_tracks, unknown_tracks], [anode_time_tol, cathode_time_tol, unknown_time_tol], ['anode', 'cathode','unknown']):
        for trk in trk_sel:
            if(trk.match_pds_cluster >= 0):
                continue
        
            trk_start = trk.timestamp
            trk_stop  = trk.timestamp_r
            trk_vol   = int(trk.module_ini/cf.n_drift_volumes)
            
            overlaps = list(rtree.intersection((trk_start - time_tol, 0, trk_stop + time_tol, 0)))
            overlaps = [ov for ov in overlaps if dc.pds_cluster_list[ov-id_cluster_shift].match_trk3D[trk_vol] < 0]
            
            """ only one match possible """
            if(len(overlaps) != 1):
                continue

            clus = dc.pds_cluster_list[overlaps[0]-id_cluster_shift]
            if(clus.match_trk3D[trk_vol] >= 0 or clus.match_single >=0 ):
                continue
            set_track_pds_matching(trk, clus, id_trk3d_shift)

            delay = clus.timestamp - trk_start
            
            """ debug """
            """
            print('[',name,'] Potential Track-light match')
            trk.dump()
            print('---> with delay ', delay)
            clus.dump()
            """
            n_trk += 1

            #for pds_ch in clus.glob_chans:
            for pds_ch, pds_q in zip(clus.glob_chans, clus.charges):
                pds_module = dc.chmap_pds[pds_ch].module
                
                dist, pt_track, pt_pds = dist_trk_to_pds(trk, pds_module)

                """
                print('---- pds channel: ', pds_ch, '= module', pds_module)
                print('closest distance: ', dist, ' charge ', pds_q)
                print('closest on track ', pt_track)
                print('closest on pds ', pt_pds)
                """
                
                """
                return         
                clus.dist_closest_strip.extend([r[0] for r in res])
                clus.id_closest_strip.extend([r[1] for r in res])
                clus.point_impact.extend([r[2] for r in res])
                clus.point_closest_above.extend([r[3] for r in res])
                clus.point_closest.extend([r[4] for r in res])
                """

    print('\n---->>>> nb of pds-matched tracks ', n_trk)        


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



    print('\n---->>>> nb of pds-matched single hits ', n_sh)

def extrapolate_trk_to_z(a0, a1, z_end):
    dx = a1[0] - a0[0]
    dy = a1[1] - a0[1]
    dz = a1[2] - a0[2]
    
    dzprime = z_end - a0[2]

    x = a0[0] + dx*dzprime/dz
    y = a0[1] + dy*dzprime/dz
    z = a0[2] + dz*dzprime/dz

    return np.array([x, y, z])


def above_xarapuca(ch, b):
    l_arap = cf.pds_length    

    x0, y0, z0 = cf.pds_x_centers[ch], cf.pds_y_centers[ch], cf.pds_z_centers[ch]

    if(cf.pds_modules_type[ch] == "Cathode"):
        if(b[0] < x0-l_arap or b[0] > x0+l_arap): return False
        elif(b[1] < y0-l_arap or b[1] > y0+l_arap): return False
        else: return True
        
def xarapucas_siPM_strips(ch):
    l_arap = cf.pds_length    

    x0, y0, z0 = cf.pds_x_centers[ch], cf.pds_y_centers[ch], cf.pds_z_centers[ch]

    if(cf.pds_modules_type[ch] == "Cathode"):
        xx = [x0-l_arap/2, x0-l_arap/2, x0-l_arap/2, x0, x0+l_arap/2, x0+l_arap/2, x0+l_arap/2, x0, x0-l_arap/2]
        yy = [y0+l_arap/2, y0, y0-l_arap/2, y0-l_arap/2, y0-l_arap/2, y0, y0+l_arap/2, y0+l_arap/2, y0+l_arap/2]
        zz = [z0 for x in range(len(xx))]
    
        b0 = [ [xx[i], yy[i], zz[i]] for i in range(8)]
        b1 = [ [xx[i+1], yy[i+1], zz[i+1]] for i in range(8)]
        
        return b0, b1

def dist_trk_to_pds(trk, pds_mod):
    t0 = np.array([trk.ini_x, trk.ini_y, trk.ini_z+trk.z0_corr])
    t1 = np.array([trk.end_x, trk.end_y, trk.end_z+trk.z0_corr])
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

    dist, pt_track, pt_pds = closest_line_to_pds(t0, tdir, pds_bounds)
    return dist, pt_track, pt_pds
    
def closest_line_to_pds(p0, d, bounds):
    """
    p0: 3-vector, track origin
    d:  3-vector, track direction
    bounds: (xmin, xmax, ymin, ymax, zmin, zmax)
            Two ranges must have finite size, one must collapse to zero.
    """

    #P0 = np.array(P0, float)
    #d  = np.array(d, float)

    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    
    # Determine which axis is collapsed
    flat_x = xmin == xmax
    flat_y = ymin == ymax
    flat_z = zmin == zmax

    # Intersection with the plane containing the panel
    t = None
    if flat_x:
        # panel lies on plane x = xmin
        if abs(d[0]) > 1e-12:
            t = (xmin - p0[0]) / d[0]
            I = p0 + t * d
            if ymin <= I[1] <= ymax and zmin <= I[2] <= zmax:
                return 0.0, I, I
    elif flat_y:
        if abs(d[1]) > 1e-12:
            t = (ymin - p0[1]) / d[1]
            I = p0 + t * d
            if xmin <= I[0] <= xmax and zmin <= I[2] <= zmax:
                return 0.0, I, I
    elif flat_z:
        if abs(d[2]) > 1e-12:
            t = (zmin - p0[2]) / d[2]
            I = p0 + t * d
            if xmin <= I[0] <= xmax and ymin <= I[1] <= ymax:
                return 0.0, I, I

    # No intersection --> closest distance to rectangle
    # Compute the closest point on the rectangle to the infinite line.
    # Strategy: find the point Q on the panel rectangle that minimizes distance to line.
    
    # First clamp p0 to the rectangle projection
    Q = np.array([p0[0], p0[1], p0[2]])
    
    # Clamp depending on collapsed axis
    if flat_x:
        Q[0] = xmin
        Q[1] = np.clip(Q[1], ymin, ymax)
        Q[2] = np.clip(Q[2], zmin, zmax)
    elif flat_y:
        Q[1] = ymin
        Q[0] = np.clip(Q[0], xmin, xmax)
        Q[2] = np.clip(Q[2], zmin, zmax)
    elif flat_z:
        Q[2] = zmin
        Q[0] = np.clip(Q[0], xmin, xmax)
        Q[1] = np.clip(Q[1], ymin, ymax)

    # closest point on line to Q
    t_line = np.dot(Q - p0, d) / np.dot(d, d)
    P_line = p0 + t_line * d

    dist = np.linalg.norm(P_line - Q)

    return dist, P_line, Q

    
def prev_dist_trk_to_pds(trk, pds_chan):
    a0 = np.array([trk.ini_x, trk.ini_y, trk.ini_z+trk.z0_corr])
    a1 = np.array([trk.end_x, trk.end_y, trk.end_z+trk.z0_corr])
    
    adir = a1-a0
    if(cf.pds_modules_type[pds_chan] == "Cathode"):
        a1_extrap = extrapolate_trk_to_z(a0, a1, cf.pds_z_centers[pds_chan])
        
    all_b0, all_b1 = xarapucas_siPM_strips(pds_chan)

    res = []
    for idx, (b0, b1) in enumerate(zip(all_b0, all_b1)):
            
        c, p, dist = closest_distance_between_lines(a0,a1,np.asarray(b0),np.asarray(b1))
        c = c.tolist() #closest point on the track
        p = p.tolist() #closest point on the xarapuca side
        res.append((dist, idx+10*(pds_chan+1), p, above_xarapuca(pds_chan, c),c))

    sort_res = sorted(res, key=lambda tup: tup[0])
                
    return sort_res[:2]


def closest_distance_between_lines(a0,a1,b0,b1):

    ''' Given two lines defined by numpy.array pairs (a0,a1,b0,b1)
        Return the closest points on each segment and their distance
        from : https://stackoverflow.com/questions/2824478/shortest-distance-between-two-line-segments
    '''


    # Calculate denomitator
    A = a1 - a0
    B = b1 - b0
    magA = np.linalg.norm(A)
    magB = np.linalg.norm(B)
    
    _A = A / magA
    _B = B / magB
    
    cross = np.cross(_A, _B);
    denom = np.linalg.norm(cross)**2
    
    
    # If lines are parallel (denom=0) test if lines overlap.
    # If they don't overlap then there is a closest point solution.
    # If they do overlap, there are infinite closest positions, but there is a closest distance
    if not denom:
        d0 = np.dot(_A,(b0-a0))
        
        d1 = np.dot(_A,(b1-a0))
            
        # Is segment B before A?
        if d0 <= 0 >= d1:        
            if np.absolute(d0) < np.absolute(d1):
                return a0,b0,np.linalg.norm(a0-b0)
            return a0,b1,np.linalg.norm(a0-b1)
                
                
        # Is segment B after A?
        elif d0 >= magA <= d1:            
            if np.absolute(d0) < np.absolute(d1):
                return a1,b0,np.linalg.norm(a1-b0)
            return a1,b1,np.linalg.norm(a1-b1)
                
                
        # Segments overlap, return distance between parallel segments
        # NB : to be improved
        return np.asarray([-9999, -9999, -9999]),B/2.,np.linalg.norm(((d0*_A)+a0)-b0)
            
    # Lines criss-cross: Calculate the projected closest points
    t = (b0 - a0);
    detA = np.linalg.det([t, _B, cross])
    detB = np.linalg.det([t, _A, cross])

    t0 = detA/denom;
    t1 = detB/denom;

    pA = a0 + (_A * t0) # Projected closest point on segment A
    pB = b0 + (_B * t1) # Projected closest point on segment B


    # Projections
    if t0 < 0:
        pA = a0
    elif t0 > magA:
        pA = a1
        
    if  t1 < 0:
        pB = b0
    elif t1 > magB:
        pB = b1
            
    # Clamp projection A
        if (t0 < 0) or ( t0 > magA):
            dot = np.dot(_B,(pA-b0))
            if dot < 0:
                dot = 0
            elif dot > magB:
                dot = magB
            pB = b0 + (_B * dot)
    
        # Clamp projection B
        if ( t1 < 0) or ( t1 > magB):
            dot = np.dot(_A,(pB-a0))
            if  dot < 0:
                dot = 0
            elif dot > magA:
                dot = magA
            pA = a0 + (_A * dot)

    
    return pA,pB,np.linalg.norm(pA-pB)


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
 
