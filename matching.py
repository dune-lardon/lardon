import config as cf
import data_containers as dc
import lar_param as lar

import numpy as np
from rtree import index

def matching_charge_pds():
    if(len(dc.tracks3D_list)== 0 or len(dc.pds_cluster_list) == 0):
        return

    time_tol = dc.reco['pds']['matching']['time_tol'] #in mus
    
    """ create  Rtree, now filled with the found clusters """
    pties = index.Property()
    pties.dimension = 2

    rtree = index.Index(properties=pties)

    
    ''' filling the R-tree with the 3D tracks on y axis 0'''
    for t in dc.tracks3D_list:
        if(t.match_pds_cluster >= 0):
            continue
        start = t.timestamp
        idx   = t.ID_3D
        rtree.insert(idx, (start, 0, start, 0))

    
    ''' filling the R-tree with the single hits on y axis 1'''
    for sh in dc.single_hits_list:
        if(sh.match_pds_cluster >= 0):
            continue
        start = sh.timestamp
        idx   = sh.ID_SH
        rtree.insert(idx, (start, 1, start, 1))

    
    ''' filling the R-tree with the light clusters on y axis 2'''
    for c in dc.pds_cluster_list:
        if(c.match_trk3D >= 0 or c.match_single >=0 ):
            continue
 
        start = c.timestamp
        idx   = c.ID
        rtree.insert(idx, (start, 2, start, 2))



    id_trk3d_shift = dc.n_tot_trk3d
    id_cluster_shift = dc.n_tot_pds_clusters
    id_sh_shift = dc.n_tot_sh
    
    """ search for the 3D track - light clusters """
    for trk in dc.tracks3D_list:
        if(trk.match_pds_cluster >= 0):
            continue

        trk_start = trk.timestamp
        trk_idx   = trk.ID_3D



        overlaps = list(rtree.intersection((trk_start-time_tol, 2, trk_start+time_tol, 2)))
        
        
        if(len(overlaps) == 0):
            continue
        
        min_delay, min_clus = 99999, -1        

        for ov in overlaps:
            clus = dc.pds_cluster_list[ov-id_cluster_shift]

            if(clus.match_trk3D >= 0 or clus.match_single >=0 ):
                continue

            delay = clus.timestamp-trk_start

            if(np.abs(delay) < np.abs(min_delay)):
                min_delay = delay
                min_clus  = ov
                

        if(min_clus < 0):
            continue
        
        trk.match_pds_cluster = min_clus
        clus = dc.pds_cluster_list[min_clus-id_cluster_shift]
        clus.match_trk3D = trk_idx

        for pds_ch in clus.glob_chans[::2]:
            module_chan = int(pds_ch/2)
            res = dist_trk_to_pds(trk, module_chan)            
        
            clus.dist_closest_strip.extend([r[0] for r in res])
            clus.id_closest_strip.extend([r[1] for r in res])
            clus.point_impact.extend([r[2] for r in res])
            clus.point_closest_above.extend([r[3] for r in res])
            clus.point_closest.extend([r[4] for r in res])
    
    vdrift = lar.drift_velocity()
                
    """ search for the single hits - light clusters """
    for sh in dc.single_hits_list:
        if(sh.match_pds_cluster >= 0):
            continue
        
        sh_start = sh.timestamp
        sh_idx   = sh.ID_SH

        z_anode = cf.anode_z[sh.module]
        ''' maximum drift distance given the time window '''
        max_drift = cf.drift_length/(cf.drift_direction[sh.module] * vdrift)        
        overlaps = list(rtree.intersection((sh_start-time_tol-max_drift, 2, sh_start+time_tol, 2)))
        
        free_overlaps = []

        for ov in overlaps:
            clus = dc.pds_cluster_list[ov-id_cluster_shift]

            if(clus.match_trk3D >= 0 or clus.match_single >=0 ):
                continue
            free_overlaps.append(ov)
            

        best_overlap = -1
        
        if(len(free_overlaps) == 1):
            best_overlap = free_overlaps[0]
        elif(len(free_overlaps) > 1):
            best_overlap = sh_closest_cluster(sh, free_overlaps, id_cluster_shift)

        if(best_overlap >=0):
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

def dist_trk_to_pds(trk, pds_chan):

    a0 = np.array([trk.ini_x, trk.ini_y, trk.ini_z+trk.z0_corr])
    a1 = np.array([trk.end_x, trk.end_y, trk.end_z+trk.z0_corr])
    
    
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


def sh_closest_cluster(sh, cluster_idx, id_shift):
    clus_dist = []

    for idx in cluster_idx:
        clus = dc.pds_cluster_list[idx-id_shift]        
        distances = []

        for pds_ch in clus.glob_chans:
            mod = dc.chmap_pds[pds_ch].module
            d = np.sqrt(pow(sh.X-cf.pds_x_centers[mod], 2) + pow(sh.Y-cf.pds_y_centers[mod], 2))
            distances.append(d)

        clus_dist.append((idx, min(distances)))
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
 
