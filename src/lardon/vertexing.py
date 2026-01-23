import lardon.config as cf
import lardon.data_containers as dc
import lardon.lar_param as lar

from rtree import index
import numpy as np


''' 
once the 3D tracks are found, try to assemble tracks together
searches for 
- michel
- ghosts (pdvd case)
'''

def min_distance_endpoints(seg1, seg2):
    """
    Compute the minimum distance between the endpoints of two segments
    in the yz-plane.

    seg1: ((y1, z1), (y2, z2))
    seg2: ((y3, z3), (y4, z4))
    """
    (y1, z1), (y2, z2) = seg1
    (y3, z3), (y4, z4) = seg2

    def dist(a, b):
        return np.hypot(a[0] - b[0], a[1] - b[1])

        lengths = np.asarray([[np.linalg.norm(e-i) for e in ends] for i in inis])
        imaxs = np.unravel_index(np.nanargmax(lengths, axis=None), lengths.shape)
        imins = np.unravel_index(np.nanargmin(lengths, axis=None), lengths.shape)

    
    dists = np.asarray([[dist((y1, z1), (y3, z3)), dist((y1, z1), (y4, z4))],
                        [dist((y2, z2), (y3, z3)), dist((y2, z2), (y4, z4))]])
    idx = np.unravel_index(np.argmin(dists, axis=None), dists.shape)
    #print(dists, np.min(dists), idx)
    return np.min(dists), idx

def vertexing():
    d_min = 10.

    trk3D_ID_shift = dc.n_tot_trk3d
    trk2D_ID_shift = dc.n_tot_trk2d
    
    drift_dir = cf.drift_direction[cf.imod]
    idx_ghost = 1 if drift_dir < 0 else 0
    idx_michel = 1

    t3d = [t for t in dc.tracks3D_list if t.module_ini == cf.imod]
    t2d_coll = [t for t in dc.tracks2D_list if t.module_ini == cf.imod and t.view==2]

    print('nb of 3D tracks : ', len(t3d), 'nb of 2D tracks on collection view ', len(t2d_coll))
    for trk in t3d:
        match_coll = trk.match_ID[cf.imod][2]
        trk_endpts = ((trk.ini_y, trk.ini_z), (trk.end_y, trk.end_z))

        for tc in t2d_coll:
            if(match_coll == tc.trackID or tc.match_3D >=0):
                continue
            coll_endpts = ((tc.path[0][0], tc.path[0][1]), (tc.path[-1][0], tc.path[-1][1]))
            d, idx = min_distance_endpoints(trk_endpts, coll_endpts)
            if(d < d_min):
                if(idx[0]==idx[1]==idx_ghost):
                    print('\ncould be ghost! ', d, idx)
                    trk.dump()
                    print('with')
                    tc.mini_dump()
                    trk_2D = dc.tracks2D_list[match_coll-trk2D_ID_shift]
                    trk_2D.mini_dump()
                    qt = trk_2D.tot_charge
                    qg = tc.tot_charge
                    if(qg >= qt):
                        continue
                    trk_2D_slope = trk_2D.end_slope if drift_dir < 0 else trk_2D.ini_slope
                    tc_slope = tc.end_slope if drift_dir < 0 else tc.ini_slope
                    if(trk_2D_slope*tc_slope >=0):
                        continue
                    print('THIS IS VERY PROBABLY A GHOST')
                    
