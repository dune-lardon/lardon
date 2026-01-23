import lardon.config as cf
import lardon.data_containers as dc
import lardon.lar_param as lar

import lardon.track_2d as trk2d
import lardon.track_3d as trk3d

import numpy as np
import numba as nb

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from collections import Counter

import time as time


def extrapolate_2D(gap):
    return 

def extrapolate_3D(gap):
    return

def is_track_in_module(t, modules):
    """ test if the track is in the given module(s) """
    if(t.module_ini in modules or t.module_end in modules):
        return True
    else:
        return False
    


def stitch2D_test_and_merge(tracks, align_thr, dma_thr, dist_thr, unwrappers, debug=False):
    ntrks = len(tracks)    
    sparse = np.zeros((ntrks, ntrks))
    trk_ID_shift = dc.n_tot_trk2d 

    n=0
    for ti in tracks[:-1]:
        stitchable = [tracks2D_compatibility(ti, tt, unwrappers, align_thr, dma_thr, dist_thr, False) for tt in tracks[n+1:]]
             
        for k in np.where(stitchable)[0]:
            sparse[n, n+k+1] = 1
        n = n+1

    graph = csr_matrix(sparse)
    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)

    count = Counter(labels)
    
    n_merge = 0
    for lab, nelem in count.items():     
        if(nelem == 1): continue            
        else:

            tmerge = [it for it, l in zip(tracks, labels) if l == lab]
            
            if(debug):
                print('!!!!!!!!!!!!!!!! lets merge tracks ', [it.trackID for it in tmerge])
            merge_2D(tmerge, align_thr, dma_thr, dist_thr)
            n_merge += 1
    return n_merge


def stitch2D_from_3Dbuilder(tracks, debug=False):
    """ from track3D builder we have hint that these tracks may be stitchable, test if these tracks should be merged with looser constraints """
    
    if(len(tracks)==1):
        return
    

    align_thr = dc.reco['stitching_2d']['from_3d']['align_thr']
    dma_thr = dc.reco['stitching_2d']['from_3d']['dma_thr']
    dist_thr = dc.reco['stitching_2d']['from_3d']['dist_thr']
    
    
    if(debug):
        print('-------\n try to merge')
        [t.mini_dump() for t in tracks]
        

    view = tracks[0].view

    """ in principle, try to merge tracks in same unwrapping situations """
    module = tracks[0].module_ini
    if(dc.evt_list[-1].det == 'pdhd' and view < 2):
        unwrappers = [(0,0), (0, cf.unwrappers[module][1]), (cf.unwrappers[module][0], 0)]
    else:
        unwrappers = [(0,0)]

    _ = stitch2D_test_and_merge(tracks, align_thr, dma_thr, dist_thr, unwrappers, debug)
        
        
def stitch2D_in_module(modules = [cf.imod]):
    
    
    align_thr = dc.reco['stitching_2d']['in_module']['align_thr']
    dma_thr = dc.reco['stitching_2d']['in_module']['dma_thr']
    dist_thr = dc.reco['stitching_2d']['in_module']['dist_thr']
    
    debug=False

    n_merge = 0
    for iv in range(cf.n_view):
        tracks = [t for t in dc.tracks2D_list if t.module_ini in modules and t.view == iv and t.chi2_fwd < 9999. and t.chi2_bkwd < 9999.]
        
        if(dc.evt_list[-1].det == 'pdhd' and iv < 2):
            unwrappers = [(0,0), (0, cf.unwrappers[cf.imod][1]), (cf.unwrappers[cf.imod][0], 0)]
        else:
            unwrappers = [(0,0)]
        
        n_merge += stitch2D_test_and_merge(tracks, align_thr, dma_thr, dist_thr, unwrappers, debug)

    if(n_merge>0):
        reset_track2D_list()
    
    
        
def reset_track2D_list():
    """ re-assign ID of tracks since merging occured """
    idx = dc.n_tot_trk2d

    
    new_trk_list = []
    n_per_view = [0 for x in range(cf.n_view)]


    for t in dc.tracks2D_list:
        if(t.trackID == -1):
            continue
        else:
            if(t.trackID == idx):
                new_trk_list.append(t)
                n_per_view[t.view] += 1
                idx += 1
            else:
                t.set_match_hits_2D(idx)
                t.set_ID(idx)
                new_trk_list.append(t)
                n_per_view[t.view] += 1
                idx += 1
                
    dc.tracks2D_list.clear()
    dc.tracks2D_list = new_trk_list

    for iv in range(cf.n_view):
        dc.evt_list[-1].n_tracks2D[iv] = n_per_view[iv]

    
def reset_track2D_and_update_track3D_lists():
    idx_2D = dc.n_tot_trk2d

    n_trk2d = len(dc.tracks2D_list)
    
    old_trk_ID_list = [t.trackID for t in dc.tracks2D_list]
    n_ok = sum([1 if x>=0 else 0 for x in old_trk_ID_list])

    if(n_ok == n_trk2d):
        return

    prev_id = idx_2D
    new_trk_ID_list = []
    for x in range(n_trk2d):
        old = old_trk_ID_list[x]
        if(old<0):
            new_trk_ID_list.append(-1)
        else:
            new_trk_ID_list.append(prev_id)
            prev_id += 1
            


    reset_track2D_list()

    for t in dc.tracks3D_list:
        for imod in range(cf.n_module):
            for iv in range(cf.n_view):
                old_ID = t.match_ID[imod][iv]
                if(old_ID < 0):
                    continue
                else:
                    old_ID -= idx_2D
                
                t.match_ID[imod][iv] = new_trk_ID_list[old_ID]

    


def merge_2D(trks,  align_thr, dma_thr, dist_thr, debug=False):


    unwrappers = [(0,0)]
    if(dc.evt_list[-1].det == 'pdhd' and trks[0].view < 2):
        unwrappers = [(0,0), (0, cf.unwrappers[cf.imod][1]), (cf.unwrappers[cf.imod][0], 0)]

    trks = sorted(trks, key=lambda k: -1.*k.path[0][1])


    for ii, ta in enumerate(trks[:-1]):
        if(ta.trackID <0):
            continue
        merge_ID = ta.trackID

        for tb in trks[ii+1:]:
            if(tb.trackID < 0):
                continue


            """ re-check the compatibility between tracks (in case some merging) """
            if(tracks2D_compatibility(ta, tb, unwrappers, align_thr, dma_thr, dist_thr, False)==False):

                if(debug):
                    print(ta.trackID, ' with ', tb.trackID,' IS NOPE')
                continue
            


            a1, a2, b1, b2 = np.array(ta.path[0]), np.array(ta.path[-1]), np.array(tb.path[0]), np.array(tb.path[-1])
        
            if(in_between(a1, a2, b1) or in_between(b1, b2, a2)):
                ta.merge(tb)
                tb.set_ID(-1)
            else:

                utests = [min(np.linalg.norm(a1+(i,0) - b2-(j,0)),  np.linalg.norm(b1+(j,0) - a2-(i,0))) for i,j in unwrappers]
                unw = unwrappers[np.argmin(utests)]
            
                i,j = unw
            
                if(i != 0):
                    ta.shift_x_coordinates(i)
                if(j != 0):
                    tb.shift_x_coordinates(j)                
                ta.merge(tb)
                tb.set_ID(-1)

        trk2d.refilter_and_find_drays(merge_ID)
        
           

@nb.jit(nopython=True)
def dist(p, p1, p2):
    s = p2 - p1
    q = p1 + (np.dot(p - p1, s) / np.dot(s, s)) * s
    return np.linalg.norm(p - q)



@nb.jit(nopython=True)
def dot_vector(a,b):
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    return abs(a.dot(b))


@nb.jit(nopython=True)
def is_aligned(a, b, c, d, thr):
    li = b-a
    lj = d-c
    li /= np.linalg.norm(li)
    lj /= np.linalg.norm(lj)
    return abs(li.dot(lj)) > thr

def is_aligned_debug(a, b, c, d, thr):
    li = b-a
    lj = d-c
    li /= np.linalg.norm(li)
    lj /= np.linalg.norm(lj)
    print('aligned ? ',abs(li.dot(lj)), '-> ', abs(li.dot(lj)) > thr)
    return abs(li.dot(lj)) > thr

def is_close_debug(a1, a2, b1, b2, thr):
    dists = np.array([[dist(a1, b1, b2), dist(b1, a1, a2)],
                      [dist(a2, b1, b2), dist(b2, a1, a2)]])
        
    print(dists)
    print('close : ', np.all(dists< thr))
    if(np.all(dists< thr)):        
        return True
    return False


def one_close(a1, a2, b1, b2, thr):
    """ test if one endpoint of trk a is close to trk b """
    dists = np.array([dist(a1, b1, b2), dist(a2, b1, b2)])
    if(np.all(dists< thr)):
        return True
    return False

    #return False


def is_all_close(a1, a2, b1, b2, thr):
    """ test if both enpoints of trk a are close to trk b endpoints """
    dists = np.array([[dist(a1, b1, b2), dist(b1, a1, a2)],
                      [dist(a2, b1, b2), dist(b2, a1, a2)]])
        
    if(np.all(dists< thr)):
        return True
    return False
    
    
def in_between(a,b,c):
    return np.all([a[i]<c[i]<b[i] or a[i]>c[i]>b[i]  for i in range(2) ])


def tracks2D_compatibility(ta, tb, unwrappers, align_thr, dma_thr, dist_thr, debug=False):
    """ test the compatibility between trk a and trk b """
    """ track are considered as segments using their endpoints """

    if(ta.label3D != tb.label3D):
        return  False
    
    a1 = np.array(ta.path[0])
    a2 = np.array(ta.path[-1])
    
    b1 = np.array(tb.path[0])
    b2 = np.array(tb.path[-1])


    """ test the slopes """
    if(is_aligned(a1, a2, b1, b2, align_thr) == False):        
        return False


    if(in_between(a1, a2, b1) and in_between(b1, b2, a2)):
        """ test if trk a completely overlaps with trk b (or vice versa) """
        return is_all_close(a1, a2, b1, b2, dma_thr )

    
    elif(in_between(a1, a2, b1) or in_between(b1, b2, a2)):
        """ test if trk a partially overlaps with trk b (or vice versa) """
        if(in_between(a1, a2, b1) and in_between(a1, a2, b2)):
            return one_close(b1, b2, a1, a2, dma_thr)
        elif(in_between(b1, b2, a1) and in_between(b1, b2, a2)):
            return one_close(a1, a2, b1, b2, dma_thr)
        else:
            return False
        
    else:
        """ test if trk a & b are broken because of wrapped wires """
        return np.any([[np.linalg.norm(a1+(i,0) - b2-(j,0))< dist_thr, np.linalg.norm(b1+(j,0) - a2-(i,0))< dist_thr] for i,j in unwrappers])

    
    return False


def tracks3D_volume_bounds(t, modules, tol):
    """ return false for boundary to boundary tracks """
    
    xmin = min([cf.x_boundaries[modules[0]][0], cf.x_boundaries[modules[1]][0]])
    xmax = max([cf.x_boundaries[modules[0]][1], cf.x_boundaries[modules[1]][1]])
    ymin = min([cf.y_boundaries[modules[0]][0], cf.y_boundaries[modules[1]][0]])
    ymax = max([cf.y_boundaries[modules[0]][1], cf.y_boundaries[modules[1]][1]])

    if(np.fabs(t.ini_x-xmin)<tol and np.fabs(t.end_x-xmax)<tol):
        return False
    if(np.fabs(t.ini_x-xmax)<tol and np.fabs(t.end_x-xmin)<tol):
        return False
    if(np.fabs(t.ini_y-ymin)<tol and np.fabs(t.end_y-ymax)<tol):
        return False
    if(np.fabs(t.ini_y-ymax)<tol and np.fabs(t.end_y-ymin)<tol):
        return False
    return True
    
def track3D_module_bounds(t, tol):
    """does the 3D track stopped near the boundary of the module ?"""

    #if(dc.evt_list[-1].det != 'pdhd'):
        #print('In track3D_module_bounds(), Please check the geometry!')

    if(dc.evt_list[-1].det == 'pdhd'):
        for y_pts, mod in zip([t.ini_y, t.end_y], [t.module_ini, t.module_end]):
            ylow, yhigh = cf.y_boundaries[mod][0], cf.y_boundaries[mod][1]
            if(np.fabs(y_pts-ylow) < tol or np.fabs(y_pts-yhigh) < tol):
                return True
            
    elif(dc.evt_list[-1].det == 'pdvd'):
        for x_pts, mod in zip([t.ini_x, t.end_x], [t.module_ini, t.module_end]):
            xlow, xhigh = cf.x_boundaries[mod][0], cf.x_boundaries[mod][1]
            if(np.fabs(x_pts-xlow) < tol or np.fabs(x_pts-xhigh) < tol):
                return True
    else:
        print('In track3D_module_bounds(), Please check the geometry!')        
    return False
    

def tracks3D_compatibility(ta, tb, d_thr, align_thr, debug=False):
    """ test if track a and b are not too far away and have similar angles """
    
    if( (ta.module_ini == tb.module_ini) or (ta.module_end == tb.module_end)):
        return False

    if(dc.evt_list[-1].det == 'pdhd'):
        if(ta.ini_x < tb.ini_x):
            ta, tb = tb, ta
    elif(dc.evt_list[-1].det == 'pdvd'):
        if(ta.ini_z < tb.ini_z):
            ta, tb = tb, ta
        
    ta_bounds = np.asarray([[ta.ini_x, ta.ini_y, ta.ini_z],  [ta.end_x, ta.end_y, ta.end_z]])  
    tb_bounds = np.asarray([[tb.ini_x, tb.ini_y, tb.ini_z],  [tb.end_x, tb.end_y, tb.end_z]])  
    lengths = np.asarray([[np.linalg.norm(b-a) for b in tb_bounds] for a in ta_bounds])
    

    if(np.any(lengths < d_thr)):
        if(is_aligned(ta_bounds[0], ta_bounds[1], tb_bounds[0], tb_bounds[1], align_thr)):
            if(debug):
                print(ta.ID_3D, ' with ', tb.ID_3D, ' --> ', is_aligned_debug(ta_bounds[0], ta_bounds[1], tb_bounds[0], tb_bounds[1], align_thr))
                print('lengths: ', lengths)
            return True
    return False


def merge_3D(trks, is_module_crosser=False):
    dx_tol = dc.reco['track_3d']['timing']['dx_tol']
    dy_tol = dc.reco['track_3d']['timing']['dy_tol']
    dz_tol = dc.reco['track_3d']['timing']['dz_tol']


    if(len(trks)>2):
        print('More than two 3D tracks asked to be merged !!!!!')
        [t.dump() for t in trks]
        return
    
    ta, tb =  trks[0], trks[1]

    
    if(dc.evt_list[-1].det == 'pdhd'):
        if(ta.ini_x < tb.ini_x):
            ta, tb = tb, ta
    elif(dc.evt_list[-1].det == 'pdvd'):
        if(ta.ini_z < tb.ini_z):
            ta, tb = tb, ta


            
    ta_bounds = np.asarray([[ta.ini_x, ta.ini_y, ta.ini_z],  [ta.end_x, ta.end_y, ta.end_z]])  
    tb_bounds = np.asarray([[tb.ini_x, tb.ini_y, tb.ini_z],  [tb.end_x, tb.end_y, tb.end_z]])  
    lengths = np.asarray([[np.linalg.norm(b-a) for b in tb_bounds] for a in ta_bounds])
    idx = np.unravel_index(np.argmin(lengths, axis=None), lengths.shape)
    crossing_point = []
    if(is_module_crosser):
        if(idx[0] == 0):
            a_mod, a_x, a_y, a_z, a_theta, a_phi = ta.module_ini, ta.ini_x, ta.ini_y, ta.ini_z, ta.ini_theta, ta.ini_phi
        else:
            a_mod, a_x, a_y, a_z, a_theta, a_phi = ta.module_end, ta.end_x, ta.end_y, ta.end_z, ta.end_theta, ta.end_phi
        
        crossing_point.append((a_mod, a_x, a_y, a_z, a_theta, a_phi))


        if(idx[1] == 0):
            b_mod, b_x, b_y, b_z, b_theta, b_phi = tb.module_ini, tb.ini_x, tb.ini_y, tb.ini_z, tb.ini_theta, tb.ini_phi
        else:
            b_mod, b_x, b_y, b_z, b_theta, b_phi = tb.module_end, tb.end_x, tb.end_y, tb.end_z, tb.end_theta, tb.end_phi
        
        crossing_point.append((b_mod, b_x, b_y, b_z, b_theta, b_phi))

    
        crossing_point = sorted(crossing_point, key=lambda tup: tup[0])
        #print('crossing module at ')
        #print(crossing_point)

    
    ta.merge(tb, idx)

    if(is_module_crosser):
        ta.set_module_crosser(crossing_point)
    
    isok = trk3d.finalize_3d_track(ta)

    trk3d.correct_timing(ta, dx_tol, dy_tol, dz_tol)

    return ta

def stitch3D_across_modules(modules):



    """ stitch together 3D tracks in adjacent modules """
    debug=False
    dist_thr = dc.reco['stitching_3d']['module']['dist_thr']
    align_thr = dc.reco['stitching_3d']['module']['align_thr']
    boundary_tol = dc.reco['stitching_3d']['module']['boundary_tol']


    n_trks_tot = dc.evt_list[-1].n_tracks3D
    sparse = np.zeros((n_trks_tot, n_trks_tot))
    trk_ID_shift = dc.n_tot_trk3d
    
    trks_bound = [t for t in dc.tracks3D_list if is_track_in_module(t, modules) and track3D_module_bounds(t, boundary_tol)]

    n_trks_bound = len(trks_bound)
    
    if(n_trks_bound <2):
        return 
    
    n=0
    for ti in trks_bound[:-1]:
        stitchable = [tracks3D_compatibility(ti, tt, dist_thr, align_thr, debug) for tt in trks_bound[n+1:]]


        for k in np.where(stitchable)[0]:
            sparse[ti.ID_3D-trk_ID_shift, trks_bound[k+n+1].ID_3D-trk_ID_shift] = 1

        n = n+1


    graph = csr_matrix(sparse)
    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)

    count = Counter(labels)

    n_merge = 0
    for lab, nelem in count.items():     
        if(nelem == 1): continue
        else:
            tmerge = [it for it, l in zip(dc.tracks3D_list, labels) if l == lab]
            merge_3D(tmerge, is_module_crosser=True)

            n_merge += 1            

    if(n_merge>0):
        reset_track3D_list()
    print(modules, 'merged ', n_merge, ' 3D tracks together across module !!! ')



def tracks3D_cathode_crossing_test(ta, tb, dx_thresh, dy_thresh, dz_thresh, aligned_thresh):
    debug = False

    is_horizontal = False
    if(dc.evt_list[-1].det == 'pdhd'):
        is_horizontal = True
        if(ta.ini_x < tb.ini_x):
            ta, tb = tb, ta
    elif(dc.evt_list[-1].det == 'pdvd'):
        if(ta.module_ini < tb.module_ini):
            ta, tb = tb, ta

    
    
            
    a1 = np.asarray([ta.ini_x, ta.ini_y, ta.ini_z])
    a2 = np.asarray([ta.end_x, ta.end_y, ta.end_z])
    
    b1 = np.asarray([tb.ini_x, tb.ini_y, tb.ini_z])
    b2 = np.asarray([tb.end_x, tb.end_y, tb.end_z])



    if(debug):
        dz = np.fabs(a2[2]+b1[2])
        dtrack = np.fabs(a2-b1)[:2]
        if(dz < 30 or np.all([d<t for d,t in zip(dtrack, [dx_thresh, dy_thresh])])):

            ta.dump()
            print('with')
            tb.dump()
            print('DZ = ', dz)
            print('---> delta endpoints ', dtrack)
            is_aligned_debug(a1, a2, b1, b2, aligned_thresh) 

            dtrack_ratio = np.fabs(dtrack[0]/dtrack[1])
            ang_ratio_a = np.fabs(np.tan(np.radians(ta.end_phi)))
            ang_ratio_b = np.fabs(np.tan(np.radians(tb.ini_phi)))
            print('ratio : ', dtrack_ratio, ' vs ', ang_ratio_a, ' and ', ang_ratio_b)
            
    if(is_aligned(a1, a2, b1, b2, aligned_thresh) ):            

        dz = np.fabs(a2[2]+b1[2])

        if(dz < dz_thresh):
            """ compute tracks delta x and delta y """
            dtrack = np.fabs(a2-b1)[:2]        
            if(np.all([d<t for d,t in zip(dtrack, [dx_thresh, dy_thresh])])):
                """ early / on-time cathode crossing track found """
                if(debug): print('EARLY BINGPOT!')
                return True
            else:
                """ test if the track is late """
                """ in that case, each track has an endpoints at max drift time """
                """ and the relation is true: """
                """ for PDHD: delta x/delta y = tan theta / cos phi """
                """ for PDVD: delta x/delta y = 1/ tan phi """
                vdrift = lar.drift_velocity()
                z_cathodes = np.asarray([cf.anode_z[ta.module_end] - cf.n_sample[ta.module_end]*cf.drift_direction[ta.module_end] * vdrift /cf.sampling[ta.module_end], cf.anode_z[tb.module_ini] - cf.n_sample[tb.module_ini]*cf.drift_direction[tb.module_ini] * vdrift /cf.sampling[tb.module_ini]])

                z_ends = np.asarray([ta.end_z, tb.ini_z])
                dz = np.fabs(z_cathodes - z_ends)

                if(np.all([d<dz_thresh for d in dz])):

                    if(is_horizontal):
                        dtrack_ratio = np.fabs(dtrack[1]/dtrack[0])
                        ang_ratio_a = np.fabs(np.tan(np.radians(ta.end_theta))*np.cos(np.radians(ta.end_phi)))
                        ang_ratio_b = np.fabs(np.tan(np.radians(tb.ini_theta))*np.cos(np.radians(tb.ini_phi)))
                    else:
                        dtrack_ratio = np.fabs(dtrack[0]/dtrack[1])
                        ang_ratio_a = np.fabs(np.tan(np.radians(ta.end_phi)))
                        ang_ratio_b = np.fabs(np.tan(np.radians(tb.ini_phi)))

                    if(np.allclose([ang_ratio_a, ang_ratio_b], [dtrack_ratio, dtrack_ratio], rtol=0.5)):
                        if(debug): print('LATE BINGPOT!')
                        return True
               
               
               
        return False

def set_cathode_crossing_tracks(ta, tb, dz_thresh):
    debug = False
    
    xtol= dc.reco['track_3d']['timing']['dx_tol']
    ytol= dc.reco['track_3d']['timing']['dy_tol']

    if(dc.evt_list[-1].det == 'pdhd'):
        if(ta.ini_x < tb.ini_x):
            ta, tb = tb, ta
    elif(dc.evt_list[-1].det == 'pdvd'):
        if(ta.module_ini < tb.module_ini):
            ta, tb = tb, ta

    ta.reset_anode_crosser()
    tb.reset_anode_crosser()


    
    if(debug):
        print('CATHODE CROSSERS!')
        ta.dump()
        print('with')
        tb.dump()
    
    

    a2 = np.asarray([ta.end_x, ta.end_y, ta.end_z])    
    b1 = np.asarray([tb.ini_x, tb.ini_y, tb.ini_z])


    crossing_point = [(ta.module_end, ta.end_x, ta.end_y, ta.end_z, ta.end_theta, ta.end_phi),
                      (tb.module_ini, tb.ini_x, tb.ini_y, tb.ini_z, tb.ini_theta, tb.ini_phi)]

    ta.set_cathode_crosser(crossing_point, tb.ID_3D, 1)
    tb.set_cathode_crosser(crossing_point, ta.ID_3D, 0)

    vdrift = lar.drift_velocity()
    max_drifts = np.asarray([cf.anode_z[ta.module_end] - cf.n_sample[ta.module_end]*cf.drift_direction[ta.module_end] * vdrift /cf.sampling[ta.module_end], cf.anode_z[tb.module_ini] - cf.n_sample[tb.module_ini]*cf.drift_direction[tb.module_ini] * vdrift /cf.sampling[tb.module_ini]])

    z_cathodes = [cf.anode_z[ta.module_end] - cf.drift_direction[ta.module_end]*cf.drift_length[ta.module_end], cf.anode_z[tb.module_ini] - cf.drift_direction[tb.module_ini]*cf.drift_length[tb.module_ini]]
    z_anodes = [cf.anode_z[ta.module_end], cf.anode_z[tb.module_ini]]

    drift_times = [ cf.drift_length[ta.module_end]/vdrift, cf.drift_length[tb.module_ini]/vdrift]

    inside_a_x = cf.x_boundaries[ta.module_ini][0]+xtol[ta.module_ini][0] <= ta.ini_x <= cf.x_boundaries[ta.module_ini][1]-xtol[ta.module_ini][1]
    inside_a_y = cf.y_boundaries[ta.module_ini][0]+ytol[ta.module_ini][0] <= ta.ini_y <= cf.y_boundaries[ta.module_ini][1]-ytol[ta.module_ini][1]
    through_wall_a = np.any([not inside_a_x, not inside_a_y])



    inside_b_x = cf.x_boundaries[tb.module_end][0]+xtol[tb.module_end][0] <= tb.end_x <= cf.x_boundaries[tb.module_end][1]-xtol[tb.module_end][1]
    inside_b_y = cf.y_boundaries[tb.module_end][0]+ytol[tb.module_end][0] <= tb.end_y <= cf.y_boundaries[tb.module_end][1]-ytol[tb.module_end][1]
    through_wall_b = np.any([not inside_b_x, not inside_b_y])


    z_ends = np.asarray([ta.ini_z, tb.end_z])



    borders_a = np.fabs(z_ends-z_anodes)<dz_thresh
    z_cross = np.asarray([ta.end_z, tb.ini_z])

    is_max_drift = np.fabs(z_cross-max_drifts) < dz_thresh


    """
                zref = z_cathodes[idx_o]
                tref = trk_times[idx_o] - drift_times[idx_o]
    """
    
    """ Fix the z0/t0 """
    if(np.all(is_max_drift)):
        #print('SUPER LATE CATHODE CROSSERS !')
        """ very late track """
        dx = np.fabs(a2[0]-b1[0])
        dz_a = np.fabs(-dx*np.sin(np.radians(ta.end_phi))*np.tan(np.radians(ta.end_theta)))
        dz_b = np.fabs(-dx*np.sin(np.radians(tb.ini_phi))*np.tan(np.radians(tb.ini_theta)))

        
        z0 = z_cathodes[0] - ta.end_z + cf.drift_direction[ta.module_end]*dz_a/2.
        tref = ta.end_time/cf.sampling[ta.module_end] - drift_times[0] + cf.drift_direction[ta.module_end]*dz_a/2./vdrift
        t0 = tref#z0/vdrift
        #if(t0 < 0):
        #    t0 *= -1
        ta.set_t0_z0(t0, z0)
        ta.set_timestamp(tref, tref)
        """
        print('DZ :: a ', dz_a)
        print('t0 was a: ', z0/vdrift)
        ta.dump()
        """
        z0 = z_cathodes[1] - tb.ini_z + cf.drift_direction[tb.module_ini]*dz_b/2.
        #t0 = z0/vdrift
        tref = tb.ini_time/cf.sampling[tb.module_ini] - drift_times[1] + cf.drift_direction[tb.module_ini]*dz_b/2./vdrift
        t0 = tref
        #if(t0 < 0):
        #    t0 *= -1
        tb.set_t0_z0(t0, z0)
        tb.set_timestamp(tref, tref)
        """
        print('DZ :: b ', dz_a)
        print('t0 was b: ', z0/vdrift)
        tb.dump()
        """
        #dy_data = ta.end_y - tb.ini_y
        #dy_th_a = dz_a/np.tan(np.radians(ta.end_phi))
        #dy_th_b = dz_b/np.tan(np.radians(tb.ini_phi))
        

    elif(np.any(borders_a)):        
        """ early tracks, fix t0 """
        #print('NOT SO LATE CATHODE CROSSERS -- one track through walls')

        z0 = z_cathodes[0] - a2[2]
        tref = ta.end_time/cf.sampling[ta.module_end] - drift_times[0]
        t0 = tref#z0/vdrift        
        #if(t0 > 0):
        #    t0 *= -1

        ta.set_t0_z0(t0, z0)
        ta.set_timestamp(tref, tref)
        """
        print('t0 was a: ', z0/vdrift)
        ta.dump()
        """
        
        z0 = z_cathodes[1] - b1[2]
        tref = tb.ini_time/cf.sampling[tb.module_ini] - drift_times[1]#z0/vdrift
        t0 = tref
        #if(t0 > 0):
        #    t0 *= -1
        tb.set_t0_z0(t0, z0)
        tb.set_timestamp(tref, tref)
        #print('t0 was b: ', z0/vdrift)
        #tb.dump()

    else:
        """ on time track/slightly delayed ! """
        #print(' ON TIME ? CATHODE CROSSERS')
        
        z0 = z_cathodes[0] - a2[2]
        tref = ta.end_time/cf.sampling[ta.module_end] - drift_times[0]
        t0 = tref#z0/vdrift        
        #if(t0 > 0):
        #    t0 *= -1

        ta.set_t0_z0(t0, z0)
        ta.set_timestamp(tref, tref)
        """
        print('t0 was a: ', z0/vdrift)
        ta.dump()
        """

        z0 = z_cathodes[1] - b1[2]
        tref = tb.ini_time/cf.sampling[tb.module_ini] - drift_times[1]#z0/vdrift
        t0 = tref
        #if(t0 > 0):
        #    t0 *= -1
        tb.set_t0_z0(t0, z0)
        tb.set_timestamp(tref, tref)
        """
        print('t0 was b: ', z0/vdrift)
        tb.dump()
        """
        
def stitch3D_across_cathode(modules):
    
    dx_thresh = dc.reco['stitching_3d']['cathode']['dx_thresh']
    dy_thresh = dc.reco['stitching_3d']['cathode']['dy_thresh']
    dz_thresh = dc.reco['stitching_3d']['cathode']['dz_thresh']
    align_thresh = dc.reco['stitching_3d']['cathode']['align_thresh']
    boundary_tol = dc.reco['stitching_3d']['cathode']['boundary_tol']
    
    n_trks_tot = dc.evt_list[-1].n_tracks3D
    trk_ID_shift = dc.n_tot_trk3d

    trks_bound_drift_a = [t for t in dc.tracks3D_list if is_track_in_module(t, modules[0]) and tracks3D_volume_bounds(t, modules[0], boundary_tol)]
    trks_bound_drift_b = [t for t in dc.tracks3D_list if is_track_in_module(t, modules[1]) and tracks3D_volume_bounds(t, modules[1], boundary_tol)]

    n_trks_bound = len(trks_bound_drift_a) + len(trks_bound_drift_b)
    

    sparse = np.zeros((n_trks_tot, n_trks_tot))
    trk_ID_shift = dc.n_tot_trk3d

    print(len(trks_bound_drift_a), ' vs ', len(trks_bound_drift_b))
    
    n=0
    for ti in trks_bound_drift_a:        
        stitchable = [tracks3D_cathode_crossing_test(ti, tt, dx_thresh, dy_thresh, dz_thresh, align_thresh) for tt in trks_bound_drift_b]
        n = n+1


        for k in np.where(stitchable)[0]:
            sparse[ti.ID_3D-trk_ID_shift, trks_bound_drift_b[k].ID_3D-trk_ID_shift] = 1

        n = n+1

    
    graph = csr_matrix(sparse)
    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
    
    count = Counter(labels)

    n_cross = 0
    for lab, nelem in count.items():     
        if(nelem == 1): continue
        elif(nelem == 2):
            tcross = [it for it, l in zip(dc.tracks3D_list, labels) if l == lab]
            set_cathode_crossing_tracks(*tcross, dz_thresh)
            n_cross += 1

            
        else:
            print('WHAAAAT ? too many possibility for cathode stitcher, do not do anything ')

    print('Found ', n_cross, ' cathode crossing tracks ')


    
def reset_track3D_list():
    idx = dc.n_tot_trk3d
    trk2D_ID_shift = dc.tracks2D_list[0].trackID

    
    n_prev = dc.evt_list[-1].n_tracks3D
    new_trk_list = []

    for t in dc.tracks3D_list:
        if(t.ID_3D == -1):
            continue
        else:
            if(t.ID_3D == idx):
                new_trk_list.append(t)
                idx += 1
            else:
                t.set_ID(idx)
                for imod in range(cf.n_module):
                    for iv in range(cf.n_view):
                        if(t.match_ID[imod][iv] >=0):
                            t2d = dc.tracks2D_list[t.match_ID[imod][iv]-trk2D_ID_shift]
                            t2d.set_match_hits_3D(idx)

                new_trk_list.append(t)                
                idx += 1
                
    dc.tracks3D_list.clear()
    dc.tracks3D_list = new_trk_list


    dc.evt_list[-1].n_tracks3D = len(new_trk_list)
    #print('after stitching, now have ', dc.evt_list[-1].n_tracks3D, ' 3D tracks (',n_prev,'before)')
