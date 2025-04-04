import config as cf
import data_containers as dc
import numpy as np
from copy import deepcopy
import track_2d as trk2d
import track_3d as trk3d
from operator import itemgetter
import scipy.sparse as csp
from collections import Counter
import time as time
from itertools import tee

import numba as nb

def compare_slope(ls, ls_e, rs, rs_e):
    if(ls * rs < 0.):
            return 9999. #False

    """ if slope error is too low, re-assign it to 5 percent """
    if(ls_e == 0 or np.fabs(ls_e/ls) < 0.05):
        ls_e = np.fabs(ls*0.05)
        
    if(rs_e == 0 or np.fabs(rs_e/rs) < 0.05):
        rs_e = np.fabs(rs*0.05)

    return np.fabs( (ls - rs) / (ls_e + rs_e))

    

def track_in_module(t, modules):
    if(t.module_ini in modules or t.module_end in modules):
        return True
    else:
        return False
    

def test_stitching_2D_and_merge(tracks, debug=False):

    if(len(tracks)==1):
        return True
    
    """ from track3D builder, test if these tracks should be merged """
    align_thr = dc.reco['stitching_2d']['from_3d']['align_thr']
    dma_thr = dc.reco['stitching_2d']['from_3d']['dma_thr']
    dist_thr = dc.reco['stitching_2d']['from_3d']['dist_thr']
    
    
    if(debug):
        print('-------')
        [t.mini_dump() for t in tracks]
        
    ntrks = len(tracks)

    
    sparse = np.zeros((ntrks, ntrks))
    trk_ID_shift = dc.n_tot_trk2d 

    view = tracks[0].view

    """ in principle, try to merge tracks in same unwrapping situations """
    module = tracks[0].module_ini
    if(dc.evt_list[-1].det == 'pdhd' and view < 2):
        unwrappers = [(0,0), (0, cf.unwrappers[module][1]), (cf.unwrappers[module][0], 0)]
    else:
        unwrappers = [(0,0)]

        
    n = 0
    for ti in tracks[:-1]:
        stitchable = [tracks_compatibility(ti, tt, unwrappers, align_thr, dma_thr, dist_thr, False) for tt in tracks[n+1:]]
             
        for k in np.where(stitchable)[0]:
            sparse[n, n+k+1] = 1
        n = n+1

    graph = csp.csr_matrix(sparse)
    n_components, labels = csp.csgraph.connected_components(csgraph=graph, directed=False, return_labels=True)

    count = Counter(labels)
    
    n_merge = 0
    for lab, nelem in count.items():     
        if(nelem == 1): continue            
        else:

            tmerge = [it for it, l in zip(tracks, labels) if l == lab]
            
            #print('!!!!!!!!!!!!!!!! lets merge tracks ', [it.trackID for it in tmerge])

            merge_2D(tmerge, align_thr, dma_thr, dist_thr)
            
            n_merge += 1


    #if(n_merge>0):
        #reset_track2D_list()



        
        
def stitch2D_tracks_in_module():
    
    align_thr = dc.reco['stitching_2d']['in_module']['align_thr']
    dma_thr = dc.reco['stitching_2d']['in_module']['dma_thr']
    dist_thr = dc.reco['stitching_2d']['in_module']['dist_thr']
    
    """
    align_thr = dc.reco['stitching_2d']['from_3d']['align_thr']
    dma_thr = dc.reco['stitching_2d']['from_3d']['dma_thr']
    dist_thr = dc.reco['stitching_2d']['from_3d']['dist_thr']
    """
    
    ntrks = sum(dc.evt_list[-1].n_tracks2D)
    sparse = np.zeros((ntrks, ntrks))
    trk_ID_shift = dc.n_tot_trk2d


    
    for iv in range(cf.n_view):
        tracks = [t for t in dc.tracks2D_list if t.module_ini == cf.imod and t.view == iv and t.chi2_fwd < 9999. and t.chi2_bkwd < 9999.]

        if(dc.evt_list[-1].det == 'pdhd' and iv < 2):
            unwrappers = [(0,0), (0, cf.unwrappers[cf.imod][1]), (cf.unwrappers[cf.imod][0], 0)]
        else:
            unwrappers = [(0,0)]
        

        n = 0
        for ti in tracks[:-1]:
            stitchable = [tracks_compatibility(ti, tt, unwrappers, align_thr, dma_thr, dist_thr) for tt in tracks[n+1:]]
             
            
            for k in np.where(stitchable)[0]:
                sparse[ti.trackID-trk_ID_shift, tracks[k+n+1].trackID-trk_ID_shift] = 1
            n = n+1

    graph = csp.csr_matrix(sparse)
    n_components, labels = csp.csgraph.connected_components(csgraph=graph, directed=False, return_labels=True)

    count = Counter(labels)

    n_merge = 0
    for lab, nelem in count.items():     
        if(nelem == 1): continue
        else:
            tmerge = [it for it, l in zip(dc.tracks2D_list, labels) if l == lab]
            
            #print('--->> for label ', lab, ' --> ', len(tmerge), ' tracks to go', nelem)
            #print('::', [x.trackID for x in tmerge])
            
            merge_2D(tmerge, align_thr, dma_thr, dist_thr)
            n_merge += 1


    if(n_merge>0):
        reset_track2D_list()
    #print('--->  took ', time.time()-tst)
        
def reset_track2D_list():
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

    


def merge_2D(trks,  align_thr, dma_thr, dist_thr):


    unwrappers = [(0,0)]
    if(dc.evt_list[-1].det == 'pdhd' and trks[0].view < 2):
        unwrappers = [(0,0), (0, cf.unwrappers[cf.imod][1]), (cf.unwrappers[cf.imod][0], 0)]

    trks = sorted(trks, key=lambda k: -1.*k.path[0][1])



    ta = trks[0]
    merge_ID = ta.trackID
    


    for ii, ta in enumerate(trks[:-1]):
        if(ta.trackID <0):
            continue
        merge_ID = ta.trackID

        
        for tb in trks[ii+1:]:
            if(tb.trackID < 0):
                continue

            if(tracks_compatibility(ta, tb, unwrappers, align_thr, dma_thr, dist_thr, False)==False):

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
    

        
def stitch_tracks(modules):
    i = 0

    while(i < len(dc.tracks2D_list)):
        ti = dc.tracks2D_list[i]
        if(track_in_module(ti, modules) == False):
            i += 1
            continue
        if(ti.chi2_fwd == 9999. and ti.chi2_bkwd == 9999.):
            i += 1
            continue
        j = 0
        #join = []
        while( j < len(dc.tracks2D_list) ):
            if(i==j):
                j += 1
                if(j >= len(dc.tracks2D_list) ):
                    break

            tj = dc.tracks2D_list[j]
            if(track_in_module(tj, modules) == False):
                j += 1
                continue

            if(tj.chi2_fwd == 9999. and tj.chi2_bkwd == 9999.):
                j += 1
                continue
            
            ok = check_and_merge_tracks([ti,tj], 15)
            if(ok):
                i=0
                break
            else:
                j += 1
        i = i+1
            


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
    #scale = np.linalg.norm([a2 - a1, b2 - b1], axis=-1)
    dists = np.array([[dist(a1, b1, b2), dist(b1, a1, a2)],
                      [dist(a2, b1, b2), dist(b2, a1, a2)]])
        
    print(dists)
    print('close : ', np.all(dists< thr))
    if(np.all(dists< thr)):        
        return True
    return False


def one_close(a1, a2, b1, b2, thr):
    ##scale = np.linalg.norm([a2 - a1, b2 - b1], axis=-1)
    #dists = np.array([[dist(a1, b1, b2), dist(b1, a1, a2)],
    #                  [dist(a2, b1, b2), dist(b2, a1, a2)]])
        
    #if(np.any(np.all(dists< thr, axis=1))):
        #return True

    """ test if a points are all close to b points """
    dists = np.array([dist(a1, b1, b2), dist(a2, b1, b2)])
    if(np.all(dists< thr)):
        return True
    return False

    #return False


def is_all_close(a1, a2, b1, b2, thr):
    #scale = np.linalg.norm([a2 - a1, b2 - b1], axis=-1)
    dists = np.array([[dist(a1, b1, b2), dist(b1, a1, a2)],
                      [dist(a2, b1, b2), dist(b2, a1, a2)]])
        
    if(np.all(dists< thr)):
        return True
    return False
    
    
def segment_collinear(a1, a2, b1, b2, dot_thr, prox_thr):
    scale = np.linalg.norm([a2 - a1, b2 - b1], axis=-1)
    dot = dot_vector(a2-a1, b2-b1)
    dists = np.array([[dist(a1, b1, b2), dist(b1, a1, a2)],
                      [dist(a2, b1, b2), dist(b2, a1, a2)]])

    
    is_para, is_close = False, False
    if(dot > dot_thr):
        is_para = True
        
    if(np.all(dists<prox_thr*scale)):
        is_close = True
    
    return is_para, is_close




def in_between(a,b,c):
    return np.all([a[i]<c[i]<b[i] or a[i]>c[i]>b[i]  for i in range(2) ])




def tracks_compatibility(ta, tb, unwrappers, align_thr, dma_thr, dist_thr, debug=False):
    if(ta.label3D != tb.label3D):
        return  False

    
    a1 = np.array(ta.path[0])
    a2 = np.array(ta.path[-1])
    
    b1 = np.array(tb.path[0])
    b2 = np.array(tb.path[-1])


    if(debug):
        is_aligned_debug(a1, a2, b1, b2, align_thr)
        is_close_debug(a1, a2, b1, b2, dma_thr )
        print([[np.linalg.norm(a1+(i,0) - b2-(j,0)), np.linalg.norm(b1+(j,0) - a2-(i,0))] for i,j in unwrappers], '-->', [[np.linalg.norm(a1+(i,0) - b2-(j,0))< dist_thr, np.linalg.norm(b1+(j,0) - a2-(i,0))< dist_thr] for i,j in unwrappers])
        
        print('-----')


        
    if(is_aligned(a1, a2, b1, b2, align_thr) == False):        
        return False

    if(in_between(a1, a2, b1) and in_between(b1, b2, a2)):            
        return is_all_close(a1, a2, b1, b2, dma_thr )

    elif(in_between(a1, a2, b1) or in_between(b1, b2, a2)):
        if(in_between(a1, a2, b1) and in_between(a1, a2, b2)):
            return one_close(b1, b2, a1, a2, dma_thr)
        elif(in_between(b1, b2, a1) and in_between(b1, b2, a2)):
            return one_close(a1, a2, b1, b2, dma_thr)
        else:
            return False
        
    else:
        return np.any([[np.linalg.norm(a1+(i,0) - b2-(j,0))< dist_thr, np.linalg.norm(b1+(j,0) - a2-(i,0))< dist_thr] for i,j in unwrappers])

    
    return False



def track3D_module_bounds(t):

    tol = 5.
    if(dc.evt_list[-1].det != 'pdhd'):
        print('In track3D_module_bounds(), Please check the geometry!')

        
    for y_pts, mod in zip([t.ini_y, t.end_y], [t.module_ini, t.module_end]):

        ylow, yhigh = cf.y_boundaries[mod][0], cf.y_boundaries[mod][1]
        if(np.fabs(y_pts-ylow) < tol or np.fabs(y_pts-yhigh) < tol):
            #t.dump()
            return True
    return False
    

def test3D_distance_and_slopes(ta, tb, d_thr, theta_thr, phi_thr):

    #print('testing ', ta.ID_3D, ' with ', tb.ID_3D)

    if(ta.ini_x < tb.ini_x):
        ta, tb = tb, ta
        

    a1 = np.asarray([ta.ini_x, ta.ini_y, ta.ini_z])
    a2 = np.asarray([ta.end_x, ta.end_y, ta.end_z])
    
    b1 = np.asarray([tb.ini_x, tb.ini_y, tb.ini_z])
    b2 = np.asarray([tb.end_x, tb.end_y, tb.end_z])


    ta_bounds = np.asarray([[ta.ini_x, ta.ini_y, ta.ini_z],  [ta.end_x, ta.end_y, ta.end_z]])  
    tb_bounds = np.asarray([[tb.ini_x, tb.ini_y, tb.ini_z],  [tb.end_x, tb.end_y, tb.end_z]])  
    lengths = np.asarray([[np.linalg.norm(b-a) for b in tb_bounds] for a in ta_bounds])
    

    #print(lengths)
    if(np.any(lengths < d_thr)):
        #print('\n-----\n close by ! ', ta.ID_3D, tb.ID_3D)
        #print('test alignement of code : ', is_aligned_debug(a1, a2, b1, b2, 0.98))

        """
        ta.dump()
        print("with")
        tb.dump()
        """
        
        idx = np.unravel_index(np.argmin(lengths, axis=None), lengths.shape)
        #print('index of close point: ', idx)

        
        ta_theta = ta.ini_theta if idx[0]==0 else ta.end_theta
        ta_phi   = ta.ini_phi   if idx[0]==0 else ta.end_phi

        tb_theta = tb.ini_theta if idx[1]==0 else tb.end_theta
        tb_phi   = tb.ini_phi   if idx[1]==0 else tb.end_phi

        #print('angles ', ta_theta, ta_phi, ' and ', tb_theta, tb_phi)
        if(np.fabs(ta_theta-tb_theta) < theta_thr and np.fabs(ta_phi-tb_phi) < phi_thr):
            #print(' OMG let merge')
            #llll
            return True
    return False


def merge_3D(trks, is_module_crosser=False):
    dx_tol= dc.reco['track_3d']['dx_tol']
    dy_tol= dc.reco['track_3d']['dy_tol']
    dz_tol = dc.reco['track_3d']['dz_tol']


    if(len(trks)>2):
        print('More than two 3D tracks asked to be merged !!!!!')
        [t.dump() for t in trks]
        return
    
    ta, tb =  trks[0], trks[1]
    
    if(ta.ini_x < tb.ini_x):
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
    if(isok == False):
        print('whaaaaaaat????')

    trk3d.correct_timing(ta, dx_tol, dy_tol, dz_tol)


    """
    ta.dump()
    print(' and now ')
    tb.dump()
    """
    
    return ta

def stitch_across_modules(modules):
    dist_thr = 5. #cm
    theta_thr = 5. #degrees
    phi_thr = 5. #degrees

    n_trks_tot = dc.evt_list[-1].n_tracks3D
    sparse = np.zeros((n_trks_tot, n_trks_tot))
    trk_ID_shift = dc.n_tot_trk3d

    trks_bound = [t for t in dc.tracks3D_list if track_in_module(t, modules) and track3D_module_bounds(t)]
    n_trks_bound = len(trks_bound)

    #print('---> in modules ', modules, ' :: ', n_trks_tot, ' tracks, and ', n_trks_bound, ' at boundaries')
    #[t.dump() for t in trks]
    

    if(n_trks_bound <2):
        return 
    
    n=0
    for ti in trks_bound[:-1]:        
        stitchable = [test3D_distance_and_slopes(ti, tt, dist_thr, theta_thr, phi_thr) for tt in trks_bound[n+1:]]


        for k in np.where(stitchable)[0]:
            sparse[ti.ID_3D-trk_ID_shift, trks_bound[k+n+1].ID_3D-trk_ID_shift] = 1

        n = n+1


    graph = csp.csr_matrix(sparse)
    n_components, labels = csp.csgraph.connected_components(csgraph=graph, directed=False, return_labels=True)

    count = Counter(labels)

    n_merge = 0
    for lab, nelem in count.items():     
        if(nelem == 1): continue
        else:
            tmerge = [it for it, l in zip(dc.tracks3D_list, labels) if l == lab]
            
            #print('--->> for label ', lab, ' --> ', len(tmerge), ' tracks to go', nelem)
            #print('::', [x.trackID for x in tmerge])
            
            merge_3D(tmerge, is_module_crosser=True)

            n_merge += 1

            

    if(n_merge>0):
        reset_track3D_list()
    #print('merged ', n_merge, ' 3D tracks together !!! ')
    #print('--->  took ', time.time()-tst)

def stitch_across_cathode(modules):

    n_trks_tot = dc.evt_list[-1].n_tracks3D
    trk_ID_shift = dc.n_tot_trk3d

    trks_bound = [t for t in dc.tracks3D_list if track_in_module(t, modules) and track3D_module_bounds(t)]
    n_trks_bound = len(trks_bound)

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
