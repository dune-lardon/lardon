import lardon.config as cf
import lardon.data_containers as dc
import lardon.lar_param as lar

from rtree import index
import numpy as np

from operator import itemgetter
from collections import Counter
from sklearn.cluster import DBSCAN

from itertools import product
import time as time

from numba import njit

'''
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
'''

    
    
def is_inside_volume(module, x,y):

    module_xlow, module_xhigh = cf.x_boundaries[module][0], cf.x_boundaries[module][1] 

    module_ylow, module_yhigh = cf.y_boundaries[module][0], cf.y_boundaries[module][1]

    
    if(x < module_xlow or x > module_xhigh):
        return False
    if(y< module_ylow or y > module_yhigh):

        return False

    return True


def check_combination(comb):
    if(dc.evt_list[-1].det == 'pdvd'):
        """ test if hits combinations are in the same CRU """
        
        cru_cutoff = [476, 476, 584]
        tests = [h.channel < cru_cutoff[h.view] for h in comb]

        if(all(tests)==True or not any(tests) == True):
            return True
        else:
            return False
    

@njit(nopython=True, fastmath=True, cache=True)
def xy_leastsq_nb_prev(ha, hb, hc, angle_v0, angle_v1, angle_v2):

    # Convert inputs to float64 explicitly
    angle_rad = np.radians(np.asarray([angle_v0, angle_v1, angle_v2], dtype=np.float64))
    #hh_arr = np.asarray([ha, hb, hc], dtype=np.float64)
    
    # Precompute trigonometric functions
    sin_ang = np.sin(angle_rad)
    cos_ang = np.cos(angle_rad)
    
    # Build matrices directly without temporary arrays
    ang_mtx_mult = np.empty((3, 2), dtype=np.float64)
    ang_mtx_sum = np.empty((3, 2), dtype=np.float64)
    
    for i in range(3):
        ang_mtx_mult[i, 0] = sin_ang[i]
        ang_mtx_mult[i, 1] = -cos_ang[i]
        ang_mtx_sum[i, 0] = -cos_ang[i]
        ang_mtx_sum[i, 1] = -sin_ang[i]
    
    # Diagonal matrix construction
    pos = np.zeros((3, 3), dtype=np.float64)
    pos[0,0] = ha
    pos[1,1] = hb
    pos[2,2] = hc
    
    # Main computations
    A = pos @ ang_mtx_mult
    V = ang_mtx_sum
    
    # Optimized tensor operations
    T = np.empty((3, 2, 2), dtype=np.float64)
    for i in range(3):
        vi = V[i]
        for j in range(2):
            for k in range(2):
                T[i, j, k] = vi[j] * vi[k]
        T[i, 0, 0] -= 1.0
        T[i, 1, 1] -= 1.0
    
    # Sum reductions
    S = np.zeros((2, 2), dtype=np.float64)
    C = np.zeros(2, dtype=np.float64)
    for i in range(3):
        for j in range(2):
            for k in range(2):
                S[j, k] += T[i, j, k]
                C[j] += T[i, j, k] * A[i, k]
    
    # Solve and final computations
    X = np.linalg.solve(S, C)
    
    max_R = 0.0
    for i in range(3):
        U = (X[0] - A[i, 0]) * V[i, 0] + (X[1] - A[i, 1]) * V[i, 1]
        dx = X[0] - (A[i, 0] + U * V[i, 0])
        dy = X[1] - (A[i, 1] + U * V[i, 1])
        R = np.sqrt(dx*dx + dy*dy)
        if R > max_R:
            max_R = R
    
    return X, max_R

    
@njit(nopython=True, cache=True)
def xy_leastsq_nb(ha, hb, hc, angle_v0, angle_v1, angle_v2):
    # work if there is one missing view (position= NaN in that case)
    hh = np.empty(3, dtype=np.float64)
    hh[0] = ha
    hh[1] = hb
    hh[2] = hc

    # Convert angles to radians
    ang = np.radians(np.array([angle_v0, angle_v1, angle_v2], dtype=np.float64))
    sin_ang = np.sin(ang)
    cos_ang = np.cos(ang)

    # Count valid planes
    valid = np.zeros(3, dtype=np.int64)
    n_valid = 0
    for i in range(3):
        if hh[i] == hh[i]: #not np.isnan(hh[i]):
            valid[n_valid] = i
            n_valid += 1

    # Need at least 2 planes
    if n_valid < 2:
        # return invalid result
        return np.array([np.nan, np.nan]), 99999

    # Prepare matrices with max size 3
    A = np.zeros((3,2), dtype=np.float64)
    V = np.zeros((3,2), dtype=np.float64)
    used = 0

    # Build A and V only for valid planes
    for idx in range(n_valid):
        i = valid[idx]

        # Line direction matrices
        A[used,0] = hh[i] * sin_ang[i]
        A[used,1] = -hh[i] * cos_ang[i]

        V[used,0] = -cos_ang[i]
        V[used,1] = -sin_ang[i]

        used += 1

    # Build T, S, C using only "used" rows
    S = np.zeros((2,2), dtype=np.float64)
    C = np.zeros(2, dtype=np.float64)

    for i in range(used):
        v0 = V[i,0]
        v1 = V[i,1]

        # Compute T_i = v v^T − I
        # Manually unrolled for speed
        T00 = v0*v0 - 1.0
        T01 = v0*v1
        T10 = T01
        T11 = v1*v1 - 1.0

        # Accumulate S = Σ T_i
        S[0,0] += T00
        S[0,1] += T01
        S[1,0] += T10
        S[1,1] += T11

        # C += T_i * A_i
        C[0] += T00*A[i,0] + T01*A[i,1]
        C[1] += T10*A[i,0] + T11*A[i,1]

    # Solve S X = C
    X = np.linalg.solve(S, C)

    # Compute max R using only valid planes
    max_R = 0.0
    for i in range(used):
        U = (X[0] - A[i,0])*V[i,0] + (X[1] - A[i,1])*V[i,1]
        dx = X[0] - (A[i,0] + U*V[i,0])
        dy = X[1] - (A[i,1] + U*V[i,1])
        R = np.sqrt(dx*dx + dy*dy)
        if R > max_R:
            max_R = R

    return X, max_R


def xy_leastsq(hits, unwrap=[0., 0., 0.]):
    module = hits[0].module
    
    hh = [h.X+u for h,u in zip(hits, unwrap)]
        
    ang_mtx_mult = np.array([[np.sin(np.radians(cf.view_angle[module][iv])), -np.cos(np.radians(cf.view_angle[module][iv]))] for iv in range(3)])
    ang_mtx_sum  = np.array([[-np.cos(np.radians(cf.view_angle[module][iv])), -np.sin(np.radians(cf.view_angle[module][iv]))] for iv in range(3)])


    pos = np.zeros((3,3))
    np.fill_diagonal(pos, hh)
    
    A = pos @ ang_mtx_mult
    B = A + ang_mtx_sum

    V = ang_mtx_sum
      
    N, D = A.shape  # Number of points & dimensions

    # T = V .* V - I (I is identity matrix)
    T = np.einsum('ij,ik->ijk', V, V) - np.eye(D)[np.newaxis, :, :]  # V.*V-1 as N*D*D
    S = np.sum(T, axis=0)  # Sum T along N, result is D*D
    C = np.sum(np.einsum('ijk,ij->ik', T, A), axis=0)  # T*A, result is D

    # Solve the linear system S * X = C
    X = np.linalg.solve(S, C)  # Solve for X: S*X=C, in least squares sense
    
    
    if(is_inside_volume(module, X[0],X[1])==False):
        return X, 9999
    
    U = np.sum((X - A) * V, axis=1)  # dot(X-A,V) distance from A to nearest point on each line
    P = A + U[:, np.newaxis] * V  # Nearest point on each line

    R = np.sqrt(np.sum((X - P) ** 2, axis=1))  # Distance from intersection to each line

    return X, max(R)


def compute_xy(ov, h, d_thresh, unwrap, debug=False):
    nmatch = [len(x) for x in ov]

    '''
    """ ditch 2-view only matches atm """
    if(any([x == 0 for x in nmatch])):
        return -11111, [], None
    '''
    
    combinations = list(product(*ov))
    #n_comb = np.prod(nmatch)
    #print("nb of match, ", nmatch, ' nb comb ', n_comb)

    module = h.module
    angle  = cf.view_angle[module]
    result = []
    ncomp = 0

    if(debug):
        print('nb of combination: ', len(combinations), "nmatch", nmatch)
        print('results:')
        print([xy_leastsq_nb(*[h.X+u if h is not None else np.nan for h,u in zip(c,unwrap)], *angle) for c in combinations])
        
    result = [(*xy_leastsq_nb(*[h.X+u if h is not None else np.nan for h,u in zip(c,unwrap)], *angle),c) for c in combinations]
    result = [r for r in result if is_inside_volume(module, r[0][0],r[0][1])]
    
    if(len(result)==0):        
        return -1, [], None
    
    result = sorted(result, key=itemgetter(1))
    xy, d, comb = result[0]
    if(d < d_thresh):
        return d, xy, comb
    
    return -1*d, [], None

    
def build_hit_3D(ov, h, d_thresh, unwrap):
    module = h.module

    angle = [np.radians(x) for x in cf.view_angle[module]]
    nmatch = [len(x) for x in ov]

    
    """ ditch 2-view only matches atm """
    if(any([x == 0 for x in nmatch])):
        return False, 2

    """ make a list of all hit combinations possible """
    combinations = list(product(*ov))

    if(len(combinations) == 0):
        return False, 0
    
    result = []
    ncomp = 0

    result = [(*xy_leastsq(c,unwrap),c) for c in combinations if check_combination(c)==True]
    
    if(len(result)==0):
        return False, 0

    """ sort results by increase residual distance """
    result = sorted(result, key=itemgetter(1))
    xy, d, comb = result[0]
    
    if(d < d_thresh):
        h.set_3D(xy[0], xy[1], [c.ID for c in comb])
        return True, comb

    return False, None


def clustering_3D(eps,min_samp):


    n_clusters = 0


    hits = [x for x in dc.hits_list if x.has_3D==True]
    if(len(hits)==0): return


    data = [[x.x_3D,x.y_3D, x.Z] for x in hits]
    X = np.asarray(data)
    db = DBSCAN(eps=eps,min_samples=min_samp).fit(X)
    labels = db.labels_

    [h.set_cluster_3D(l) for h,l in zip(hits,labels)]
        
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    print('total nb of 3D clusters=', n_clusters)


    
def clustering_hits_per_view(y_squeez, eps, min_samp, debug):

    if(debug):
        import matplotlib.pyplot as plt
        import colorcet as cc
        fig = plt.figure()
        ax = [fig.add_subplot(131+i) for i in range(3)]
        [ax[i].sharey(ax[0]) for i in [1,2]]
        
    n_clusters, n_tot_clusters = 0,0
    ID_shift = dc.n_tot_hits_clusters

    
    for iview in range(cf.n_view):
        hits = [x for x in dc.hits_list if x.view==iview and x.has_3D==False]
        if(len(hits)==0): continue

        """ squeeze y axis instead of defining a new metric """
        data = [[x.X,x.Z*y_squeez] for x in hits]
        X = np.asarray(data)
        db = DBSCAN(eps=eps,min_samples=min_samp).fit(X)
        labels = db.labels_

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        count = Counter([l+n_tot_clusters+ID_shift for l in labels])
        
               
        
        [dc.hits_cluster_list.append(dc.hits_clusters(l+n_tot_clusters+ID_shift, iview)) for l in range(n_clusters)]
        [h.set_cluster(l+n_tot_clusters+ID_shift) for h,l in zip(hits,labels)]
        [dc.hits_cluster_list[h.cluster-ID_shift].add_hit(h.ID) for h in hits]
        
        
        n_tot_clusters += n_clusters        
        
        if(debug):
            hx = [h.X for h,l in zip(hits, labels) if l>=0]
            hy =  [h.Z for h,l in zip(hits, labels) if l>=0]
            clus = [l for l in labels if l >=0]
            ax[iview].scatter(hx, hy, s=1, c=clus, cmap=cc.cm.glasbey)            
            [ax[iview].scatter(h.X, h.Z, s=1,c='grey') for h,l in zip(hits, labels) if l==-1]

    if(debug):
        plt.show()
    dc.evt_list[-1].n_hits_clusters = n_tot_clusters
    

    
def xy_from_clusters(selection, rtree_idx, ID_shift, d_thresh, unwrap, mlp_nn=None):

    cluster = [s[0].cluster for s in selection]

    n=0
    for sel in selection:
        for h in sel:
            if(h.has_3D):
                continue
            z = h.Z
            
            overlaps = [[] for x in range(cf.n_view)]            
            for iview in range(3):
                if iview == h.view:
                    overlaps[iview].append(h)
                else:
                    intersect = list(rtree_idx.intersection((iview, cluster[iview], z, iview, cluster[iview], z)))                
                    
                    [overlaps[iview].append(dc.hits_list[k-ID_shift]) for k in intersect]
                

            isok,out = build_hit_3D(overlaps, h, d_thresh, unwrap)
            n = n+1
            
def build_3D_hits_with_cluster():
    time_spread = dc.reco['hits3d']['time_spread']
    d_tol = dc.reco['hits3d']['d_tol']
    y_squeez = dc.reco['hits3d']['y_squeez']
    eps      = dc.reco['hits3d']['eps']
    min_samp = dc.reco['hits3d']['min_samp']

    
    if(dc.evt_list[-1].det == 'pdhd'):
        unwrappers = [[0,0,0], [cf.unwrappers[cf.imod][0], 0, 0], [0, cf.unwrappers[cf.imod][1], 0], [cf.unwrappers[cf.imod][0], cf.unwrappers[cf.imod][1], 0]]
    else:
        unwrappers = [[0,0,0]]

    
    #cluster the hits in 2D    
    
    """ y_squeez parameter shrinks the y(time)-axis to avoid having to define a new metric """
    clustering_hits_per_view(y_squeez=y_squeez, eps=eps, min_samp=min_samp, debug=False)
        

    pties = index.Property()
    pties.dimension = 3

    ''' create an rtree index (4D : module, view, cluster, time )'''
    rtree_idx = index.Index(properties=pties)

    ID_shift = dc.hits_list[0].ID
    hits = [x for x in dc.hits_list if x.is_free == True and x.module == cf.imod]
    nhits = len(hits)

    """ index the free hits in a 3D r-tree: (view, cluster, Z) """
    for h in hits: 
        start = max(h.Z_start, h.Z_stop )
        stop  = min(h.Z_start, h.Z_stop )

        rtree_idx.insert(h.ID, (h.view, h.cluster, stop-time_spread, h.view, h.cluster, start+time_spread))

    
    n_seed = 0
    for h in hits:
        if(h.has_3D == True):
            continue
        n_seed += 1

        start = h.Z_start
        stop  = h.Z_stop

        z = h.Z
        
        overlaps = [[] for x in range(cf.n_view)]

        """ start with searching for intersections irregardless of cluster nb """
        for iview in range(3):
            if iview == h.view:
                overlaps[iview].append(h)
            else:
                intersect = list(rtree_idx.intersection((iview, -99999, z, iview, 99999, z)))
                
                [overlaps[iview].append(dc.hits_list[k-ID_shift]) for k in intersect]
                
        
        for u in unwrappers:
            isok, comb = build_hit_3D(overlaps, h, d_tol, u)

            if(isok):
                break


        """ eventually, once a 3D match is found, one can further match all hits in the same clusters of the 3D match """
        """ --> does not work as well/fast as I thought, need some extra work here """
        
        if(False):             
            overlaps = [[] for x in range(cf.n_view)]
            
            for iview in range(3):
                intersect = list(rtree_idx.intersection((iview, comb[iview].cluster, -99999, iview, comb[iview].cluster, 99999)))                
                [overlaps[iview].append(dc.hits_list[k-ID_shift]) for k in intersect]

            xy_from_clusters(overlaps, rtree_idx, ID_shift, d_thresh, unwrap)


            

    n3d_found = sum([1 for h in dc.hits_list if h.has_3D==True ])
    print('----> nb of 3D hits = ', n3d_found,'/', len(hits))

    
