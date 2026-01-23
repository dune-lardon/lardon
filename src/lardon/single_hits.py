import lardon.config as cf
import lardon.data_containers as dc
import lardon.lar_param as lar
import lardon.hits_3d as h3d

from rtree import index
import numpy as np


from itertools import chain
from collections import Counter
from sklearn.cluster import DBSCAN






def is_inside_volume(module, x,y):
    module_xlow, module_xhigh = cf.x_boundaries[module][0], cf.x_boundaries[module][1] 

    module_ylow, module_yhigh = cf.y_boundaries[module][0], cf.y_boundaries[module][1]

    
    if(x < module_xlow or x > module_xhigh):
        return False
    if(y< module_ylow or y > module_yhigh):    
        return False

    return True

def t(p, q, r):
    x = p-q
    return np.dot(r-q, x)/np.dot(x, x)

def dist(p, q, r):
    return np.linalg.norm(t(p, q, r)*(p-q)+q-r)


def closest_activity_3D(x0,y0,z0):
    d_min = 9999;
    for t in dc.tracks3D_list:
        start = np.array([t.ini_x, t.ini_y, t.ini_z])
        end = np.array([t.end_x, t.end_y, t.end_z])
        hit = np.array([x0, y0, z0])
        
        d = dist(start, end, hit)
        if(d < d_min):
            d_min = d

    return d_min


def closest_activity_2D(x0,y0):
    d_min = 9999;

    for t in dc.tracks3D_list:
        start = np.array([t.ini_x, t.ini_y])
        end = np.array([t.end_x, t.end_y])
        hit = np.array([x0, y0])
        
        d = dist(start, end, hit)
        if(d < d_min):
            d_min = d

    return d_min


def veto(view, hits, dist):
    
    """ make sure there is no activity around the single hit on a square of size 2*dist centered on the single hit """
    ID_shift = dc.hits_list[0].ID
    IDs = [h.ID for h in hits]
    
    coord = np.array([[h.X, h.Z] for h in hits])
    
    xmin, xmax = np.min(coord[:,0]), np.max(coord[:,0])
    ymin, ymax = np.min(coord[:,1]), np.max(coord[:,1])
    module = hits[0].module


    ''' get all the hits in this region '''
    intersect = list(dc.rtree_hit_idx.intersection((module, view, xmin-dist, ymin-dist, module, view, xmax+dist, ymax+dist)))

    ''' remove hits belonging to the single hit '''
    nearest  = [dc.hits_list[k-ID_shift] for k in intersect if k not in IDs]
    
    return len(nearest)
               

def compute_sh_properties(hits):
    charge_pos = 0.
    charge_neg = 0.
    min_t = 99999
    zero_t = 0
    max_t = 0
    max_Q = 0.
    start = 0
    stop  = 0
    view = -1

    for h in hits:
        if(np.fabs(h.charge) > max_Q):
            max_Q = np.fabs(h.charge)

            min_t = h.min_t
            zero_t = h.zero_t
            max_t = h.max_t
            start = h.start
            stop  = h.stop
            view = h.view

        charge_pos += h.charge_pos
        charge_neg += h.charge_neg
    

    return view, charge_pos, charge_neg, start, stop, max_t, zero_t, min_t

def check_times(h, ov):
    return ov
    new_ov = [[] for x in range(cf.n_view)]
    h_view = h.view
    new_ov[h_view].append(h)
    for iv in range(cf.n_view):
        for ho in ov[iv]:
            if(iv < h_view and ho.t <= h.t):
                new_ov.append(ov)
            if(iv>h_view and ho.t >= h.t):
                new_ov.append(ov)
    return new_ov


def check_cru(h, ov):
    cru_limit = [472, 472, 584]
    crus = [int(h.channel/cru_limit[h.view]) for o in ov for h in o]
    if(len(set(crus)) > 1):
        #print('we have cru situation!')
        #print(set(crus))
        
        new_ov = [[] for x in range(cf.n_view)]
        h_view = h.view
        h_cru = int(h.channel/cru_limit[h.view])
        #new_ov[h_view].append(h)
        for iv in range(cf.n_view):
            for ho in ov[iv]:
                if(int(ho.channel/cru_limit[ho.view]) == h_cru):
                    new_ov[iv].append(ho)
        return new_ov
    else:
        return ov



def check_nmatch(ov, max_per_view):
    nmatch = [len(x) for x in ov]
    is_good = True
    for x in nmatch:
        is_good = is_good and (x>0 and x <max_per_view+1)
    return is_good
    
def find_outliers(coord, d_max):
    med_x = np.median([x[2] for x in coord])
    med_y = np.median([x[3] for x in coord])
    
    dx = [np.fabs(med_x-x[2]) for x in coord]
    dy = [np.fabs(med_y-x[3]) for x in coord]
        
    out = []
    out.extend([[x[0],x[1]] for x,d,l in zip(coord,dx, dy) if d>d_max or l>d_max])
    
    out = list(chain(*out))
    c = Counter(out)
    ID_to_rm = [k for k,v in c.items() if v>=2]
    return med_x, med_y, ID_to_rm

def barycenter(coord, all_z):
    bx = np.mean([x[0] for x in coord])
    by = np.mean([x[1] for x in coord])
    bz = np.mean(all_z)

    dmax = np.max([np.sqrt(pow(bx-h[0],2)+pow(by-h[1], 2)) for h in coord])
    return bx, by, bz, dmax



def get_hit_xy(module, ha, hb):
    v_a, v_b = ha.view, hb.view
    ang_a = np.radians(cf.view_angle[module][v_a])
    ang_b = np.radians(cf.view_angle[module][v_b])
    x_a, x_b = ha.X, hb.X

    
    A = np.array([[-np.cos(ang_a), np.cos(ang_b)],
                  [-np.sin(ang_a), np.sin(ang_b)]])
    
    D = A[0,0]*A[1,1]-A[0,1]*A[1,0]

    """ this should never happen though """
    if(D == 0.):
        print("MEGA PBM :::  DETERMINANT IS ZERO")
        return -9999, -9999

    
    xy = A.dot([x_b, x_a])/D
    x, y = xy[0], xy[1]

    
    if(dc.evt_list[-1].det == 'pdhd'):
        if(is_inside_volume(module, x,y)==False):
            for d0 in [0] if v_a == 2 else [0, cf.unwrappers[module][v_a]]:
                for d1 in [0] if v_b == 2 else [0, cf.unwrappers[module][v_b]]:
                    xy = A.dot([x_b+d1, x_a+d0])/D
                    xt, yt = xy[0], xy[1]
                    if(is_inside_volume(module, xt,yt)==True):
                        x, y = xt, yt
                        break

    if(is_inside_volume(module, x,y)==True):
        return x,y, True
    else:
        return x,y, False



        
def single_hit_finder(modules = [cf.imod]):
    debug = False
    
    ''' reconstruction parameters '''
    
    ''' max nb of hits to make a single hit in a given view '''
    max_per_view  = dc.reco['single_hit']['max_per_view']

    ''' max distance allowed among the 3 views association to make a 3D point '''
    outlier_dmax = dc.reco['single_hit']['outlier_dmax']

    ''' if barycenter too large, do not store the single hit '''
    max_bary = dc.reco['single_hit']['max_bary']
    
    ''' DBSCAN distance search parameter '''
    eps = dc.reco['single_hit']['cluster_eps']    

    ''' min cluster size for DBSCAN '''
    min_samp = 1

    ''' veto distance for other activity near the single hit '''
    dist_veto = dc.reco['single_hit']['dist_veto']


    
    if(len(dc.hits_list) < 3):
        return

    ID_shift = dc.hits_list[0].ID

    if(dc.evt_list[-1].det != 'pdhd'):
        unwrappers = [[0,0,0]]
    else:
        unwrappers = [[0,0,0], [cf.unwrappers[cf.imod][0], 0, 0], [0, cf.unwrappers[cf.imod][1], 0], [cf.unwrappers[cf.imod][0], cf.unwrappers[cf.imod][1], 0]]


    
    pties = index.Property()
    pties.dimension = 4
                        
    ''' create a local rtree index (3D : module, view, time, cluster)'''
    rtree_idx = index.Index(properties=pties)

    """ make a subset of unmatched hits """
    free_hits = [x for x in dc.hits_list if x.is_free==True and x.signal == cf.view_type[x.module][x.view] and x.module in modules]

    if(debug): print('nb of free hits ', len(free_hits))
    
    n_tot_clusters = 0
    
    for iv in range(cf.n_view):
               
        """ cluster the free hits """
        for m in modules:
            hits = [x for x in free_hits if x.module == m and x.view==iv]
            data = [[x.X,x.Z] for x in hits]
            #print('SH: ', len(data))
            if(len(data) == 0):
                continue
            
            X = np.asarray(data)
            db = DBSCAN(eps=eps,min_samples=min_samp).fit(X)
            labels = db.labels_

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            count = Counter([l for l in labels])

            """ discard too large clusters """
            [h.set_cluster_SH(l+n_tot_clusters) for h,l in zip(hits,labels) if count[l] <= max_per_view]

            ''' fill local Rtree with free and isolated hits '''
            [rtree_idx.insert(h.ID, (m, h.view, h.start, h.cluster_SH, m, h.view, h.stop, h.cluster_SH)) for h in hits if h.cluster_SH >= 0]
            n_tot_clusters += n_clusters
            #print(rtree_idx.count(rtree_idx.bounds), ' in SH rtree')
            


        n_tot_clusters += n_clusters

    if(debug): print('nb of clusters : ', n_tot_clusters)
    
    ''' for each free hits, search in the other view if free hits are compatible in time '''
    for h in free_hits:
        if(h.is_free == False or h.cluster_SH < 0):
            continue

        
        start = h.start - 1.
        stop  = h.stop + 1.
        clus  = h.cluster_SH
        mod   = h.module

        overlaps = [[] for x in range(cf.n_view)]
        for iview in range(cf.n_view):
            if(iview == h.view):
                overlaps[iview].append(h)
            else:
                 intersect = list(rtree_idx.intersection((mod, iview, start, -9999, mod, iview, stop, 9999)))
                 [overlaps[iview].append(dc.hits_list[k-ID_shift]) for k in intersect if dc.hits_list[k-ID_shift].is_free and dc.hits_list[k-ID_shift].cluster_SH >=0]




        overlaps = check_times(h, overlaps)
        overlaps = check_cru(h, overlaps)
        if(check_nmatch(overlaps, max_per_view)==False):
            continue

        if(debug):
            print('\n------------')
            #h.mini_dump()
            [x.mini_dump() for ov in overlaps  for x in ov]
            print('overlaps: ', [len(ov) for ov in overlaps])
        ''' from the time-compatible set of hits, try to make a 3D point '''
        best_comb = None
        best_xy = None
        best_d = 9999
        best_u = None
        ok = False
        
        for u in unwrappers:
            unwrap = u
            d, xy, comb = h3d.compute_xy(overlaps, h, outlier_dmax, u)

            if(debug): print(u, ':', d, xy, comb)
            
            if(d >= 0 and d < outlier_dmax):
                ok = True
                if(d<best_d):
                    best_d = d
                    best_comb = comb
                    best_xy = xy
                    best_u = u

        if(ok == False):
            continue
        if(debug):
            h.mini_dump()
            print('good comb at ', best_xy, 'dist ', best_d)
        
        ''' expand the good combination with all clustered hits of the best combo '''        
        overlaps = [[] for x in range(cf.n_view)]
        for c,iview in zip(best_comb,range(cf.n_view)):
            intersect = list(rtree_idx.intersection((c.module, iview, -9999, c.cluster_SH, c.module, iview, 9999, c.cluster_SH)))
            [overlaps[iview].append(dc.hits_list[k-ID_shift]) for k in intersect if dc.hits_list[k-ID_shift].is_free]

        #overlaps = check_times(h, overlaps)
        overlaps = check_cru(h, overlaps)
        
        """ check that there is 1 or 2 overlaps in the other views """
        if(check_nmatch(overlaps, max_per_view)==False):
            continue

        n_sh = max([len(x) for x in overlaps])
        view_max = np.argmax([len(x) for x in overlaps])


        coord = []
        dist_max = []
        for hh in overlaps[view_max]:
            ov_comb = [overlaps[iv] if iv!=view_max else [hh] for iv in range(cf.n_view)]

            best_xy = None
            best_d = 9999

            ok = False
            
            for u in unwrappers:
                unwrap = u
                d, xy, comb = h3d.compute_xy(ov_comb, hh, outlier_dmax, u)
            
                if(d >= 0 and d < outlier_dmax):
                    ok = True
                    if(d<best_d):
                        best_d = d
                        best_xy = xy


            if(ok == False):
                continue
            else:
                coord.append((best_xy[0],best_xy[1]))
                dist_max.append(best_d)
            
        if(len(coord) == 0):
            continue
        
        bar_x, bar_y, bar_z, bar_dmax = barycenter(coord, [x.Z for x in list(chain(*overlaps))])
        bar_dmax = max(dist_max)


        """ compute the shortest distance to a 3D track """
        d_min_3D = closest_activity_3D(bar_x, bar_y, bar_z)
        d_min_2D = closest_activity_2D(bar_x, bar_y)


        nhits = [len(x) for x in overlaps]
        IDs = [[x.ID for x in ov] for ov in overlaps]
        
        
        sh_ID = dc.evt_list[-1].n_single_hits + dc.n_tot_sh
        
        sh = dc.singleHits(sh_ID, cf.imod, nhits, IDs, bar_x, bar_y, bar_z, bar_dmax, d_min_3D, d_min_2D)
        
        for iv in range(cf.n_view):
            n = veto(iv, overlaps[iv], dist_veto)
            sh.set_veto(iv, n)

        
        for ov in overlaps:            
            sh.set_view(*compute_sh_properties(ov))

        if(sh.d_bary_max <= max_bary):

            for ov in overlaps:                        
                for hit in ov:
                    hit.set_match_sh(sh_ID)

            sh.set_timestamp()
            dc.single_hits_list.append(sh)
            dc.evt_list[-1].n_single_hits += 1

                      
