import config as cf
import data_containers as dc
import lar_param as lar
from rtree import index
import numpy as np
import channel_mapping as cmap
from itertools import chain
from collections import Counter
import hits_3d as h3d
import sklearn.cluster as skc



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

def in_veto_region(ch, t, hit_ch, hit_t, nchan_int, nticks_int):
    if(ch >= hit_ch - nchan_int and ch <= hit_ch + nchan_int):
        if(t >= hit_t - nticks_int and t <= hit_t +nticks_int):
            return False
    return True

def veto(module, hits, nchan, nticks, nchan_int, nticks_int):

    """ find the largest hit in this view """
    best_hit = -1
    max_Q = 0.
    
    
    for h in hits:        
        charge = np.fabs(h.charge_pos) + np.fabs(h.charge_neg)
        if(np.fabs(charge) > max_Q):
            max_Q = charge
            best_hit = h

    #print('best hit ')
    #best_hit.mini_dump()


    if(best_hit.glob_channel+1 in cf.broken_channels):
        return True, -1, -1, 1

    if(best_hit.glob_channel-1 in cf.broken_channels):
        return True, -1, -1, 1

    """ time/position cut """
    """ irrelevant for pdhd uv, and can be done offline """
    #if(best_hit.channel%cf.view_chan_repet[best_hit.view] < nchan or best_hit.channel%cf.view_chan_repet[best_hit.view] > cf.view_chan_repet[best_hit.view]-nchan):
        #return True, -1, -1, 1

    if(best_hit.start < nticks or best_hit.stop > cf.n_sample[module]-nticks):
        return True, -1, -1, 1

        
    tmin, tmax = best_hit.start - nticks, best_hit.stop + nticks+1
    vetoed = False
    
    daqch_shift = cf.module_daqch_start[module]
    #print('daqch shift ', daqch_shift)
    
    daqchan = best_hit.daq_channel
    daqch_neighbours = [-1 for i in range(2*nchan+1)]
    daqch_neighbours[nchan] = daqchan

    for i in range(nchan):
        daqch_neighbours[nchan-i-1] = dc.chmap[daqch_neighbours[nchan-i]].prev_daqch
        daqch_neighbours[nchan+i+1] = dc.chmap[daqch_neighbours[nchan+i]].next_daqch




    #print(daqchan, " : ", daqch_neighbours)
    #print([(dc.chmap[x].view, dc.chmap[x].vchan) for x in daqch_neighbours])

    
    for i in daqch_neighbours:
        if(i < 0):
            continue
        if(dc.chmap[i].globch in cf.broken_channels):
            continue
        if(dc.chmap[i].module != dc.chmap[daqchan].module or dc.chmap[i].view != dc.chmap[daqchan].view):            
            continue

        
        
        roi = ~dc.mask_daq[i-daqch_shift, tmin:tmax]
        index = np.where(roi==True)[0]
            
        for ir in index:
            vetoed = vetoed | in_veto_region(dc.chmap[i].vchan, ir+tmin, best_hit.channel, best_hit.max_t, nchan_int, nticks_int)


    if(vetoed == False):
        int_q, int_pos_q, int_neg_q = 0., 0., 0.
        for i in daqch_neighbours:
            if(i < 0):
                continue
            if(dc.chmap[i].globch in cf.broken_channels):
                continue
            if(dc.chmap[i].module != dc.chmap[daqchan].module or dc.chmap[i].view != dc.chmap[daqchan].view):            
                continue

            
            g = dc.chmap[i].gain
            d = dc.data_daq[i-daqch_shift, best_hit.start-nticks_int:best_hit.stop+nticks_int+1]
            int_q += np.sum(d)*g
            int_pos_q += np.sum((d>0)*d)*g
            int_neg_q += np.sum((d<0)*d)*g


        return False, int_q, int_pos_q, int_neg_q

    return True, -1, -1, 1
    
            

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


def same_view_compatibility(ha, hb):
    if(np.fabs(ha.channel - hb.channel)!=1):
        return False
    if(np.fabs(ha.max_t-hb.max_t)>15):
        return False
    return True



def single_hit_finder():    
    #cmap.arange_in_view_channels()


    max_per_view  = dc.reco['single_hit']['max_per_view']
    outlier_dmax = dc.reco['single_hit']['outlier_dmax']
    veto_nchan = dc.reco['single_hit']['veto_nchan']
    veto_nticks = dc.reco['single_hit']['veto_nticks']
    int_nchan = dc.reco['single_hit']['int_nchan']
    int_nticks = dc.reco['single_hit']['int_nticks']



    if(len(dc.hits_list) < 3):
        return

    ID_shift = dc.hits_list[0].ID


    pties = index.Property()
    pties.dimension = 2
                        
    ''' create an rtree index (3D : view, time)'''
    rtree_idx = index.Index(properties=pties)

    """ make a subset of unmatched hits """
    free_hits = [x for x in dc.hits_list if x.is_free==True and x.signal == cf.view_type[cf.imod][x.view] and x.module == cf.imod]


    for h in free_hits: 
        start = h.start
        stop  = h.stop
        
        rtree_idx.insert(h.ID, (h.view, start, h.view, stop))

    #print(cf.imod, 'module has : ', len(free_hits))

    for h in free_hits:
        if(h.is_free == False):
            continue
        start = h.start-0.5
        stop  = h.stop+0.5

        overlaps = [[] for x in range(cf.n_view)]

        for iview in range(cf.n_view):
            intersect = list(rtree_idx.intersection((iview, start, iview, stop)))
            [overlaps[iview].append(dc.hits_list[k-ID_shift]) for k in intersect]

        #if(h.view==0 and h.module==2):
            #h.mini_dump()
            #print(' has ', [len(ov) for ov in overlaps])
            
        """ check that there is 1 or 2 overlaps in the other views """
        if(check_nmatch(overlaps, max_per_view)==False):
            continue

        """ get the 3D coordinates of all hits combination """
        coord = []
        for iview in range(cf.n_view-1):
            for ha in overlaps[iview]:
                for jview in range(iview+1, cf.n_view):
                    for hb in overlaps[jview]:
                        x, y, ok = get_hit_xy(cf.imod, ha, hb)                        
                        coord.append((ha.ID, hb.ID, x, y))

        #if(h.view==0 and h.module==2):
            #print(coord)
            
        """ compute the median (x,y) coordinates, and remove the far away hits"""
        med_x, med_y, to_rm = find_outliers(coord, outlier_dmax)
        if(h.view==0 and h.module==2):
            print(med_x, med_y, to_rm)
            
        i=0
        while(i < len(to_rm)):
            o = to_rm[i]
            for iv in range(len(overlaps)):
                for ih in range(len(overlaps[iv])):
                    if(overlaps[iv][ih].ID == o):
                        overlaps[iv].pop(ih)
                        i+=1
                        break
            i+=1
        if(check_nmatch(overlaps, max_per_view)==False):
            continue

        """ re-get the 3D coordinates of all hits combination """
        coord = []
        for iview in range(cf.n_view-1):
            for ha in overlaps[iview]:
                for jview in range(iview+1, cf.n_view):
                    for hb in overlaps[jview]:
                        x,y, ok = get_hit_xy(cf.imod, ha, hb)                        
                        coord.append((x,y))#(ha.ID, hb.ID, x, y))
        
        #if(h.view==0 and h.module==2):
            #print(coord)

        bar_x, bar_y, bar_z, bar_dmax = barycenter(coord, [x.Z for x in list(chain(*overlaps))])
        #if(h.view==0 and h.module==2):
            #print(bar_x, bar_y, bar_z, bar_dmax)

        """ compute the shortest distance to a 3D track """
        d_min_3D = closest_activity_3D(bar_x, bar_y, bar_z)
        d_min_2D = closest_activity_2D(bar_x, bar_y)
        
        nhits = [len(x) for x in overlaps]
        
        IDs = [[x.ID for x in ov] for ov in overlaps]
        
        
        sh_ID = dc.evt_list[-1].n_single_hits + dc.n_tot_sh
        
        sh = dc.singleHits(sh_ID, cf.imod, nhits, IDs, bar_x, bar_y, bar_z, bar_dmax, d_min_3D, d_min_2D)
        
        for iv in range(cf.n_view):
            v, q, p, n = veto(cf.imod, overlaps[iv], veto_nchan, veto_nticks, int_nchan, int_nticks)

            sh.set_veto(iv, v, q, p, n)
        
        for ov in overlaps:            
            sh.set_view(*compute_sh_properties(ov))


        if(sh.max_t[0] <= sh.max_t[1] <= sh.max_t[2] and sh.d_bary_max <= 20):
            for ov in overlaps:                        
                for hit in ov:
                    hit.set_match_sh(sh_ID)
                    start, stop = hit.start, hit.stop
                    rtree_idx.delete(hit.ID, (hit.view, start, hit.view, stop))

            sh.set_timestamp()
            dc.single_hits_list.append(sh)
            dc.evt_list[-1].n_single_hits += 1
            #sh.dump()


        
def single_hit_finder_new():
    debug = False

    if(debug):
        import matplotlib.pyplot as plt
        import colorcet as cc
        fig = plt.figure()
        ax = [fig.add_subplot(131+i) for i in range(3)]
        [ax[i].sharey(ax[0]) for i in [1,2]]

    
    max_per_view  = dc.reco['single_hit']['max_per_view']
    outlier_dmax = dc.reco['single_hit']['outlier_dmax']
    veto_nchan = dc.reco['single_hit']['veto_nchan']
    veto_nticks = dc.reco['single_hit']['veto_nticks']
    int_nchan = dc.reco['single_hit']['int_nchan']
    int_nticks = dc.reco['single_hit']['int_nticks']

    eps = dc.reco['single_hit']['cluster_eps']
    min_samp = 1

    
    if(len(dc.hits_list) < 3):
        return

    ID_shift = dc.hits_list[0].ID

    if(dc.evt_list[-1].det != 'pdhd'):
        unwrappers = [[0,0,0]]
    else:
        unwrappers = [[0,0,0], [cf.unwrappers[cf.imod][0], 0, 0], [0, cf.unwrappers[cf.imod][1], 0], [cf.unwrappers[cf.imod][0], cf.unwrappers[cf.imod][1], 0]]


    
    pties = index.Property()
    pties.dimension = 3
                        
    ''' create an rtree index (3D : view, time, cluster)'''
    rtree_idx = index.Index(properties=pties)

    """ make a subset of unmatched hits """
    free_hits = [x for x in dc.hits_list if x.is_free==True and x.signal == cf.view_type[cf.imod][x.view] and x.module == cf.imod]

    n_tot_clusters = 0
    
    for iv in range(cf.n_view):
        hits = [x for x in free_hits if x.view==iv]
        if(len(hits)==0): continue

        
        
        """ squeeze y axis instead of rebinning or defining a new metric """
        data = [[x.X,x.Z] for x in hits]
        X = np.asarray(data)
        db = skc.DBSCAN(eps=eps,min_samples=min_samp).fit(X)
        labels = db.labels_

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        count = Counter([l for l in labels])

        #print(len(hits), ' --->>> ', n_clusters)
        #print(count)
        [h.set_cluster_SH(l+n_tot_clusters) for h,l in zip(hits,labels) if count[l] <= max_per_view]
        n_tot_clusters += n_clusters

        [rtree_idx.insert(h.ID, (h.view, h.start, h.cluster_SH, h.view, h.stop, h.cluster_SH)) for h in hits if h.cluster_SH >= 0]
    
        if(debug):
            
            #nok = sum([1 if h.cluster_SH>=0 else 0 for hits])
            
            #print('nb of hits considered : ', nok)
            #hallx = [h.X for h in hits]
            #hally =  [h.Z for h in hits]
            #clus = [l for l in labels if l >=0]
            hx = [h.X for h in hits if h.cluster_SH>=0]
            hz = [h.Z for h in hits if h.cluster_SH>=0]
            clus = [h.cluster_SH for h in hits if h.cluster_SH>=0]
            [ax[iv].scatter(h.X, h.Z, s=1, c='grey') for h in hits]
            
            ax[iv].scatter(hx, hz, s=1, c=clus, cmap=cc.cm.glasbey)

            #print([(h.X, h.Z, h.cluster_SH) for h in hits if h.cluster_SH>=0])

    if(debug):
        plt.show()
        
        

    for h in free_hits:
        if(h.is_free == False or h.cluster_SH < 0):
            continue
        #h.mini_dump()
        
        start = h.start-0.5
        stop  = h.stop+0.5
        clus  = h.cluster_SH
        
        overlaps = [[] for x in range(cf.n_view)]
        for iview in range(cf.n_view):
            if(iview == h.view):
                overlaps[iview].append(h)
            else:
                 intersect = list(rtree_idx.intersection((iview, start, -9999, iview, stop, 9999)))
                 [overlaps[iview].append(dc.hits_list[k-ID_shift]) for k in intersect if dc.hits_list[k-ID_shift].is_free]

        """
        print('overlaps with ', [len(x) for x in overlaps])

        for iv in range(3):
            if(iv==h.view):
                continue
            else:
                for oo in overlaps[iv]:
                    oo.mini_dump()
        """
        overlaps = check_times(h, overlaps)
                 
        """ check that there is 1 or 2 overlaps in the other views """
        if(check_nmatch(overlaps, max_per_view)==False):
            continue

        #print('ok')
        
        best_comb = None
        best_xy = None
        best_d = 9999
        best_u = None
        ok = False
        
        for u in unwrappers:
            unwrap = u
            d, xy, comb = h3d.compute_xy(overlaps, h, outlier_dmax, u)
            #print(d, xy)
            
            if(d >= 0 and d < outlier_dmax):
                ok = True
                if(d<best_d):
                    best_d = d
                    best_comb = comb
                    best_xy = xy
                    best_u = u

        if(ok == False):
            continue

        ''' expand the good combination with all clustered hits of the best combo '''        
        overlaps = [[] for x in range(cf.n_view)]
        for c,iview in zip(best_comb,range(cf.n_view)):
            intersect = list(rtree_idx.intersection((iview, -9999, c.cluster_SH, iview, 9999, c.cluster_SH)))
            [overlaps[iview].append(dc.hits_list[k-ID_shift]) for k in intersect if dc.hits_list[k-ID_shift].is_free]

        #overlaps = check_times(h, overlaps)
        
        """ check that there is 1 or 2 overlaps in the other views """
        if(check_nmatch(overlaps, max_per_view)==False):
            continue

        n_sh = max([len(x) for x in overlaps])
        view_max = np.argmax([len(x) for x in overlaps])

        #print('\n',[len(x) for x in overlaps], '--> ', n_sh, ' view ', view_max)
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

        #print('-- coord:', coord, 'dists ', dist_max)
        
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
            v, q, p, n = veto(cf.imod, overlaps[iv], veto_nchan, veto_nticks, int_nchan, int_nticks)

            sh.set_veto(iv, v, q, p, n)
        
        for ov in overlaps:            
            sh.set_view(*compute_sh_properties(ov))

        if(sh.d_bary_max <= 20):

            for ov in overlaps:                        
                for hit in ov:
                    hit.set_match_sh(sh_ID)
                    #start, stop = hit.start, hit.stop
                    #rtree_idx.delete(hit.ID, (hit.view, start, hit.view, stop))

            sh.set_timestamp()
            dc.single_hits_list.append(sh)
            dc.evt_list[-1].n_single_hits += 1
            #sh.dump()
                      
'''
def get_xy_leastsq():
   # Convert cell input [{x, y, ...}] to A, B form
    if B is None:
        A = np.transpose(np.stack(A), (1, 2, 0))  # 2*N*D > N*D*2
        A, B = A[:, :, 0], A[:, :, 1]

    # Find intersection
    V = B - A  # Vectors from A to B
    V = V / np.sqrt(np.sum(V * V, axis=1))[:, np.newaxis]  # Normalized vectors
    N, D = A.shape  # Number of points & dimensions

    T = np.einsum('ij,ik->ijk', V, V) - np.eye(D)  # V.*V-1 as D*N*D
    S = np.sum(T, axis=1)  # Sum T along N, as D*D
    C = np.einsum('ijk,ik->ij', T, A)  # T*A, as D*1
    X = np.linalg.lstsq(S, C.sum(axis=0), rcond=None)[0]  # Solve for X: S*X=C, in least squares sense

    # Checks
    if np.isnan(V).any():  # Zero length lines
        warnings.warn('One or more lines with zero length.', UserWarning)
    elif np.linalg.cond(S) > 1 / np.finfo(S.dtype).eps:  # Parallel lines
        warnings.warn('Lines are near parallel.', UserWarning)

    # Extra outputs
    P = R = x = p = l = None
    if 'P' in locals() or 'R' in locals():
        U = np.sum((X - A) * V, axis=1)  # dot(X-A,V) distance from A to nearest point on each line
        P = A + U[:, np.newaxis] * V  # Nearest point on each line
    
    if 'R' in locals():
        R = np.sqrt(np.sum((X - P) ** 2, axis=1))  # Distance from intersection to each line

    # Plot outputs
    if 'x' in locals():
        x = [X[i] for i in range(D)]  # Intersection point X
    
    if 'p' in locals():
        p = [P[:, i] for i in range(D)]  # Tangent points P
    
    if 'l' in locals():
        l = [np.stack((A[:, i], B[:, i]), axis=1) for i in range(D)]  # Initial lines A,B using cell format {x y..}

    return X, P, R, x, p, l
'''
