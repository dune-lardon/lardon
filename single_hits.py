import config as cf
import data_containers as dc
import lar_param as lar
from rtree import index
import numpy as np
import channel_mapping as cmap
from itertools import chain
from collections import Counter

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

def veto(hits, nchan, nticks, nchan_int, nticks_int):

    """ find the largest hit in the given view """
    best_hit = -1
    max_Q = 0.
    for h in hits:
        if(np.fabs(h.charge) > max_Q):
            max_Q = h.charge
            best_hit = h
    

    if(best_hit.daq_channel+1 in cf.broken_channels):
        return True, -1, -1, 1

    if(best_hit.daq_channel-1 in cf.broken_channels):
        return True, -1, -1, 1

    """ time/position cut """
    if(best_hit.channel%cf.view_chan_repet[best_hit.view] < nchan or best_hit.channel%cf.view_chan_repet[best_hit.view] > cf.view_chan_repet[best_hit.view]-nchan):
        return True, -1, -1, 1

    if(best_hit.start < nticks or best_hit.stop > cf.n_sample-nticks):
        return True, -1, -1, 1


    
    tmin, tmax = best_hit.start - nticks, best_hit.stop + nticks+1
    vetoed = False

    daqchan = best_hit.daq_channel
    daqch_neighbours = [-1 for i in range(2*nchan+1)]
    daqch_neighbours[nchan] = daqchan

    for i in range(nchan):
        daqch_neighbours[nchan-i-1] = dc.chmap[daqch_neighbours[nchan-i]].prev_daqch
        daqch_neighbours[nchan+i+1] = dc.chmap[daqch_neighbours[nchan+i]].next_daqch
    
    for i in daqch_neighbours:
        if(i in cf.broken_channels):
            continue
        if(i < 0):
            continue

        roi = ~dc.mask_daq[i, tmin:tmax]
        index = np.where(roi==True)[0]
            
        for ir in index:
            vetoed = vetoed | in_veto_region(dc.chmap[i].vchan, ir+tmin, best_hit.channel, best_hit.max_t, nchan_int, nticks_int)


    if(vetoed == False):
        int_q, int_pos_q, int_neg_q = 0., 0., 0.
        for i in daqch_neighbours:
            g = dc.chmap[i].gain
            d = dc.data_daq[i, best_hit.start-nticks_int:best_hit.stop+nticks_int+1]
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
    bx = np.mean([x[2] for x in coord])
    by = np.mean([x[3] for x in coord])
    bz = np.mean(all_z)

    dmax = np.max([np.sqrt(pow(bx-h[2],2)+pow(by-h[3], 2)) for h in coord])
    return bx, by, bz, dmax

def get_hit_xy(ha, hb):
    v_a, v_b = ha.view, hb.view
    ang_a = np.radians(cf.view_angle[v_a])
    ang_b = np.radians(cf.view_angle[v_b])
    x_a, x_b = ha.X, hb.X

    mod_a, mod_b = ha.module, hb.module
    if(mod_a != mod_b):
        return -9999, -9999
    
    A = np.array([[-np.cos(ang_a), np.cos(ang_b)],
                  [-np.sin(ang_a), np.sin(ang_b)]])
    
    D = A[0,0]*A[1,1]-A[0,1]*A[1,0]

    """ this should never happen though """
    if(D == 0.):
        print("MEGA PBM :::  DETERMINANT IS ZERO")
        return -9999, -9999

    
    xy = A.dot([x_b, x_a])/D
    x, y = xy[0], xy[1]
    
    
    return x,y


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
    free_hits = [x for x in dc.hits_list if x.is_free==True and x.signal == cf.view_type[x.view]]


    for h in free_hits: 
        start = h.start
        stop  = h.stop
        
        rtree_idx.insert(h.ID, (h.view, start, h.view, stop))


    for h in free_hits:
        if(h.is_free == False):
            continue
        start = h.start
        stop  = h.stop

        overlaps = [[] for x in range(cf.n_view)]

        for iview in range(cf.n_view):
            intersect = list(rtree_idx.intersection((iview, start, iview, stop)))
            [overlaps[iview].append(dc.hits_list[k-ID_shift]) for k in intersect]

            
        """ check that there is 1 or 2 overlaps in the other views """
        if(check_nmatch(overlaps, max_per_view)==False):
            continue
        
        """ get the 3D coordinates of all hits combination """
        coord = []
        for iview in range(cf.n_view-1):
            for ha in overlaps[iview]:
                for jview in range(iview+1, cf.n_view):
                    for hb in overlaps[jview]:
                        x,y = get_hit_xy(ha, hb)                        
                        coord.append((ha.ID, hb.ID, x, y))


        """ compute the median (x,y) coordinates, and remove the far away hits"""
        med_x, med_y, to_rm = find_outliers(coord, outlier_dmax)

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
                        x,y = get_hit_xy(ha, hb)                        
                        coord.append((ha.ID, hb.ID, x, y))
        

        bar_x, bar_y, bar_z, bar_dmax = barycenter(coord, [x.Z for x in list(chain(*overlaps))])



        """ compute the shortest distance to a 3D track """
        d_min_3D = closest_activity_3D(bar_x, bar_y, bar_z)
        d_min_2D = closest_activity_2D(bar_x, bar_y)

        nhits = [len(x) for x in overlaps]

        IDs = [[x.ID for x in ov] for ov in overlaps]

        module = h.module 
        sh_ID = dc.evt_list[-1].n_single_hits + dc.n_tot_sh
        
        sh = dc.singleHits(sh_ID, module, nhits, IDs, bar_x, bar_y, bar_z, bar_dmax, d_min_3D, d_min_2D)
        
        for iv in range(cf.n_view):
            v, q, p, n = veto(overlaps[iv], veto_nchan, veto_nticks, int_nchan, int_nticks)

            sh.set_veto(iv, v, q, p, n)
        
        for ov in overlaps:            
            sh.set_view(*compute_sh_properties(ov))
            
            for hit in ov:
                hit.set_match_sh(sh_ID)
                start, stop = hit.start, hit.stop
                rtree_idx.delete(hit.ID, (hit.view, start, hit.view, stop))

        sh.set_timestamp()
        dc.single_hits_list.append(sh)
        dc.evt_list[-1].n_single_hits += 1
        #sh.dump()
