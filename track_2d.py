import config as cf
import data_containers as dc
import math
import numpy as np
from operator import itemgetter

import scipy.stats as stat
import scipy.spatial as spatial
import scipy.sparse.csgraph as csgr

import R_tree as myrtree

from scipy.interpolate import UnivariateSpline

import pierre_filter as pf


def linear_reg(X, Y, cut):
    N = len(X)
    x0, y0 = X[0], Y[0]
    fits = []
    for i in range(1, N):
        for j in range(i+1, N):
            x = [x0, X[i], X[j]]
            y = [y0, Y[i], Y[j]]
            if x[0] == x[1]:
                x[0] += 0.00000000001
            slope, intercept, r, p, s = stat.linregress(x, y)
            if(r**2 > cut):
                fits.append( (r**2, slope, intercept, i, j) )

    if(len(fits)==0):
        return (9999., 0, 0, -1, -1)
    elif(len(fits)==1):
        return fits[0]
    else:
        """ sort by correlation factor """
        fits = sorted(fits, key=itemgetter(0), reverse=True)
        return fits[0]

def get_path(Pr, i, j):
    path = [j]
    k = j
    while Pr[i, k] != -9999:
        path.append(Pr[i, k])
        k = Pr[i, k]
    return path[::-1]


def dump_track(idx):
    t = dc.tracks2D_list[idx]

    print("Track ", idx)
    print("VIEW : ", t.view)
    print("NB of hits attached ", t.n_hits)
    print("X : ", t.path[0][0], " to ", t.path[0][1])
    print("Z : ", t.path[-1][0], " to ", t.path[-1][1])
    print("slope : %.2f [%.2f] to %.2f [%.2f]"%(t.ini_slope, t.ini_slope_err, t.end_slope, t.end_slope_err))
    print("Final Chi2 %.2f"%(t.chi2))
    print(" ")



def refilter_and_find_drays(idtrk, debug=False):     
    y_err = dc.reco['track_2d']['y_error']
    slope_err = dc.reco['track_2d']['slope_error']
    pbeta = dc.reco['track_2d']['pbeta']
    dray_dmax = dc.reco['track_2d']['dray_dmax']
    
    """error on y axis, error on slope, pbeta hyp"""
    filt = pf.PFilter(y_err, slope_err, pbeta)
    n_NN = 4

    track = [x for x in dc.tracks2D_list if x.trackID == idtrk]
    if(len(track) == 0 or len(track) > 1):
        print(" THERE IS AN ID PROBLEM !!")
    else:
        track = track[0]

        
    if(debug):
        print('\ntrack', idtrk, ' has ', track.n_hits, 'hits, ' , track.n_hits_dray, ' delta rays')

    track.remove_all_drays()
    hits = [x for x in dc.hits_list if x.match_2D==idtrk ]
    
    if(debug):
        print('track', idtrk, ' now  has ', track.n_hits, 'hits, ' , track.n_hits_dray, ' delta rays, check', len(hits))

    
    """sort by decreasing Z and increasing channel """
    hits = sorted(hits, key=lambda k: (-1.*k.Z, k.X))

    coord = np.asarray([(x.X, x.Z) for x in hits])
       
    """ compute the distance between each points"""
    graph = spatial.distance.cdist(coord, coord, 'euclidean')
    """ keep only the two closest points """
    graph = graph * (graph < np.sort(graph, axis=-1)[:,[n_NN]])

    """ compute the MST from this graph """
    T = csgr.minimum_spanning_tree(csgraph=graph)

    """ get the number of disconnected graphs """

    T = T.toarray()
    n_elem = np.count_nonzero(T, axis=0) + np.count_nonzero(T, axis=1)
    borders = np.nonzero(n_elem==1)[0]
    vertex  = np.nonzero(n_elem>2)[0]


    if(debug):
        print('n borders: ', len(borders), ' vertex ', len(vertex))


    """ identify potential delta rays from MST"""
    drays = []
    D, Pr = csgr.shortest_path(T, directed=False, method='FW', return_predecessors=True)
    d_min = 9999.
    for v in vertex:
        d_min = 9999.
        for b in borders:
            if(D[v, b] < d_min):
                p = get_path(Pr,v,b)
                d_min = D[v,b]
        for i in p[1:]:
            drays.append(i)

    if(debug):
        print('n dray ', len(drays), ' dmin = ', d_min)
        

    trk_hits = []
    for i in range(len(hits)):
        if(i not in drays):
            trk_hits.append(hits[i])


    if(debug):
        print('after graph : track', idtrk, ' has ', track.n_hits, 'hits, ' , track.n_hits_dray, ' delta rays', " TEST trk", len(trk_hits))

        
    """sort by decreasing Z and increasing channel """
    trk_hits = sorted(trk_hits, key=lambda k: (-1.*k.Z, k.X))

    x_r = [x.X for x in reversed(trk_hits)]
    z_r = [x.Z for x in reversed(trk_hits)]
    
    """ spline needs unique 'x' points to work --> remove duplicate """
    z_r_unique, z_idx = np.unique(z_r, return_index=True)
    x_r_unique, x_idx = np.unique(x_r, return_index=True)
    
    x_r = np.asarray(x_r)
    x_r_uz = x_r[z_idx]

    z_r = np.asarray(z_r)
    z_r_ux = z_r[x_idx]


    """at least 3 points for the spline """
    if(len(z_r_unique) < 4 and len(x_r_unique) < 4):
        print(" ... Track ", idtrk, " has not enough point to spline (Z: only ", len(z_r_unique), " vs ", len(z_r), ", X: only ",len(x_r_unique), " vs ", len(x_r),")")
        track.update_forward(9999., 9999., 9999.)
        track.update_backward(9999., 9999., 9999.)

    else:
        if(len(z_r_unique) >=4):
            spline_z = UnivariateSpline(z_r_unique, x_r_uz)        
            res_z = spline_z.get_residual()
        else:
            res_z = 9999
        if(len(x_r_unique) >= 4):
            spline_x = UnivariateSpline(x_r_unique, z_r_ux)
            res_x = spline_x.get_residual()
        else:
            res_x = 9999

        if(res_z < res_x):
            for i, (ix, iz) in enumerate(zip(x_r, z_r)):
                if(np.fabs(spline_z(iz) - ix) > dray_dmax):
                    drays.append(i)
            deriv = spline_z.derivative()
            track.update_forward(res_z, deriv(z_r[0]), deriv(z_r[0])*0.05)
            track.update_backward(res_z, deriv(z_r[-1]), deriv(z_r[-1])*0.05)

        else:
            for i, (ix, iz) in enumerate(zip(x_r, z_r)):
                if(np.fabs(spline_x(ix) - iz) > dray_dmax):
                    drays.append(i)
            deriv = spline_x.derivative()
            track.update_forward(res_x, 1./deriv(z_r[0]), (1./deriv(z_r[0]))*0.05)
            track.update_backward(res_x, 1./deriv(z_r[-1]), (1./deriv(z_r[-1]))*0.05)
            

    
    """ final trk/dray separation """
    trk_hits = []
    for i in range(len(hits)):
        if(i in drays):
            #trk_drays.append(hits[i])
            hdr = hits[i]
            track.add_drays(hdr.X, hdr.Z, hdr.charge, hdr.ID)
            hdr.set_match_dray(idtrk)
        else:
            trk_hits.append(hits[i])
           

        
    if(debug):
        print('after spline, track', idtrk, ' has ', track.n_hits, 'hits, ' , track.n_hits_dray, ' delta rays check ::: ', len(drays))


    
    coord = [(x.X, x.Z) for x in trk_hits]
    charge = [x.charge for x in trk_hits]
    IDs = [x.ID for x in trk_hits]
    track.reset_path(coord, charge, IDs)

    track.set_match_hits_2D(idtrk)

    if(debug):
        print('after spline, track', idtrk, ' has ', track.n_hits, 'hits, ' , track.n_hits_dray, ' track hits check ', len(trk_hits))



def find_tracks_rtree(direction="vertical"):

    min_hits = dc.reco['track_2d']['min_nb_hits']
    rcut = dc.reco['track_2d']['rcut']
    chicut = dc.reco['track_2d']['chi2cut']
    y_err = dc.reco['track_2d']['y_error']
    slope_err = dc.reco['track_2d']['slope_error']
    pbeta = dc.reco['track_2d']['pbeta']
    slope_max = dc.reco['track_2d']['slope_max']

    
    """error on y axis, error on slope, pbeta hyp"""
    filt = pf.PFilter(y_err, slope_err, pbeta)

    
    trackID = len(dc.tracks2D_list) + dc.n_tot_trk2d

    """ initialize the R-tree """
    tt = myrtree.R_tree(rcut)

    
    for iview in range(cf.n_view):

        hits = [x for x in dc.hits_list if x.view==iview and x.module == cf.imod and x.is_free == True]
        n_hits = len(hits)

        if(n_hits < min_hits):
            continue

        """sort by decreasing Z and increasing channel """
        hits.sort()

        """ build the R-tree """
        tt.create_index(iview)

        [tt.insert_hit(h, i) for i,h in enumerate(hits)]

        visited = np.zeros((n_hits),dtype=bool)

        seeded = False

        while(np.sum(visited) < n_hits):

            idx_list = []

            """get the first not yet visited hit in the list"""
            idx = np.argmax(visited==0)
            idx_list.append(idx)


            """ in case everything has already been visited """
            if(idx == 0 and visited[idx] is True):
                break

            visited[idx] = True
            tt.remove_hit(hits[idx], idx)

            """ get the N nearest hits """
            nn_id = tt.nearest_id(hits[idx], 5)
            """ sort them by closest distance """
            nn = [(i,tt.distance(hits[idx], hits[i])) for i in nn_id if tt.close_enough(hits[idx], hits[i])]
            nn.sort(key=itemgetter(1))


            """start the filter"""
            if(seeded is False and len(nn)>2):

                """ fit lines with all NN combination and idx """
                if(direction == "vertical"):
                    X = [hits[idx].Z]
                    Y = [hits[idx].X]
                    X.extend([hits[i].Z for i,d in nn])
                    Y.extend([hits[i].X for i,d in nn])
                    
                    """ returns the sorted fit results """
                    fit = linear_reg(X, Y, 0.9)
                    
                    x0, y0, t0 = hits[idx].Z, hits[idx].X, hits[idx].start #

                else:
                    X = [hits[idx].X]
                    Y = [hits[idx].Z]
                    X.extend([hits[i].X for i,d in nn])
                    Y.extend([hits[i].Z for i,d in nn])
                    
                    """ returns the sorted fit results """
                    fit = linear_reg(X, Y, 0.9)
                    
                    x0, y0, t0 = hits[idx].X, hits[idx].Z, hits[idx].start #

                    
                if(fit[0] != 9999):
                    seeded = True
                    slope  = fit[1]
                    intercept = fit[2]
                    ystart  = slope*x0 + intercept
                    filt.initiate(ystart, slope)
                    
                    if(direction == "vertical"):
                        track = dc.trk2D(trackID, iview, slope, slope_err, y0, x0, t0, hits[idx].charge, hits[idx].ID, filt.getChi2())
                    else:
                        track = dc.trk2D(trackID, iview, slope, slope_err, x0, y0, t0, hits[idx].charge, hits[idx].ID, filt.getChi2())
                    for i in [fit[3], fit[4]]:
                        """ add the seeding hits to the filter and remove them from the index """
                        nn_idx = nn[i-1][0]
                        
                        if(direction == "vertical"):
                            x1, y1, t1 = hits[nn_idx].Z, hits[nn_idx].X, hits[nn_idx].stop #t
                            chi2_up = filt.update(y1, x1-x0)
                            track.add_hit_update(slope, filt.getSlopeErr(), y1, x1, t1, hits[nn_idx].charge, hits[nn_idx].ID, filt.getChi2())

                        else:
                            x1, y1, t1 = hits[nn_idx].X, hits[nn_idx].Z, hits[nn_idx].stop #t
                            chi2_up = filt.update(y1, x1-x0)
                            track.add_hit_update(slope, filt.getSlopeErr(), x1, y1, t1, hits[nn_idx].charge, hits[nn_idx].ID, filt.getChi2())

                        idx_list.append(nn_idx)

                        visited[nn_idx] = True
                        tt.remove_hit(hits[nn_idx], nn_idx)



            """update the track from nearby hits"""
            finished = False

            while(seeded is True and finished is False and np.sum(visited) < n_hits):
                idx = nn_idx
                x0, y0 = x1, y1

                """ get the N nearest hits """
                nn_id = tt.nearest_id(hits[idx], 15)

                """ sort the NN hits by |pred-meas| distance if they are close enough """
                if(direction == "vertical"):
                    nn = [(i,filt.delta_y(hits[i].X, hits[i].Z-x0)) for i in nn_id if tt.close_enough(hits[idx], hits[i])]
                else:
                    nn = [(i,filt.delta_y(hits[i].Z, hits[i].X-x0)) for i in nn_id if tt.close_enough(hits[idx], hits[i])]
                    
                nn.sort(key=itemgetter(1))


                if(len(nn)==0):
                    finished = True
                    seeded = False

                    if(track.n_hits >= min_hits):
                        dc.tracks2D_list.append(track)
                        [hits[i].set_match_2D(trackID) for i in idx_list]

                        refilter_and_find_drays(trackID)

                        if(np.fabs(track.ini_slope) < slope_max and np.fabs(track.end_slope) < slope_max):
                            dc.evt_list[-1].n_tracks2D[iview] += 1
                            trackID += 1

                        else:
                            del dc.tracks2D_list[-1]
                            track.remove_all_drays()
                            [hits[i].reset_match() for i in idx_list]
                    continue

                
                updated = False

                ok_idx = []
                best_idx = -1
                best_chi2 = 99999.

                for j,d in nn:
                    nn_idx = j
                    if(d > 2.): continue
                    if(direction=="vertical"):
                        x1, y1 = hits[nn_idx].Z, hits[nn_idx].X
                    else:
                        x1, y1 = hits[nn_idx].X, hits[nn_idx].Z
                        
                    if(x1 >= x0):
                        if(tt.overlap_in_time(hits[idx], hits[nn_idx])==False):
                            continue

                    yp = filt.predict(x1-x0)
                    chi2m = filt.chi2_if_update(y1, x1-x0)
                    slope = (y1-y0)/(x1-x0) if x1!=x0 else 0
                    prod_slope = slope * filt.getSlope()

                    if(chi2m < chicut and prod_slope >= 0.):
                        ok_idx.append( (nn_idx, tt.distance(hits[idx], hits[nn_idx])) )

                        if(chi2m < best_chi2):
                            best_idx = nn_idx
                            best_chi2 = chi2m



                """ sort accepted hits by distance to idx """
                ok_idx.sort(key=itemgetter(1))


                if(len(ok_idx) > 0):
                    for j,t in ok_idx:

                        nn_idx = j
                        if(direction=="vertical"):
                            x1, y1, t1 = hits[nn_idx].Z, hits[nn_idx].X, hits[nn_idx].stop #t
                        else:
                            x1, y1, t1 = hits[nn_idx].X, hits[nn_idx].Z, hits[nn_idx].stop #t


                        d = filt.delta_y(y1, x1-x0)
                        if(d>2.):
                            continue

                        chi2m = filt.chi2_if_update(y1, x1-x0)

                        if(chi2m < chicut):

                            chi2_up = filt.update(y1, x1-x0)
                            if(direction=="vertical"):
                                track.add_hit_update(filt.getSlope(), filt.getSlopeErr(), y1, x1, t1, hits[nn_idx].charge, hits[nn_idx].ID, filt.getChi2())
                            else:
                                track.add_hit_update(filt.getSlope(), filt.getSlopeErr(), x1, y1, t1, hits[nn_idx].charge, hits[nn_idx].ID, filt.getChi2())
                                
                            idx_list.append(nn_idx)


                            visited[nn_idx]=True
                            tt.remove_hit(hits[nn_idx], nn_idx)
                            updated = True
                            x0, y0 = x1, y1
                        else:
                            continue


                        if(j==best_idx):
                            break
                if(updated is False or np.sum(visited) == n_hits):
                    finished = True
                    seeded = False



                    if(track.n_hits >= min_hits):
                        dc.tracks2D_list.append(track)
                        [hits[i].set_match_2D(trackID) for i in idx_list]

                        refilter_and_find_drays(trackID)

                        if(np.fabs(track.ini_slope) < slope_max and np.fabs(track.end_slope) < slope_max):
                            dc.evt_list[-1].n_tracks2D[iview] += 1
                            trackID += 1

                        else:
                            del dc.tracks2D_list[-1]
                            track.remove_all_drays()
                            [hits[i].reset_match() for i in idx_list]

                    continue


    return
