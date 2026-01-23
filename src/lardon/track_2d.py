import lardon.config as cf
import lardon.data_containers as dc
import lardon.pierre_filter as pf


import numpy as np
from operator import itemgetter

from scipy.spatial.distance import cdist
from scipy.interpolate import UnivariateSpline
from scipy.sparse.csgraph import minimum_spanning_tree, shortest_path



import time

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
    dray_thr = dc.reco['track_2d']['dray_thr']
    dray_dmax = dc.reco['track_2d']['dray_dmax']

    """ graph nearest neighbours cut """
    n_NN = 4
    
    """ create filter """
    pf.PFilter(y_err, slope_err, pbeta)

    """ Retrieve track """
    tracks = [x for x in dc.tracks2D_list if x.trackID == idtrk]
    if len(tracks) != 1:
        print("ERROR: Track ID issue!")
        return
    track = tracks[0]

    if debug:
        print(f"\ntrack {idtrk} has {track.n_hits} hits, {track.n_hits_dray} delta rays")

    """ reset the hit/delta ray separation """
    track.remove_all_drays()

    """ get the track hits """
    ID_shift = dc.hits_list[0].ID    
    hits = [dc.hits_list[i - ID_shift] for i in track.hits_ID]
    hits = sorted(hits, key=lambda k: (-k.Z, k.X))

    #print('\n\n track ', track.trackID)
    #print([h.ID for h in hits])
    
    
    """ build graph from hits """
    coords = np.array([(h.X, h.Z) for h in hits])
    
    if(len(coords) <= n_NN):
        return
    dist_matrix = cdist(coords, coords, 'euclidean')

    """ only keep the two closest points  of each hit in the graph """
    mask = dist_matrix < np.sort(dist_matrix, axis=-1)[:, [n_NN]]
    graph = dist_matrix * mask

    """ make MST graph from distance matrix """
    T = minimum_spanning_tree(graph).toarray()
    node_count = np.count_nonzero(T, axis=0) + np.count_nonzero(T, axis=1)
    
    borders = np.where(node_count == 1)[0]
    vertices = np.where(node_count > 2)[0]

    if debug:
        print(f"n borders: {len(borders)}, vertex: {len(vertices)}")

    """ identify delta ray candidates (first pass) """
    """ at MST vertices, search for the shortest vertex->border path, consider that as delta-rays """
    
    dray_indices = set()
    D, predecessors = shortest_path(T, directed=False, method='FW', return_predecessors=True)

    for v in vertices:
        min_dist = np.inf
        best_path = []
        for b in borders:
            if D[v, b] < min_dist:
                best_path = get_path(predecessors, v, b)
                min_dist = D[v, b]
        dray_indices.update(best_path[1:])

    """ if hits needs to be rejected """
    bad_hit_indices = set()
    
    if debug:
        print(f"Identified {len(dray_indices)} dray candidates from MST")

    """ separate track hits from drays """
    trk_hits = [(i, h) for i, h in enumerate(hits) if i not in dray_indices]
    trk_hits = sorted(trk_hits, key=lambda k: (-k[1].Z, k[1].X))


    """ now we will spline the track """
    """ univariate spline needs unique coordinate in increasing 'z' coordinates """
    x_r = np.array([h[1].X for h in reversed(trk_hits)])
    z_r = np.array([h[1].Z for h in reversed(trk_hits)])

    """ remove duplicate coordinates """
    z_r_unique, z_idx = np.unique(z_r, return_index=True)
    x_r_unique, x_idx = np.unique(x_r, return_index=True)

    x_r_uz = x_r[z_idx]
    z_r_ux = z_r[x_idx]

    """ make spline if enough points """
    if len(z_r_unique) < 4 and len(x_r_unique) < 4:
        print(f"Track {idtrk} has insufficient points for spline")
        track.update_forward(9999., 9999., 9999.)
        track.update_backward(9999., 9999., 9999.)
    else:
        res_z = res_x = 9999.
        if len(z_r_unique) >= 4:
            spline_z = UnivariateSpline(z_r_unique, x_r_uz)
            res_z = spline_z.get_residual()
        if len(x_r_unique) >= 4:
            spline_x = UnivariateSpline(x_r_unique, z_r_ux)
            res_x = spline_x.get_residual()

        """ choose the best spline from the two possibilities """
        spline = spline_z if res_z < res_x else spline_x
        res = min(res_z, res_x)
        inverse = res_z >= res_x

        for i, (ix, iz) in enumerate(zip(x_r, z_r)):
            val = spline(iz if not inverse else ix)
            delta = abs(val - (ix if not inverse else iz))
            """ hits a bit too far from the spline are considered drays """
            if dray_thr < delta < dray_dmax:
                dray_indices.add(trk_hits[i][0])
            elif delta >= dray_dmax:
                bad_hit_indices.add(trk_hits[i][0])
                
        """ re-define track slopes from the splines """
        deriv = spline.derivative()
        slope_start = deriv(z_r[0] if not inverse else x_r[0])
        slope_end = deriv(z_r[-1] if not inverse else x_r[-1])
        if inverse:
            slope_start = 1. / slope_start
            slope_end = 1. / slope_end
        track.update_forward(res, slope_start, slope_start * 0.05)
        track.update_backward(res, slope_end, slope_end * 0.05)

        
    # Final assignment of hits and drays
    final_trk_hits = []
    for i, hit in enumerate(hits):

        if i in bad_hit_indices:
            hit.reset_match()
            continue
        
        elif i in dray_indices:
            track.add_drays(hit.X, hit.Z, hit.charge, hit.ID)
            hit.set_match_dray(idtrk)
        else:
            final_trk_hits.append(hit)


    if debug:
        print(f"Final count: {track.n_hits} hits, {track.n_hits_dray} delta rays")
        print(f"DRAY IDs: {sorted(track.drays_ID)}")

    coord = [(h.X, h.Z) for h in final_trk_hits]
    charge = [h.charge for h in final_trk_hits]
    IDs = [h.ID for h in final_trk_hits]
    track.reset_path(coord, charge, IDs)
    track.set_match_hits_2D(idtrk)
    track.finalize_track()

    #print('FINAL track')
    #print(track.hits_ID)
    #print(track.drays_ID)

    if debug:
        print(f"Track {idtrk} final hit count: {track.n_hits}, delta rays: {track.n_hits_dray}")

        

    

def distanceX(hA, hB):
    dx = np.fabs(hB.X - hA.X)
    return dx

        
def find_tracks_hough(modules = [cf.imod]):

    """ min nb of hits to make a track """
    min_hits = dc.reco['track_2d']['min_nb_hits']
    """ filter chi2 cut to add a new hit to tracking """
    chicut = dc.reco['track_2d']['chi2cut']

    """ for filter initialization """
    y_err = dc.reco['track_2d']['y_error']
    slope_err = dc.reco['track_2d']['slope_error']
    pbeta = dc.reco['track_2d']['pbeta']
    """ too shallow tracks are discarded """
    slope_max = dc.reco['track_2d']['slope_max']

    """ Nearest Neighbours criteria for hough search """
    deltaX = dc.reco['track_2d']['hough_win_X']
    deltaZ = dc.reco['track_2d']['hough_win_Z']
    nn_min = dc.reco['track_2d']['hough_n_min']

    """ hough resolution """
    theta_res = dc.reco['track_2d']['hough_theta_res']
    rho_res = dc.reco['track_2d']['hough_rho_res']
    """ hough min score to seed a track """
    score_min = dc.reco['track_2d']['hough_min_score']

    """ max allowed gap between 2 hits to pursue tracking """
    max_dist_to_pred = dc.reco['track_2d']['max_gap']

    
    ID_shift = dc.hits_list[0].ID    
    
    """error on y axis, error on slope, pbeta hyp"""
    filt = pf.PFilter(y_err, slope_err, pbeta)


    trackID = len(dc.tracks2D_list) + dc.n_tot_trk2d    
    n_tot_hits = len(dc.hits_list)
    
    ''' R-tree all the hits '''
    tr = time.time()

    
    idx_start = [0]
    
    n_hits_per_module =  np.sum(dc.evt_list[-1].n_hits, axis=0)
    idx_start.extend(np.cumsum(n_hits_per_module))

    
    visited = np.zeros((n_tot_hits),dtype=bool)
    
    n_hits_tot = np.sum(dc.evt_list[-1].n_hits)

    """ only look at hits in a given view and module """
    for iview in range(cf.n_view):
        visited[:] = 1
        hits = []

        for m in modules:
            """ get the part in dc.hits_list of hits that belong to module m """
            idx0 = idx_start[m]
            idx1 = idx_start[m+1]
            hits.extend([h for h in dc.hits_list[int(idx0):int(idx1)] if h.view==iview])
            ''' set hits unvisited in that view '''
            ids = [h.ID - ID_shift for h in hits]
            visited[ids] = 0
            
        n_hits = len(hits)
        if(n_hits==0): continue
    
    
        points = np.array([[h.X, h.Z] for h in hits])

        seeded = False

        tracks = []
        th = time.time()

        """ tracking starts with seeding """
        while(np.sum(visited) < n_tot_hits):

            idx_list = []

            """get the first not yet visited hit in the list"""
            idx = np.argmax(visited==0)
            idx_list.append(idx)

            """ in case everything has already been visited """
            if( visited[idx] is True):
                break
        
            visited[idx] = 1

            tnn = time.time()

            the_hit = dc.hits_list[idx]
            
            ''' get all hits around query hit within +/- deltaX/Z '''
            intersect = list(dc.rtree_hit_idx.intersection((-999, iview, the_hit.X-deltaX, the_hit.Z-deltaZ, 999, iview, the_hit.X+deltaX, the_hit.Z+deltaZ)))


            ''' make sure those hits haven't been tested already '''
            nearest  = [dc.hits_list[k-ID_shift] for k in intersect if visited[k-ID_shift]==False]

            ''' give up if there are too few hits '''
            if(len(nearest) < nn_min ):
                continue

            
            query = np.array([the_hit.X, the_hit.Z])
            neighbours = np.array([[h.X, h.Z] for h in nearest])
            

            ''' from those hits, get the best line that goes through the query hit '''

            direction = 0 #horizontal search ie y=ax+b
            line, n = hough(query, neighbours, theta_res, rho_res)

            if(n<0):
                """when the found line is vertical, ie a=infinity
                then change the line to x=ay+b """
                
                neighbours = np.fliplr(neighbours)
                query = query[::-1]
                

                line, n = hough(query, neighbours, theta_res, rho_res)
                direction = 1
                

            if(n < score_min):                
                continue

            x0, y0, t0 = the_hit.X, the_hit.Z, the_hit.start #
            if(direction == 1):
                x0, y0 = y0, x0

            """ a track has been seeded! """
            seeded = True
            slope  = line[0]
            intercept = line[1]
            ystart  = slope*x0 + intercept
            filt.initiate(ystart, slope)
            track = dc.trk2D(trackID, iview, slope, slope_err, x0, y0, t0, the_hit.charge, the_hit.ID, filt.getChi2())

            finished = False
            x1, y1 = x0, y0
            nn_idx = idx

            """ let's add new hits to the track """
            while(seeded is True and finished is False and np.sum(visited) < n_tot_hits):
                idx = nn_idx        
                x0, y0 = x1, y1                

                xs, ys = x0, y0
                if(direction == 1):
                    xs, ys = y0, x0

                

                """ find the last hit nearest neighbours """
                """ NB: intersection is faster than nearest neighbours """
                nn_id = [i for i in dc.rtree_hit_idx.intersection((-999, iview, xs-deltaX, ys-deltaZ, 999, iview, xs+deltaX, ys+deltaZ))]

                """ sort the NN hits by |pred-meas| distance if they are close enough """
                if(direction == 0):
                    nn = [(i,filt.delta_y(dc.hits_list[i-ID_shift].Z, dc.hits_list[i-ID_shift].X-x0)) for i in nn_id if visited[i-ID_shift]==False and i!=idx]
                else:
                    nn = [(i,filt.delta_y(dc.hits_list[i-ID_shift].X, dc.hits_list[i-ID_shift].Z-x0)) for i in nn_id if visited[i-ID_shift]==False and i!=idx]      
                

                nn.sort(key=itemgetter(1))
                
                if(len(nn)==0):
                    """ no new hits to add, check if the track is OK """
                    finished = True
                    seeded = False

                    if(track.n_hits >= min_hits):
                        dc.tracks2D_list.append(track)
                        [dc.hits_list[i].set_match_2D(trackID) for i in idx_list]

                        refilter_and_find_drays(trackID)
                        
                        if(np.fabs(track.ini_slope) < slope_max and np.fabs(track.end_slope) < slope_max):
                            dc.evt_list[-1].n_tracks2D[iview] += 1
                            trackID += 1

                        else:
                            del dc.tracks2D_list[-1]
                            track.remove_all_drays()
                            [dc.hits_list[i].reset_match() for i in idx_list]
                    continue


                updated = False
                                
                ok_idx = []
                best_idx = -1
                best_chi2 = 99999.

                for j,d in nn:
                    nn_idx = j
                    if(d > max_dist_to_pred): continue

                    x1, y1 = dc.hits_list[nn_idx-ID_shift].X, dc.hits_list[nn_idx-ID_shift].Z
                    if(direction==1):
                        x1, y1 = y1, x1
                    
                    yp = filt.predict(x1-x0)
                    chi2m = filt.chi2_if_update(y1, x1-x0)
                    slope = (y1-y0)/(x1-x0) if x1!=x0 else 0
                    prod_slope = slope * filt.getSlope()


                    dx = x1-x0
                    if(chi2m < chicut and prod_slope >= 0. and dx*filt.getSlope() <=0 ):
                        ok_idx.append( (nn_idx, distanceX(dc.hits_list[idx], dc.hits_list[nn_idx-ID_shift])) )

                        
                        if(chi2m < best_chi2):
                            best_idx = nn_idx
                            best_chi2 = chi2m
                
                """ sort accepted hits by distance to idx """
                ok_idx.sort(key=itemgetter(1))

                if(len(ok_idx) > 0):
                    for j,t in ok_idx:

                        
                        nn_idx = j
                        nn_hit = dc.hits_list[nn_idx-ID_shift]
                        x1, y1, t1 = nn_hit.X, nn_hit.Z, nn_hit.stop #t
                        if(direction==1):
                            x1, y1 = y1, x1

                        d = filt.delta_y(y1, x1-x0)
                        if(d > max_dist_to_pred ):
                            continue

                        chi2m = filt.chi2_if_update(y1, x1-x0)

                        if(chi2m < chicut):

                            chi2_up = filt.update(y1, x1-x0)
                            track.add_hit_update(filt.getSlope(), filt.getSlopeErr(), x1, y1, t1, dc.hits_list[nn_idx-ID_shift].charge, dc.hits_list[nn_idx-ID_shift].ID, filt.getChi2())
                                
                            idx_list.append(nn_idx-ID_shift)


                            visited[nn_idx-ID_shift]=True

                            updated = True
                            x0, y0 = x1, y1
                            
                        else:
                            continue


                        if(j==best_idx):
                            break

                nn_idx -= ID_shift
                
                if(updated is False or np.sum(visited) == n_tot_hits):
                    """ no hits was added to the track, check if the track is OK """
                    finished = True
                    seeded = False



                    if(track.n_hits >= min_hits):
                            
                        dc.tracks2D_list.append(track)
                        [dc.hits_list[i].set_match_2D(trackID) for i in idx_list]

                        refilter_and_find_drays(trackID)
                        
                        if(np.fabs(track.ini_slope) < slope_max and np.fabs(track.end_slope) < slope_max):
                            dc.evt_list[-1].n_tracks2D[iview] += 1
                            trackID += 1

                        else:
                            del dc.tracks2D_list[-1]
                            track.remove_all_drays()
                            [dc.hits_list[i].reset_match() for i in idx_list]

                    continue

                
        #print(f'Time to find tracks {time.time()-th:.3f}, ----> {len(dc.tracks2D_list)} found so far')

        
        
def hough(center, points, theta_res, rho_res):

    """ center the points around (0,0)"""
    mean = np.mean(points, axis=0)
    
    x = points[:,0] -  mean[0]
    y = points[:, 1] - mean[1]
    x0, y0 = center - mean

    
    thetas = np.deg2rad(np.arange(-90.0, 90.0, theta_res))
    diag_len = int(np.ceil(np.hypot(np.ptp(x), np.ptp(y))))


    rhos = np.arange(-diag_len, diag_len + 1, rho_res)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    
    """ Matrix of x*cos(theta) + y*sin(theta) """
    xcos = np.outer(x, cos_t)
    ysin = np.outer(y, sin_t)
    rho_vals = xcos + ysin


    """ Digitize rho to index into rhos array """
    rho_bin_edges = np.arange(-diag_len - 0.5 * rho_res, diag_len + 1.5 * rho_res, rho_res)
    rho_idx = np.digitize(rho_vals, rho_bin_edges) - 1  # shift to 0-based index

    """ Clamp indices within accumulator range """
    valid = (rho_idx >= 0) & (rho_idx < len(rhos))
        
    """ Compute the accumulator """
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=int)
    for i in range(points.shape[0]):
        valid_thetas = valid[i]
        accumulator[rho_idx[i][valid_thetas], np.arange(len(thetas))[valid_thetas]] += 1


    
    """ original hit rho-theta """
    center_rho_vals = x0 * cos_t + y0 * sin_t
    center_rho_idx = np.digitize(center_rho_vals, rho_bin_edges) - 1

    valid = (center_rho_idx >= 0) & (center_rho_idx < len(rhos))
    center_rho_bins = center_rho_idx[valid]
    center_theta_bins = np.arange(len(thetas))[valid]

    scores = accumulator[center_rho_bins, center_theta_bins]
    top = np.argmax(scores)

    rho, theta =  rhos[center_rho_bins[top]], thetas[center_theta_bins[top]]


    if(np.sin(theta) == 0.):
        """ vertical line """
        return (), -1

    line_eq_a = -np.cos(theta)/np.sin(theta)
    line_eq_b =  rho/np.sin(theta)  - line_eq_a*mean[0] + mean[1]
    #print('BEST LINE IS ', line_eq_a, line_eq_b, ' WITH ', scores[top])

    return (line_eq_a, line_eq_b), scores[top]
