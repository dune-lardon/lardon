import config as cf
import data_containers as dc
import lar_param as lar

import numpy as np
import math
from scipy.interpolate import UnivariateSpline
from rtree import index
from operator import itemgetter



def linear_interp(dx, z0, a):
    return dx*a + z0

def complete_trajectories(tracks):
    """ Could be better ! At the moment, matches with only 2 tracks """
    n_trk = len(tracks)
        
    the_track = dc.trk3D()

    for i in range(n_trk):
        track = tracks[i]
        k = i+1 
        if(k == n_trk): k=0
        other = tracks[k]

        """
        print("\n")
        track.mini_dump()
        print(" ... goes with ... ")
        other.mini_dump()
        """
        
        v_track = track.view
        ang_track = np.radians(cf.view_angle[v_track])

        v_other = other.view
        ang_other = np.radians(cf.view_angle[v_other])
        
        A = np.array([[np.cos(ang_track), -np.cos(ang_other)],
                      [-np.sin(ang_track), np.sin(ang_other)]])    

        

        D = A[0,0]*A[1,1]-A[0,1]*A[1,0]


        if(D == 0.):
            print("MEGA PBM :::  DETERMINANT IS ZERO")
            continue


        """ spline the other track """
        #reversed because spline wants an increasing x only
        pos_o = [k[0] for k in reversed(other.path)]
        z_o = [k[1] for k in reversed(other.path)]

        """ order lists according to z increasing """ 
        z_o, pos_o = (list(t) for t in zip(*sorted(zip(z_o, pos_o))))
        
        """ get the other track z range """
        pos_o_min, z_o_min = pos_o[0],  z_o[0]
        pos_o_max, z_o_max = pos_o[-1], z_o[-1]


        """ spline needs unique 'x' points to work --> remove duplicate """
        z_o_unique, idx = np.unique(z_o, return_index=True)
        pos_o = np.asarray(pos_o)
        pos_o_unique = pos_o[idx]


        """at least 3 points for the spline """
        if(len(z_o_unique) < 4):
            print('not enought point to spline the complete track')
            continue

        spline = UnivariateSpline(z_o_unique, pos_o_unique)
        deriv = spline.derivative() #gives dpos/dz

        deriv_z_min = float(deriv(z_o_min))
        deriv_z_max = float(deriv(z_o_max))

        a0, a1 = 0., 0.
        dx, dy, dz = 0., 0., 0.
        
        trajectory = []
        dQ, ds     = [], []
        length     = 0.

        for p in range(len(track.path)):
            pos = track.path[p][0]
            z = track.path[p][1]

            if( p == 0 ):
                a0 = 0. if track.ini_slope == 0 else 1./track.ini_slope
            else:
                dp = track.path[p][0] - track.path[p-1][0]
                dz = track.path[p][1] - track.path[p-1][1]
            
                a0 = 0. if dz == 0 else dp/dz
            

            if(z >= z_o_min and z <= z_o_max):
                pos_spl = float(spline(z))
                a1 = float(deriv(z))              
                #a1 = 0. if a1 == 0 else 1./a1
            elif(z < z_o_min):
                pos_spl = linear_interp(z-z_o_min, pos_o_min, deriv_z_min)
                a1 = deriv_z_min #0. if deriv_z_min == 0 else 1./deriv_z_min

            elif(z > z_o_max):
                pos_spl = linear_interp(z-z_o_max, pos_o_max, deriv_z_max)
                a1 = deriv_z_max #0. if deriv_z_max == 0 else 1./deriv_z_max


                
            xy = A.dot([pos_spl, pos])/D
            x, y = xy[0], xy[1]

            dxdy = A.dot([a1, a0])/D
            
            dxdz, dydz = dxdy[0], dxdy[1]


            a0 = 0. if dxdz == 0 else 1/dxdz
            a1 = 0. if dydz == 0 else 1/dydz

            dr = cf.view_pitch[v_track]


            """ WARNING THE DS COMPUTATION IS PROBABLY WRONG """
            if(a1 == 0):
                dr *= math.sqrt(1. + pow(a0,2))
            else : 
                dr *= math.sqrt(1. + pow(a0, 2)*(1./pow(a1, 2) + 1.))



            """ debugging """
            """
            if(p < 3):
                print(p, " at z ", z, " : ", pos, " with ", pos_spl)
                print("  -> x ", x, " y ", y)
                print("   a0 ", a0, " a1 ", a1, " -> ds ", dr)
            """

                
            trajectory.append( (x,y,z) )
            dQ.append(track.dQ[p])
            ds.append(dr)
        the_track.set_view(track, trajectory, dQ, ds)

            
    #print('\n')
    return the_track



def correct_timing(trk, tol):
    vdrift = lar.drift_velocity()
    z_top = cf.anode_z
    
    ''' maximum drift distance given the time window '''
    z_max = z_top - cf.n_sample*vdrift/cf.sampling
    z_cath = z_top - cf.drift_length

    from_top =  (z_top - trk.ini_z) < tol
    exit_bot = (math.fabs(z_max - trk.end_z)) < tol

    from_wall_x = np.asarray([ math.fabs(trk.ini_x-s)<tol for s in cf.x_boundaries], dtype=bool)
    from_wall_y = np.asarray([ math.fabs(trk.ini_y-s)<tol for s in cf.y_boundaries], dtype=bool)

    from_wall = np.any(np.concatenate((from_wall_x, from_wall_y), axis=None))

    exit_wall_x = np.asarray([ math.fabs(trk.end_x-s)<tol for s in cf.x_boundaries], dtype=bool)
    exit_wall_y = np.asarray([ math.fabs(trk.end_y-s)<tol for s in cf.y_boundaries], dtype=bool)
    exit_wall = np.any(np.concatenate((exit_wall_x, exit_wall_y), axis=None))

    z0 = 9999.
    t0 = 9999.

    """
    print("Start ", trk.ini_x, " ", trk.ini_y, " ", trk.ini_z)
    print("top ?", from_top, " wall ? ", from_wall)
    print(" End ",  trk.end_x, " ", trk.end_y, " ", trk.end_z)
    print("bot ?", exit_bot, " wall ? ", exit_wall)
    """

    """ unknown case is when track enters through wall """
    if(from_wall):
        trk.set_t0_z0(t0, z0)
        return

    #early track case 
    if(from_top):
        #print('early case!')
        if(exit_wall == False):
            #then it exits through the cathode
            #print('exits through cathode')
            z0 = (z_cath - trk.end_z)
            if(z0 > 0.): z0 *= -1.
            t0 = z0/vdrift
            trk.set_t0_z0(t0, z0)
            return
        else:
            #exits through the wall, we don't know then
            trk.set_t0_z0(t0, z0)
            #print('but exits through wall')
            return
    

    #print('is a late track!')
    #later track case
    z0 = (z_top-trk.ini_z)
    t0 = z0/vdrift
    trk.set_t0_z0(t0, z0)
    return







def find_tracks_rtree(ztol, qfrac, len_min, d_tol):
    if(len(dc.tracks2D_list) < 2):
        return

    pties = index.Property()
    pties.dimension = 2

    ''' create an rtree index (3D : view, z)'''
    rtree_idx = index.Index(properties=pties)


    ''' as the 2D track list got sorted, list index and track ID do not match anymore '''
    idx_to_ID = []

    i = 0
    ''' fill the index '''
    ''' track start is at the top of the detector, hence start > stop '''

    for t in dc.tracks2D_list:

        start = t.path[0][1]
        stop  = t.path[-1][1]

        if(t.len_straight >= len_min):        
            rtree_idx.insert(t.trackID, (t.view, stop, t.view, start))
        i+=1
        idx_to_ID.append(t.trackID)

        
    ID_to_idx = [-1]*(max(idx_to_ID)+1)
    
    for idx, ID in enumerate(idx_to_ID):
        ID_to_idx[ID] = idx


    ''' search for the best matching track in the other view '''

    for ti in dc.tracks2D_list:
        if(ti.len_straight < len_min):
            continue

        ti_start = ti.path[0][1]
        ti_stop  = ti.path[-1][1]

        overlaps = []
        for iview in range(cf.n_view):
            if(iview == ti.view):
                continue
            else:
                overlaps.append(list(rtree_idx.intersection((iview, ti_stop, iview, ti_start))))
        
        #print("\nNEW TRACK ")
        #ti.mini_dump()
        #print(overlaps)


        for ov in overlaps: 
            #print(ov)
            matches = []
            for j_ID in ov:
                j_idx = ID_to_idx[j_ID]
                tj = dc.tracks2D_list[j_idx]
                #print("overlaps with, ", j_ID, ' idx ', j_idx)
                #tj.mini_dump()

                tj_start = tj.path[0][1]
                tj_stop  = tj.path[-1][1]
            
                zmin = max(ti_stop, tj_stop)
                zmax = min(ti_start, tj_start)
                qi = np.fabs(ti.charge_in_z_interval(zmin, zmax))
                qj = np.fabs(tj.charge_in_z_interval(zmin, zmax))

                try:
                    balance = math.fabs(qi - qj)/(qi + qj)
                except ZeroDivisionError:
                    balance = 9999.
                dmin = min(math.fabs(ti_start- tj_start), math.fabs(ti_stop - tj_stop))

                #print("balance : ", balance)
                #print("dmin : ", dmin)
                #print(" ID ", j_ID)
                #if(balance < qfrac and dmin < ztol):
                if(dmin < ztol):
                    matches.append( (j_ID, balance, dmin) )

            if(len(matches) > 0):
                ''' sort matches by balance '''        
                matches = sorted(matches, key=itemgetter(2))
                ti.matched[tj.view] = matches[0][0]
    

    ''' now do the matching !'''

    for i_idx in range(len(dc.tracks2D_list)):
        ti = dc.tracks2D_list[i_idx]
        i_ID = idx_to_ID[i_idx]

        #print('track ', i_idx, ' matches with ', ti.matched)
        trks = [ti]
        for iview in range(ti.view+1, cf.n_view):
            j_ID = ti.matched[iview]
            #print('  -> track in view ', iview, ' has ID ', j_ID)
            if(j_ID>0):
                j_idx = ID_to_idx[j_ID]
                tj = dc.tracks2D_list[j_idx]
                if(tj.matched[ti.view] == i_ID):
                    trks.append(tj)
        if(len(trks) > 1):
            #print("it's a match <3")
            #print("idx : ", i_idx, ' with ', len(trks))
            t3D = complete_trajectories(trks)           
            
            t3D.check_views()
            #print(t3D.path)
            t3D.boundaries()
            #t3D.angles(tv0, tv1)
            correct_timing(t3D, d_tol)
            dc.tracks3D_list.append(t3D)
            dc.evt_list[-1].n_tracks3D += 1
            dc.tracks3D_list[-1].dump()
