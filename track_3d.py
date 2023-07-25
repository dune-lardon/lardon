import config as cf
import data_containers as dc
import lar_param as lar

import numpy as np
import math
from scipy.interpolate import UnivariateSpline
from rtree import index
from operator import itemgetter


def theta_phi_from_deriv(dxdz, dydz):
    phi = math.degrees(math.atan2(-1.*dydz, -1.*dxdz))
    theta = math.degrees(math.atan2(math.sqrt(pow(dxdz,2)+pow(dydz,2)),-1.))

    return theta, phi


def finalize_3d_track(track, npts):
    view_used = [track.match_ID[i] >= 0 for i in range(cf.n_view)]
    nv = sum(view_used)
    zmin, zmax = track.ini_z_overlap, track.end_z_overlap

    """ Divide z-range in N slices, can be changed to 1cm slice in future """
    z_slices = np.linspace(zmin, zmax, npts)
    sx, sy = [], []

    theta_ini, theta_end, phi_ini, phi_end = [],[],[],[]


    for iv in range(cf.n_view):
        if(view_used[iv] == False): continue

        tx = [x[0] for x in track.path[iv]]
        ty = [x[1] for x in track.path[iv]]
        tz = [x[2] for x in track.path[iv]]


        """ sort by increasing z for the interpolation """
        z, x, y = (np.asarray(list(t)) for t in zip(*sorted(zip(tz, tx, ty))))

        """ interpolation wants unique z, remove duplicates (could be done better) """
        z_u, idx = np.unique(z, return_index=True)
        """ interpolation needs at least 3 points """
        if(len(z_u) < 4):
            if(nv > 2):
                nv -= 1
                continue
            else:
                return False
        x_u, y_u = x[idx], y[idx]

        """ make 2 2D splines, 3D splines doesn't work very well in our case """
        xz_spline = UnivariateSpline(z_u, x_u)
        xz_deriv = xz_spline.derivative()
        yz_spline = UnivariateSpline(z_u, y_u)
        yz_deriv = yz_spline.derivative()

        sx.append(xz_spline(z_slices))
        sy.append(yz_spline(z_slices))

        theta, phi = theta_phi_from_deriv(xz_deriv(zmin), yz_deriv(zmin))
        theta_ini.append(theta)
        phi_ini.append(phi)        

        theta, phi = theta_phi_from_deriv(xz_deriv(zmax), yz_deriv(zmax))
        theta_end.append(theta)
        phi_end.append(phi)
        

    sx = np.asarray(sx)
    sy = np.asarray(sy)

    d_sx = np.square(np.diff(sx, append=[sx[0]], axis=0))
    d_sy = np.square(np.diff(sy, append=[sy[0]], axis=0))


    d_slice = np.sqrt(d_sx+d_sy)
    dtot = np.sum(d_slice)/npts/nv


    m_theta_ini = sum(theta_ini)/nv
    m_theta_end = sum(theta_end)/nv
    m_phi_ini = sum(phi_ini)/nv
    m_phi_end = sum(phi_end)/nv

    track.set_angles(m_theta_ini, m_phi_ini, m_theta_end, m_phi_end)
    track.d_match = dtot
    
    return True

def linear_interp(dx, z0, a):
    return dx*a + z0

def complete_trajectories(tracks):
    """ Could be better ! At the moment, matches with only 2 tracks """
    n_trk = len(tracks)

    the_track = dc.trk3D()
    module_ini, module_end = -1, -1

    for i in range(n_trk):
        track = tracks[i]
        k = i+1
        if(k == n_trk): k=0
        other = tracks[k]
        
        if(module_ini < 0):
            module_ini = track.module_ini
        else:
            if(track.module_ini != module_ini):
                print('Matching problems, initial modules do not correspond')

        if(module_end < 0):
            module_end = track.module_end
        else:
            if(track.module_end != module_end):
                print('Matching problems, ending modules do not correspond')

        

        v_track = track.view
        ang_track = np.radians(cf.view_angle[v_track])

        v_other = other.view
        ang_other = np.radians(cf.view_angle[v_other])

        A = np.array([[-np.cos(ang_track), np.cos(ang_other)],
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

        trajectory          = []
        dQ, ds, t3d_hits_id = [], [], []
        length              = 0.

        """debug"""
        xp, yp, zp, pp= 0,0,0,0

        for p in range(len(track.path)):
            pos = track.path[p][0]
            z = track.path[p][1]

            if( p == 0 ):
                a0t = 0. if track.ini_slope == 0 else 1./track.ini_slope
            else:
                dp = track.path[p][0] - track.path[p-1][0]
                dz = track.path[p][1] - track.path[p-1][1]

                a0t = 0. if dz == 0 else dp/dz


            if(z >= z_o_min and z <= z_o_max):
                pos_spl = float(spline(z))
                a1t = float(deriv(z))

            elif(z < z_o_min):
                pos_spl = linear_interp(z-z_o_min, pos_o_min, deriv_z_min)
                a1t = deriv_z_min 

            elif(z > z_o_max):
                pos_spl = linear_interp(z-z_o_max, pos_o_max, deriv_z_max)
                a1t = deriv_z_max 



            xy = A.dot([pos_spl, pos])/D
            x, y = xy[0], xy[1]

            dxdy = A.dot([a1t, a0t])/D
            dxdz, dydz = dxdy[0], dxdy[1]


            a0 = 0. if dxdz == 0 else 1/dxdz
            a1 = 0. if dydz == 0 else 1/dydz


            ux =  -1.*np.sign(a0)/math.sqrt(1. + pow(a0, 2)*(1./pow(a1, 2) + 1.)) if a1!=0 else 0.
            uy =  -1.*np.sign(a1)/math.sqrt(1. + pow(a1, 2)*(1./pow(a0, 2) + 1.)) if a0 !=0 else 0.

            cosgamma = math.fabs(np.sin(ang_track-np.pi)*ux - np.cos(ang_track-np.pi)*uy)
                

            dr = cf.view_pitch[v_track]/cosgamma if cosgamma != 0 else 9999.

            if(v_track >2):
                print('----- at z=', z)
                print(v_track, " at ", pos, " with ", v_other, " at ", pos_spl)
                print("%.3f, %.3f"%(x, y))
                print('PITCH : %.2f'%dr)

            trajectory.append( (x,y,z) )
            dQ.append(track.dQ[p])
            ds.append(dr)
            t3d_hits_id.append(track.hits_ID[p])
        the_track.set_view(track, trajectory, dQ, ds, t3d_hits_id)

    the_track.set_modules(module_ini, module_end)
    return the_track



def correct_timing(trk, xtol, ytol, ztol):
    vdrift = lar.drift_velocity()
    z_top = cf.anode_z

    ''' maximum drift distance given the time window '''
    z_max = z_top - cf.n_sample*vdrift/cf.sampling
    z_cath = z_top - cf.drift_length

    from_top =  (z_top - trk.ini_z) < ztol
    exit_bot = (math.fabs(z_max - trk.end_z)) < ztol

    from_wall_x = np.asarray([ math.fabs(trk.ini_x-s)<t for t, s in zip(xtol,cf.x_boundaries[trk.module_ini])], dtype=bool)
    from_wall_y = np.asarray([ math.fabs(trk.ini_y-s)<t for t,s in zip(ytol,cf.y_boundaries[trk.module_ini])], dtype=bool)

    from_wall = np.any(np.concatenate((from_wall_x, from_wall_y), axis=None))

    exit_wall_x = np.asarray([ math.fabs(trk.end_x-s)<t for t, s in zip(xtol,cf.x_boundaries[trk.module_end])], dtype=bool)
    exit_wall_y = np.asarray([ math.fabs(trk.end_y-s)<t for t, s in zip(ytol, cf.y_boundaries[trk.module_end])], dtype=bool)
    exit_wall = np.any(np.concatenate((exit_wall_x, exit_wall_y), axis=None))

    z0 = 9999.
    t0 = 9999.

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







def find_tracks_rtree():

    ztol = dc.reco['track_3d']['ztol']
    qfrac= dc.reco['track_3d']['qfrac']
    len_min= dc.reco['track_3d']['len_min']
    dx_tol= dc.reco['track_3d']['dx_tol']
    dy_tol= dc.reco['track_3d']['dy_tol']
    dz_tol = dc.reco['track_3d']['dz_tol']
    
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

        if(t.len_straight >= len_min and t.ghost == False):
            rtree_idx.insert(t.trackID, (t.view, stop, t.view, start))
        i+=1
        idx_to_ID.append(t.trackID)


    ID_to_idx = [-1]*(max(idx_to_ID)+1)

    for idx, ID in enumerate(idx_to_ID):
        ID_to_idx[ID] = idx

    #print('id to index')
    #print(ID_to_idx)

    #print('index to ID')
    #print(idx_to_ID)
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
                if(ti.module_ini != tj.module_ini):
                    continue
                if(ti.module_end != tj.module_end):
                    continue

                #print("...overlaps with, ", j_ID, ' idx ', j_idx)
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
                if(balance < qfrac and dmin < ztol):
                #if(dmin < ztol):
                    matches.append( (j_ID, balance, dmin) )

            if(len(matches) > 0):
                ''' sort matches by balance ''' #distance '''
                matches = sorted(matches, key=itemgetter(1))
                ti.matched[tj.view] = matches[0][0]


    ''' now do the matching !'''

    for i_idx in range(len(dc.tracks2D_list)):
        ti = dc.tracks2D_list[i_idx]
        i_ID = idx_to_ID[i_idx]
        #print('-> ID ', i_ID, " index ", i_idx, " with ", ti.matched)
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


            n_fake = t3D.check_views()
            if(n_fake > 1):
                continue
            #print(t3D.path)
            t3D.boundaries()
            #print("\nFINALIZE TRACK")
            isok = finalize_3d_track(t3D, 10)
            if(isok == False):
                continue
            #t3D.angles(tv0, tv1)
            correct_timing(t3D, dx_tol, dy_tol, dz_tol)
            trk_ID = dc.evt_list[-1].n_tracks3D+1
            t3D.ID_3D = trk_ID

            dc.tracks3D_list.append(t3D)
            dc.evt_list[-1].n_tracks3D += 1
            #dc.tracks3D_list[-1].dump()
            for t in trks:
                for i in range(cf.n_view):
                    t.matched[i] = -1
                    t.match_3D = trk_ID
