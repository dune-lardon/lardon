import lardon.config as cf
import lardon.data_containers as dc
import lardon.lar_param as lar
import lardon.hits_3d as h3d
import lardon.stitch_tracks as stitch


import numpy as np
from scipy.interpolate import UnivariateSpline

from rtree import index
from operator import itemgetter

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import DBSCAN

from itertools import product

from collections import Counter
import time as time





def dist_points(xa, ya, xb, yb):
    return np.sqrt(pow(xa-xb,2)+pow(ya-yb,2))

def dist_to_boundaries(module, x,y):
    module_xlow, module_xhigh = cf.x_boundaries[module][0], cf.x_boundaries[module][1] 
    module_ylow, module_yhigh = cf.y_boundaries[module][0], cf.y_boundaries[module][1]

    x_clamp = min(max(x, module_xlow), module_xhigh)
    y_clamp = min(max(y, module_ylow), module_yhigh)
    dx = x - x_clamp
    dy = y - y_clamp
    return np.sqrt(dx*dx + dy*dy)
                
def is_inside_volume(module, x,y):
    module_xlow, module_xhigh = cf.x_boundaries[module][0], cf.x_boundaries[module][1] 

    module_ylow, module_yhigh = cf.y_boundaries[module][0], cf.y_boundaries[module][1]

    
    if(x < module_xlow or x > module_xhigh):
        return False
    if(y< module_ylow or y > module_yhigh):    
        return False

    return True
    
def theta_phi_from_deriv(dxdz, dydz):
    if(cf.tpc_orientation == 'Vertical'):
        phi = np.degrees(np.arctan2(-1.*dydz, -1.*dxdz))
        theta = np.degrees(np.arctan2(np.sqrt(pow(dxdz,2)+pow(dydz,2)),-1.))

    else:
        if(dxdz == 0 and dydz !=0):
            theta, phi = 90, 0
        else:
            dzdx = 1./dxdz
            dydx = dydz/dxdz
            theta = np.degrees(np.arctan2(np.sqrt(pow(dzdx,2)+pow(dydx,2)),-1.))
            phi   = np.degrees(np.arctan2(-1.*dzdx, -1.*dydx))
    return theta, phi


def finalize_3d_track(track):

    n_slices = dc.reco['track_3d']['goodness']['n_slices']
    d_slice_max = dc.reco['track_3d']['goodness']['d_slice_max']

    sum_match = [sum([track.match_ID[k][i] for k in range(cf.n_module)]) for i in range(cf.n_view)]

    view_used = [sum_match[i] >-1*cf.n_module for i in range(cf.n_view) ]

    
    nv = sum(view_used)
    zmin, zmax = track.ini_z_overlap, track.end_z_overlap

    """ Divide z-range in N slices, can be changed to 1cm slice in future """
    z_slices = np.linspace(zmin, zmax, n_slices)
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
                print('3D finalizing track: problem with interpolation')
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
    dtot = np.sum(d_slice)/n_slices/nv

    
    m_theta_ini = sum(theta_ini)/nv
    m_theta_end = sum(theta_end)/nv
    m_phi_ini = sum(phi_ini)/nv
    m_phi_end = sum(phi_end)/nv

    track.set_angles(m_theta_ini, m_phi_ini, m_theta_end, m_phi_end)
    #track.set_timestamp()
    track.d_match = dtot

    if(dtot > d_slice_max):
        #print('nooooo bad bad track: ', dtot)
        return False
    
    return True

def linear_interp(dx, z0, a):
    return dx*a + z0


def get_channel_from_pos(pos,view,module):
    shift = cf.view_offset_repet[module][view][0] +  cf.view_pitch[view]/2.
    pitch = cf.view_pitch[view]
    
    channel = int(pos/pitch)
    return channel

def get_channel_from_xy(x, y, view, module):
    angle = np.radians(cf.view_angle[module][v_track])    
    pos = np.sin(angle)*x - np.cos(angle)*y - shift

    return get_channel_from_pos(pos, view, module)

def complete_trajectories(tracks):
    """ Could be better ! At the moment, matches with only 2 tracks """
    n_trk = len(tracks)

    the_track = dc.trk3D()
    module_ini, module_end = -1, -1

    debug = False
    if(debug):
        print('building 3D track from ', [t.trackID for t in tracks])

        for iv in range(3):
            if(n_trk == 3):              
                print(f'\n v{iv} = ')
                print(tracks[iv].path)
        print('-------------------------')
        
    for i in range(n_trk):
        if(debug):
            print('\n\ncomputing view ',i)
        track = tracks[i]
        k = i+1
        if(k == n_trk): k=0
        other = tracks[k]



        module_ini = track.module_ini
        module_end = track.module_end

        

        v_track = track.view
        ang_track = np.radians(cf.view_angle[module_ini][v_track])

        v_other = other.view
        ang_other = np.radians(cf.view_angle[module_ini][v_other])

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


        xp, yp, zp, pp= 0,0,0,0
        dr_p = cf.view_pitch[v_track]
        hits_ID_shift = dc.hits_list[0].ID

        
        for p in range(len(track.path)):

            pos = track.path[p][0]
            z = track.path[p][1]

            hid = track.hits_ID[p]
            hit_module = dc.hits_list[hid-hits_ID_shift].module
            if(debug):
                print(p, '/',len(track.path),':', pos, z)
                dc.hits_list[hid-hits_ID_shift].mini_dump()
            
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

            if(debug):
                print('with splined ', pos_spl)
            
            xy = A.dot([pos_spl, pos])/D
            x, y = xy[0], xy[1]
                                          
            if(debug):
                print("---->3D AT ", x, y, 'inside? ', is_inside_volume(hit_module, x,y))
                
            if(dc.evt_list[-1].det == 'pdhd'):
                if(is_inside_volume(hit_module, x,y)==False):
                    if(debug):print('oh no')
                    db = dist_to_boundaries(hit_module, x,y)
                    xb, yb = x, y
                    unwrapped = False
                    for d0 in [0] if v_track == 2 else [0, cf.unwrappers[hit_module][v_track]]:
                        if(unwrapped): break
                        for d1 in [0] if v_other == 2 else [0,  cf.unwrappers[hit_module][v_other]]:
                            xy = A.dot([pos_spl+d1, pos+d0])/D
                            xt, yt = xy[0], xy[1]
                   

                            if(debug):#len(dc.tracks3D_list)==0 and v_track==0): #False):
                                print(d0,d1,"::",pos_spl+d1, pos+d0, "-->", xt, yt, is_inside_volume(hit_module, xt,yt), d_prev, dist_to_boundaries(hit_module, xt,yt))
                                
                            if(is_inside_volume(hit_module, xt,yt)==True):
                                x, y = xt, yt
                                unwrapped = True
                                break
                            else:
                                d = dist_to_boundaries(hit_module, xt,yt)
                                if(d < db):
                                    xb, yb = x, y
                                    db = d
                    if(unwrapped == False):
                        x,y=xb,yb


            dxdy = A.dot([a1t, a0t])/D
            dxdz, dydz = dxdy[0], dxdy[1]


            a0 = 0. if np.fabs(dxdz) < 1e-6 else 1/dxdz
            a1 = 0. if np.fabs(dydz) < 1e-6 else 1/dydz


            ux =  -1.*np.sign(a0)/np.sqrt(1. + pow(a0, 2)*(1./pow(a1, 2) + 1.)) if a1!=0 else 0.
            uy =  -1.*np.sign(a1)/np.sqrt(1. + pow(a1, 2)*(1./pow(a0, 2) + 1.)) if a0 !=0 else 0.

            cosgamma = np.fabs(np.sin(ang_track-np.pi)*ux - np.cos(ang_track-np.pi)*uy)
                
            
            dr = cf.view_pitch[v_track]/cosgamma if cosgamma != 0 else np.sqrt(pow(x-xp,2)+pow(y-yp,2)+pow(z-zp,2))
            if(dr < cf.view_pitch[v_track]):
                dr = dr_p
                
            xp, yp, zp = x, y, z
            dr_p = dr
            trajectory.append( (x,y,z) )
            dQ.append(track.dQ[p])
            ds.append(dr)
            t3d_hits_id.append(track.hits_ID[p])
        the_track.set_view(track, trajectory, dQ, ds, t3d_hits_id)

    the_track.set_modules(module_ini, module_end)
    return the_track

def compute_exit_point(trk, idx_anode, zcorr):
    debug = False
    
    box_min = [min([cf.x_boundaries[i][0] for i in range(cf.n_module)]),min([cf.y_boundaries[i][0] for i in range(cf.n_module)]), min(cf.anode_z)]
    box_max = [max([cf.x_boundaries[i][1] for i in range(cf.n_module)]),max([cf.y_boundaries[i][1] for i in range(cf.n_module)]), max(cf.anode_z)]
    
    if(idx_anode == 0):
        theta = trk.ini_theta
        phi   = trk.ini_phi
        point = [trk.ini_x, trk.ini_y, trk.ini_z+zcorr]
        sign = -1.*cf.drift_direction[trk.module_ini]*np.sign(phi)
    else:
        theta = trk.end_theta
        phi   = trk.end_phi
        point = [trk.end_x, trk.end_y, trk.end_z+zcorr]
        sign = -1.*cf.drift_direction[trk.module_end]*np.sign(phi)


    if(dc.evt_list[-1].det == 'pdhd'):
        dx = sign*np.cos(np.radians(theta))
        dy = sign*np.sin(np.radians(theta))*np.cos(np.radians(phi))
        dz = sign*np.sin(np.radians(theta))*np.sin(np.radians(phi))



    elif(dc.evt_list[-1].det == 'pdvd'):
        sign = cf.drift_direction[trk.module_ini] if idx_anode==0 else cf.drift_direction[trk.module_end]

        if(debug):
            print('\n = = = = = ')
            trk.dump()

            print('test sign: ', cf.drift_direction[trk.module_end], 'and ', np.sign(phi), ' -->', sign)
            print('box min', box_min)
            print('box max', box_max)
            print('zcorr :: ', zcorr)
            print('idx anode: ', idx_anode)
        
        
        dz = sign*np.cos(np.radians(theta))
        dx = sign*np.sin(np.radians(theta))*np.cos(np.radians(phi))
        dy = sign*np.sin(np.radians(theta))*np.sin(np.radians(phi))

        if(debug):
            print('--> ', dx, dy, dz)

    else:        
        print('Exit point computation to be checked for ', dc.evt_list[-1].det,' geometry')
        return False, []

    direction = np.array([dx, dy, dz])
    origin = np.array(point)

    #print('direction: ', direction)
    #print('origin :', origin, ' point start ', idx_anode)
    inv_dir = 1.0 / direction  # Inverse to avoid dividing multiple times

    # Compute tmin and tmax for slabs
    tmin = (box_min - origin) * inv_dir
    tmax = (box_max - origin) * inv_dir

    # Swap if needed to ensure correct ordering
    t1 = np.minimum(tmin, tmax)
    t2 = np.maximum(tmin, tmax)

    # We're inside the box, so t_enter < 0 and we want the smallest positive t_exit
    t_exit = np.min(t2)

    if(debug):
        print('\ntesting mins')
        for t in tmin:
            print(t, ' test out ', origin + t * direction)
        print('testing maxs')
        for t in tmax:
            print(t, ' test out ', origin + t * direction)
        print('---> t exit : ', t_exit)
    
    
    if t_exit <= 0:
        return False, []  # Line doesn't exit in the direction given

    exit_point = origin + t_exit * direction
    if(debug):
        print('====>>>>> exit point ', exit_point)
    return True, exit_point    
    
def correct_timing(trk, xtol, ytol, ztol):
    debug = False

    """ track points are ordered by decreasing vertical axis """
    vdrift = lar.drift_velocity()
    
    z0 = 9999.
    t0 = 9999.


    mod_ini, mod_end = trk.module_ini, trk.module_end
    
    z_anodes = [cf.anode_z[mod_ini], cf.anode_z[mod_end]]

    drift_times = [ cf.drift_length[mod_ini]/vdrift, cf.drift_length[mod_end]/vdrift]
    
    """ longest drifts """
    max_drifts = [cf.anode_z[mod_ini] - cf.n_sample[mod_ini]*cf.drift_direction[mod_ini] * vdrift /cf.sampling[mod_ini], cf.anode_z[mod_ini] - cf.n_sample[mod_ini]*cf.drift_direction[mod_ini] * vdrift /cf.sampling[mod_ini]]
    
    z_cathodes = [cf.anode_z[mod_ini] - cf.drift_direction[mod_ini]*cf.drift_length[mod_ini], cf.anode_z[mod_end] - cf.drift_direction[mod_end]*cf.drift_length[mod_end]]
    
    trk_z_bounds = [trk.ini_z, trk.end_z]
    trk_modules  = [mod_ini, mod_end]
    trk_times    = [trk.ini_time/cf.sampling[mod_ini], trk.end_time/cf.sampling[mod_end]]
    
    through_anode = np.asarray([np.fabs(t-z) < ztol for t,z in zip(trk_z_bounds, z_anodes)], dtype=bool)
    through_cathode = np.asarray([np.fabs(t-z) < ztol for t,z in zip(trk_z_bounds, max_drifts)], dtype = bool)

    inside_ini_x = cf.x_boundaries[mod_ini][0]+xtol[mod_ini][0] <= trk.ini_x <= cf.x_boundaries[mod_ini][1]-xtol[mod_ini][1]
    inside_ini_y = cf.y_boundaries[mod_ini][0]+ytol[mod_ini][0] <= trk.ini_y <= cf.y_boundaries[mod_ini][1]-ytol[mod_ini][1]
    through_wall_ini = np.any([not inside_ini_x, not inside_ini_y])



    inside_end_x = cf.x_boundaries[mod_end][0]+xtol[mod_end][0] <= trk.end_x <= cf.x_boundaries[mod_end][1]-xtol[mod_end][1]
    inside_end_y = cf.y_boundaries[mod_end][0]+ytol[mod_end][0] <= trk.end_y <= cf.y_boundaries[mod_end][1]-ytol[mod_end][1]
    through_wall_end = np.any([not inside_end_x, not inside_end_y])
    

    if(debug):
        trk.dump()
    
        print('-> through anode ', through_anode, '::', z_anodes)
        print('-> through cathode', through_cathode, "::", max_drifts)
        print('-- ini inside x?', inside_ini_x, ' y?', inside_ini_y)        
        print('-> through wall ini ', through_wall_ini, "::", cf.x_boundaries[mod_ini], " and ", cf.y_boundaries[mod_ini] )
        print('-- end inside x?', inside_end_x, ' y?', inside_end_y)
        print('-> through wall end ', through_wall_end, "::", cf.x_boundaries[mod_end], " and ", cf.y_boundaries[mod_end] )
        print('drift times: ', drift_times)
        print('trk_times: ', trk_times)

    
    if(through_wall_ini and through_wall_end):
        """ wall to wall track: we cannot tell """
        trk.set_t0_z0(t0, z0)
        
        """ compute time range of possibilities """
        """ idx is the track endpoint closest to anode """
        idx, idx_o = (0,1) if np.argmin([np.fabs(t-z) for t,z in zip(trk_z_bounds, z_anodes)]) == 0 else (1,0)
        tref_a = trk_times[idx] # latest possible time
        tref_b = trk_times[idx_o]-drift_times[idx_o] #earliest possible time
        if(tref_a > tref_b):
            tref_a, tref_b = tref_b, tref_a
        trk.set_timestamp(tref_a, tref_b)

        if(debug):
            print('wall to wall ... no t0!')
            print('IDX closest to anode: ', idx)
            print('possible tref from ', tref_a, 'to', tref_b)
            print('Ts= ', trk.timestamp, ' to ', trk.timestamp_r)
            
        return
    
    elif(through_wall_ini or through_wall_end):
        ''' idx is the track endpoint at fc '''
        idx, idx_o = (0, 1) if through_wall_ini else (1, 0)
        if(through_anode[idx_o] or through_cathode[idx_o]):
            """ wall to readout : we cannot tell """
            trk.set_t0_z0(t0, z0)
            
            """ compute time range of possibilities """
            """ now idx is the track endpoint closest to anode """
            idx, idx_o = (0,1) if np.argmin([np.fabs(t-z) for t,z in zip(trk_z_bounds, z_anodes)]) == 0 else (1,0)
            tref_a = trk_times[idx] # latest possible time
            tref_b = trk_times[idx_o]-drift_times[idx_o] #earliest possible time
            if(tref_a > tref_b):
                tref_a, tref_b = tref_b, tref_a
            trk.set_timestamp(tref_a, tref_b)

            if(debug):
                print('wall to anode/cathode ... no t0!')
                print('IDX closest to anode', idx)
                print('possible tref from ', tref_a, 'to', tref_b)
                print('Ts= ', trk.timestamp, ' to ', trk.timestamp_r)
            return
        else:
            zdir = 1 if (trk_z_bounds[idx_o]-trk_z_bounds[idx])>0 else -1
            
            
            is_late = False
            if(zdir == cf.drift_direction[trk_modules[idx_o]]):
                zref = z_anodes[idx_o]
                tref = trk_times[idx_o]
                
                is_late = True
                if(debug): print('wall to anode dir', zdir, 'idxs',idx, idx_o)
                trk.is_anode_crosser = True
                
                ok, exit_point = compute_exit_point(trk, idx_o, zref-trk_z_bounds[idx_o])
                if(debug) :
                    print('EXIT POINT IS ', exit_point)
                if(ok):
                    trk.set_anode_crosser(exit_point, idx)
                    
            else:
                zref = z_cathodes[idx_o]
                tref = trk_times[idx_o] - drift_times[idx_o]
                trk.is_cathode_crosser = True
                if(debug): print('wall to cathode dir', zdir,'idxs',idx,idx_o)
                is_late = True #


            z0 = zref - trk_z_bounds[idx_o]
            t0 = tref#z0/vdrift
            """
            if(is_late == False and t0 > 0):
                t0 *= -1
            if(is_late == True and t0 < 0):
                t0 *= -1
            """
            
            trk.set_t0_z0(t0, z0)
            trk.set_timestamp(tref, tref)
            if(debug):
                print('--> z0 = ', z0, 't0',t0)
                print('time ref is ', tref)
                print('Ts= ', trk.timestamp, ' to ', trk.timestamp_r)
            return



    """ if we're here : the track did not enter nor escaped by the FC """
    
    if(np.any(through_anode)):
        """ then track is early """
        """ which bound is closer to anode ?"""
        """ idx is the trk endpoint closest to the anode"""
        idx = np.argmin([np.fabs(t-z) for t,z in zip(trk_z_bounds, z_anodes)])
        idx_o = 1 if idx == 0 else 0
        z0 = z_cathodes[idx_o] - trk_z_bounds[idx_o]

        tref = trk_times[idx_o]-drift_times[idx_o]
        
        if(np.sign(z0) == cf.drift_direction[trk_modules[idx_o]]):
            z0 *= -1
        t0 = tref#z0/vdrift
        
        #if(t0 > 0):
        #    t0 *= -1
            
        trk.set_t0_z0(t0, z0)
        trk.set_timestamp(tref, tref)
        trk.is_cathode_crosser = True
        
        if(debug):
            print('early track/cathode crosser! bound ',idx,'closer to anode, z0=', z0, "t0",t0)
            print('tref: ', tref)
            print('Ts= ', trk.timestamp, ' to ', trk.timestamp_r)
        return
    
    if(np.any(through_cathode)):
        """ then track is late """
        """ which bound is closer to cathode?"""
        """ idx is the trk endpoint closer to the cathode """
        idx = np.argmin([np.fabs(t-z) for t,z in zip(trk_z_bounds, max_drifts)])
        idx_o = 1 if idx == 0 else 0
        z0 = z_anodes[idx_o] - trk_z_bounds[idx_o]
        tref = trk_times[idx_o]

        t0 = tref#z0/vdrift
        #if(t0 < 0):
        #    t0 *= -1
        trk.set_t0_z0(t0, z0)
        trk.set_timestamp(tref, tref)
        trk.is_anode_crosser = True

        
        if(debug):
            print('late track/anode crosser! bound ',idx,'closer to cathode, z0=', z0,"t0",t0)
            print('tref ', tref)
            print('Ts= ', trk.timestamp, ' to ', trk.timestamp_r)
        
        return

    
    """ what's left is a late track entering from the higher plane """
    
    highest_plane_idx = np.argmax([z_anodes[0], z_cathodes[0]])
    highest_plane = z_anodes if highest_plane_idx == 0 else z_cathodes

    """ idx is the track endpoint closer to the highest plane """    
    idx = np.argmin([np.fabs(t-z) for t,z in zip(trk_z_bounds, highest_plane)])
    z0 = highest_plane[idx] - trk_z_bounds[idx]
    tref = trk_times[idx] - highest_plane_idx*drift_times[idx]
    
    t0 = tref
    # = z0/vdrift
    #if(t0 < 0):
    #    t0 *= -1
        
    trk.set_t0_z0(t0, z0)
    trk.set_timestamp(tref, tref)
    trk.is_anode_crosser = True
    if(debug):
        print('no choice, late track!, z0=', z0, 't0',t0)
        print('highest plane id is ', highest_plane_idx, ' at ', highest_plane)
        print('tref ', tref)
        print('Ts= ', trk.timestamp, ' to ', trk.timestamp_r)

        #print('OR cathode crosser: ')
        #lll
    return

    



def check_track_3D(track):

    eps = dc.reco['track_3d']['goodness']['eps']
    min_samp = dc.reco['track_3d']['goodness']['min_samp']
    n_min = dc.reco['track_3d']['goodness']['n_min']



    sum_match = [sum([track.match_ID[k][i] for k in range(cf.n_module)]) for i in range(cf.n_view)]
    

    n_mod = cf.n_module 
    data = [list(p) for iv, pview in enumerate(track.path)  for p in pview if sum_match[iv]>-1*n_mod]

                                                                                              
    X = np.asarray(data)
    db = DBSCAN(eps=eps,min_samples=min_samp).fit(X)
    labels = db.labels_

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if(n_clusters == 1):
        return True

    count = Counter(labels)    

    hits_ID = [h for iv, hview in enumerate(track.hits_ID) for h in hview if sum_match[iv]>-1*n_mod]
    
    n_hits = len(hits_ID)
    
    
    n_above = sum([1 if nb>n_hits/4 else 0 for nb in count.values()])

    
    if(n_above > 2):
        #print('NOOOOOOOOOOTTTTTT GOOOOOOOOOOOOOOOOD :: ', n_clusters, ' n above ', n_above)
        return False
    
    hits_to_rm = [h for h,l in zip(hits_ID, labels) if count[l] < n_min]
    
    [track.remove_hit(h, dc.hits_list[h-dc.n_tot_hits].view, dc.hits_list[h-dc.n_tot_hits].module) for h in hits_to_rm]
    
            
    return True

                        
def track_in_module(t, modules):
    if(t.module_ini in modules or t.module_end in modules):
        return True
    else:
        return False



def find_3D_tracks_with_missing_view(modules):
    
    trk_ztol = dc.reco['track_3d']['trk_ztol']
    hit_ztol = dc.reco['track_3d']['hit_ztol']
    #qfrac= dc.reco['track_3d']['qfrac']
    len_min= dc.reco['track_3d']['len_min']
    dx_tol= dc.reco['track_3d']['timing']['dx_tol']
    dy_tol= dc.reco['track_3d']['timing']['dy_tol']
    dz_tol = dc.reco['track_3d']['timing']['dz_tol']

    
    #d_thresh = dc.reco['track_3d']['missing_view']['d_thresh']
    min_z_overlap = dc.reco['track_3d']['missing_view']['min_z_overlap']
    trk_min_dz = dc.reco['track_3d']['missing_view']['trk_min_dz']
    q_thr = dc.reco['track_3d']['missing_view']['q_thr']
    r_z_thr = dc.reco['track_3d']['missing_view']['r_z_thr']
    
    if(dc.evt_list[-1].det != 'pdhd'):
        unwrappers = [[0,0,0]]
    else:
        unwrappers = [[0,0,0], [cf.unwrappers[modules[0]][0], 0, 0], [0, cf.unwrappers[modules[0]][1], 0], [cf.unwrappers[modules[0]][0], cf.unwrappers[modules[0]][1], 0]]
    
    if(len(dc.tracks2D_list) < 2):
        return

    """ R-tree with the tracks """
    trk_pties = index.Property()
    trk_pties.dimension = 3

    """ R-tree along (module, view, endpoints) """
    rtree_trk_idx = index.Index(properties=trk_pties)

    trk_ID_shift = dc.n_tot_trk2d

    """ list good 2D tracks not matched """
    tracks = [x for x in dc.tracks2D_list if x.len_straight >= len_min and x.dz > trk_min_dz and x.ghost == False and track_in_module(x, modules) == True and x.match_3D < 0]

    ntracks = len(tracks)
    ntracks_all = len(dc.tracks2D_list)


    if(ntracks < 2):
        return
    
    for t in tracks:
        start = t.path[0][1]
        stop  = t.path[-1][1]

        module_start = min(t.module_ini, t.module_end)
        module_stop =  max(t.module_ini, t.module_end)

        rtree_trk_idx.insert(t.trackID, (module_start, t.view, stop, module_stop, t.view, start))

    """ R-tree with the hits associated """
    hit_pties = index.Property()
    hit_pties.dimension = 4

    ''' create an rtree index of hits in a 2D track (module, view, trackID, z)'''
    rtree_hit_idx = index.Index(properties=hit_pties)
    
    hits_ID_shift = dc.hits_list[0].ID

    hits = [x for x in dc.hits_list if x.match_2D >=0 and dc.tracks2D_list[x.match_2D-trk_ID_shift].len_straight>=len_min and dc.tracks2D_list[x.match_2D-trk_ID_shift].dz > trk_min_dz and dc.tracks2D_list[x.match_2D-trk_ID_shift].ghost==False and x.module in modules and x.match_3D < 0]
    nhits = len(hits)
    #print('nb of hits un-matched in 3D ', nhits)
    
    ''' fill a R-tree with the hits associated to a 2D track and not to a 3D track'''
    for h in hits: 
        stop = min(h.Z_stop, h.Z_start)
        start = max(h.Z_stop, h.Z_start)                
        module = h.module
        rtree_hit_idx.insert(h.ID, (h.module, h.view, h.match_2D, stop, h.module, h.view, h.match_2D, start))


    tested = set()
    ok_combo = []
    for t in tracks:

        t_start = t.path[0][1] + trk_ztol
        t_stop  = t.path[-1][1] - trk_ztol
        module_start = min(t.module_ini, t.module_end)
        module_stop =  max(t.module_ini, t.module_end)
        
        """ get all tracks in the other views compatible in time """
        trk_overlaps = [[] for x in range(cf.n_view)]

        for iview in range(cf.n_view):
            if(iview == t.view):
                trk_overlaps[iview].append(t.trackID)
            else:
                [trk_overlaps[iview].append(k) for k in list(rtree_trk_idx.intersection((module_start, iview, t_stop, module_stop, iview, t_start)))]
                
                
        
        trk_overlaps =  [[None] if not x else x for x in trk_overlaps]

        """ get all 3 tracks combinations """
        trk_comb = list(product(*trk_overlaps))

        
        for tc in trk_comb:
            """ don't test a combination already tested """
            if(tc in tested):
                continue
            else:
                tested.add(tc)



            #get_channel_from_xy(x, y, view, module):

            trks = [dc.tracks2D_list[i-trk_ID_shift] for i in tc if i!=None]
            if(len(trks) == 1):
                continue
            starts = [(it.path[0][1],it.hits_ID[0]) for it in trks]
            starts = sorted(starts, key=itemgetter(0))
            
            stops  = [(it.path[-1][1],it.hits_ID[-1], ) for it in trks]
            stops = sorted(stops, key=itemgetter(0))

            
            """ get the smallest z overlap in time from the combination """
            hit_min_start = dc.hits_list[starts[0][1]-hits_ID_shift]
            hit_max_stop = dc.hits_list[stops[-1][1]-hits_ID_shift]            

            """ get the smallest z overlap in time from the combination """
            hit_max_start = dc.hits_list[starts[-1][1]-hits_ID_shift]
            hit_min_stop = dc.hits_list[stops[0][1]-hits_ID_shift]            

            
            if(hit_max_stop.Z > hit_min_start.Z):
                continue
            
            dz_ov = np.fabs(hit_min_start.Z-hit_max_stop.Z)

            if(dz_ov < min_z_overlap):
                continue
            dz_long = np.fabs(hit_max_start.Z-hit_min_stop.Z)

            if(dz_ov/dz_long < r_z_thr):
                continue
            
            qtrk = [np.fabs(it.charge_in_z_interval(hit_max_stop.Z, hit_min_start.Z)) for it in trks]
            
            qtrk_tot=sum(qtrk)
            if(qtrk_tot == 0 or np.any(qtrk==0)):
                continue

            qtrk = [(q/qtrk_tot)<q_thr for q in qtrk]

            if(np.any(qtrk)):
                continue

            ok_combo.append(tc)
            
    for tc in ok_combo:
        flat_tracks = [dc.tracks2D_list[x-trk_ID_shift] for x in tc if x!=None]
        t3D = complete_trajectories(flat_tracks)

        isok = check_track_3D(t3D)        
        if(isok == False):            
            continue


        n_fake = t3D.check_views()

        if(n_fake > 1):
            continue
        
        t3D.boundaries()

        isok = finalize_3d_track(t3D)
        if(isok == False):
            #print('oh no!')
            continue


        

        trk_ID = dc.evt_list[-1].n_tracks3D + dc.n_tot_trk3d #+1
        t3D.ID_3D = trk_ID

        correct_timing(t3D, dx_tol, dy_tol, dz_tol)

        dc.tracks3D_list.append(t3D)
        dc.evt_list[-1].n_tracks3D += 1

        
            
        for t in flat_tracks:
            t.match_3D = trk_ID
            t.set_match_hits_3D(trk_ID)

        
def find_track_3D_rtree_new(modules, debug=False):
    ta = time.time()
    
    trk_ztol = dc.reco['track_3d']['trk_ztol']
    hit_ztol = dc.reco['track_3d']['hit_ztol']

    #qfrac= dc.reco['track_3d']['qfrac']
    len_min= dc.reco['track_3d']['len_min']

    dx_tol= dc.reco['track_3d']['timing']['dx_tol']
    dy_tol= dc.reco['track_3d']['timing']['dy_tol']
    dz_tol = dc.reco['track_3d']['timing']['dz_tol']

    d_thresh = dc.reco['track_3d']['d_thresh']
    min_z_overlap = dc.reco['track_3d']['min_z_overlap']
    trk_min_dz = dc.reco['track_3d']['trk_min_dz']
    trk_min_dx = dc.reco['track_3d']['trk_min_dx']
    
    if(dc.evt_list[-1].det != 'pdhd'):
        unwrappers = [[0,0,0]]
    else:
        unwrappers = [[0,0,0], [cf.unwrappers[modules[0]][0], 0, 0], [0, cf.unwrappers[modules[0]][1], 0], [cf.unwrappers[modules[0]][0], cf.unwrappers[modules[0]][1], 0]]
    
    if(len(dc.tracks2D_list) < 2):
        return

    """ R-tree with the tracks """
    trk_pties = index.Property()
    trk_pties.dimension = 3
    rtree_trk_idx = index.Index(properties=trk_pties)

    trk_ID_shift = dc.tracks2D_list[0].trackID
    
    tracks = [x for x in dc.tracks2D_list if x.len_straight >= len_min and x.dz > trk_min_dz and x.dx > trk_min_dx and x.ghost == False and track_in_module(x, modules) == True]
    ntracks = len(tracks)
    ntracks_all = len(dc.tracks2D_list)

    #print('3D: ', ntracks, ' considered vs ', ntracks_all)
    #print([t.trackID for t in tracks])
    
    """ fill a rtree of 2D tracks (module, view, endpoints) """
    for t in tracks:
        start = t.path[0][1]
        stop  = t.path[-1][1]

        module_start = min(t.module_ini, t.module_end)
        module_stop =  max(t.module_ini, t.module_end)

        rtree_trk_idx.insert(t.trackID, (module_start, t.view, stop, module_stop, t.view, start))

    """ R-tree with the hits associated """
    hit_pties = index.Property()
    hit_pties.dimension = 4

    ''' create an rtree index of hits in the 2D tracks (module, view, trackID, z)'''
    rtree_hit_idx = index.Index(properties=hit_pties)
    
    hits_ID_shift = dc.hits_list[0].ID

    
    hits = [x for x in dc.hits_list if x.match_2D >=0 and dc.tracks2D_list[x.match_2D-trk_ID_shift].len_straight>=len_min and dc.tracks2D_list[x.match_2D-trk_ID_shift].dz > trk_min_dz and dc.tracks2D_list[x.match_2D-trk_ID_shift].dx > trk_min_dx and dc.tracks2D_list[x.match_2D-trk_ID_shift].match_3D==-1 and dc.tracks2D_list[x.match_2D-trk_ID_shift].ghost==False and x.module in modules]


    nhits = len(hits)
    
    ''' fill a R-tree with the hits associated to a 2D track '''
    for h in hits: 
        stop = min(h.Z_stop, h.Z_start)
        start = max(h.Z_stop, h.Z_start)                
        module = h.module
        rtree_hit_idx.insert(h.ID, (h.module, h.view, h.match_2D, stop, h.module, h.view, h.match_2D, start))



        
    tested = set()
    for t in tracks:

        
        t_start = t.path[0][1] + trk_ztol
        t_stop  = t.path[-1][1] - trk_ztol

        module_start = min(t.module_ini, t.module_end)
        module_stop =  max(t.module_ini, t.module_end)
        
        """ get all tracks in the other views compatible in time """
        trk_overlaps = [[] for x in range(cf.n_view)]
        for iview in range(cf.n_view):
            if(iview == t.view):
                trk_overlaps[iview].append(t.trackID)
            else:
                [trk_overlaps[iview].append(k) for k in list(rtree_trk_idx.intersection((module_start, iview, t_stop, module_stop, iview, t_start)))]

        
        n_per_view = [len(ov) for ov in trk_overlaps]

        if(any([x == 0 for x in n_per_view])):
            continue
        
        """ get all 3 tracks combinations """
        trk_comb = list(product(*trk_overlaps))
        
        for tc in trk_comb:
            
            """ don't test a combination already tested """
            if(tc in tested):
                continue
            else:
                tested.add(tc)


            trks = [dc.tracks2D_list[i-trk_ID_shift] for i in tc]
            
            starts = [(it.path[0][1],it.hits_ID[0]) for it in trks]
            starts = sorted(starts, key=itemgetter(0))
            
            stops  = [(it.path[-1][1],it.hits_ID[-1], ) for it in trks]
            stops = sorted(stops, key=itemgetter(0))

            """ get the smallest z overlap in time from the combination """
            hit_min_start = dc.hits_list[starts[0][1]-hits_ID_shift]
            hit_max_stop = dc.hits_list[stops[-1][1]-hits_ID_shift]            

            if(hit_max_stop.Z > hit_min_start.Z):
                continue
            
            dz = np.fabs(hit_min_start.Z-hit_max_stop.Z)
            
            if(dz < min_z_overlap):
                continue

            qtrk = [it.charge_in_z_interval(hit_max_stop.Z, hit_min_start.Z) for it in trks]
            
            qtrk_tot=sum(qtrk)
            if(qtrk_tot == 0):
                continue

            
            start_ov = [[] for x in range(cf.n_view)]
            stop_ov = [[] for x in range(cf.n_view)]

            for iv in range(cf.n_view):
                if(iv == hit_min_start.view):
                    start_ov[iv].append(hit_min_start)
                else:
                    intersect = list(rtree_hit_idx.intersection((module_start, iv, tc[iv], starts[0][0]-hit_ztol, module_stop, iv, tc[iv], starts[0][0]+hit_ztol)))
                    [start_ov[iv].append(dc.hits_list[k-hits_ID_shift]) for k in intersect]

                if(iv == hit_max_stop.view):
                    stop_ov[iv].append(hit_max_stop)
                else:
                    intersect = list(rtree_hit_idx.intersection((module_start, iv, tc[iv], stops[-1][0]-1, module_stop, iv, tc[iv], stops[-1][0]+1)))
                    [stop_ov[iv].append(dc.hits_list[k-hits_ID_shift]) for k in intersect]

            for u in unwrappers:
                unwrap = u
                start_ok, stop_ok = False, False

                start_d, start_xy, start_comb = h3d.compute_xy(start_ov, hit_min_start, d_thresh, u)
                stop_d, stop_xy, stop_comb = h3d.compute_xy(stop_ov, hit_max_stop, d_thresh, u) 
                
                if(start_d>=0 and stop_d>=0):
                    for tt in trks:
                        [tt.matched_tracks[i].append(t) for i,t in enumerate(tc) if i!=tt.view]
                            
                    break




    
    tb = time.time()

    
    """ at this point, we know that track A connects with tracks B and C, we need to sort out the connections """
    """ we use graphs to undersand who's connected to whom """
    """ the connections are used to build an adjacency matrix, from which graph(s) can be extracted telling us what are the set of tracks needed to build a 3D track"""
    

    sparse = np.zeros((ntracks_all, ntracks_all))
    for t in tracks:
        tid = t.trackID
        for match in t.matched_tracks:
            for tm in match:
                sparse[tid-trk_ID_shift, tm-trk_ID_shift] = 1


    
    graph = csr_matrix(sparse)
    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)

    count = Counter(labels)
    n_3D_tracks = sum([1 if k>1 else 0 for k in count.values()])

    
    tc = time.time()
    
    """ assign the label (potentital 3D track ID) to the 2D tracks """
    [it.set_label(l) for it,l in zip(dc.tracks2D_list,labels) ]

    """ merge 2D tracks in the same view """
    for ilab in range(n_components):
        tracks_to_3D = [[] for x in range(cf.n_view)]
        [tracks_to_3D[it.view].append(it) for it in dc.tracks2D_list if it.label3D==ilab]

        n_trks_per_view = [len(x) for x in tracks_to_3D]
        
        if(any(x==0 for x in n_trks_per_view)):
            continue
        
        if(any(x>1 for x in n_trks_per_view) and all(x<=2 for x in n_trks_per_view)):
            
            """ merge if needed """
            [stitch.stitch2D_from_3Dbuilder(tracks_to_3D[iv], debug=debug) for iv in range(cf.n_view)]


            
            tracks_to_3D = [[] for x in range(cf.n_view)]
            [tracks_to_3D[it.view].append(it) for it in dc.tracks2D_list if it.label3D==ilab]
            n_trks_per_view = [len(x) for x in tracks_to_3D]
            if(debug): print('there was too many tracks, and now ... ', n_trks_per_view)

        
        if(all(x==1 for x in n_trks_per_view)):
            flat_tracks = [x for t in tracks_to_3D for x in t]


        else:
            tracks_to_3D = [sorted(t, key=lambda x: x.len_straight,reverse=True) for t in tracks_to_3D]
            flat_tracks = [t[0] for t in tracks_to_3D]
        
        t3D = complete_trajectories(flat_tracks)
        
        
        isok = check_track_3D(t3D)
            
        if(isok == False):            
            #print('oh no 1')
            continue
            
        n_fake = t3D.check_views()
            
        if(n_fake > 1):
            continue
            
        t3D.boundaries()

        isok = finalize_3d_track(t3D)
        if(isok == False):
            #print('oh no 2')
            continue

        
        trk_ID = dc.evt_list[-1].n_tracks3D + dc.n_tot_trk3d #+1
        t3D.ID_3D = trk_ID
        correct_timing(t3D, dx_tol, dy_tol, dz_tol)


        
        dc.tracks3D_list.append(t3D)
        dc.evt_list[-1].n_tracks3D += 1
            
        for t in flat_tracks:
            t.match_3D = trk_ID
            t.set_match_hits_3D(trk_ID)


    stitch.reset_track2D_and_update_track3D_lists()
