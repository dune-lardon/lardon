import lardon.config as cf
import lardon.data_containers as dc
import lardon.lar_param as lar
import numpy as np


def compute_exit_point(trk, idx_anode, zcorr):
    debug = False

    modules = [trk.module_ini, trk.module_end]
    z_bounds = [cf.anode_z[m] for m in modules]
    z_bounds.extend([cf.anode_z[m]-cf.drift_direction[m]*cf.drift_length[m] for m in modules])

    box_min = [min([cf.x_boundaries[i][0] for i in modules]),min([cf.y_boundaries[i][0] for i in modules]), min(z_bounds)]
    box_max = [max([cf.x_boundaries[i][1] for i in modules]),max([cf.y_boundaries[i][1] for i in modules]), max(z_bounds)]
    
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


    else:        
        print('Exit point computation to be checked for ', dc.evt_list[-1].det,' geometry')
        return [-9999, -9999, -9999]

    direction = np.array([dx, dy, dz])
    origin = np.array(point)

    inv_dir = 1.0 / direction  # Inverse to avoid dividing multiple times

    # Compute tmin and tmax for slabs
    tmin = (box_min - origin) * inv_dir
    tmax = (box_max - origin) * inv_dir

    # Swap if needed to ensure correct ordering
    t1 = np.minimum(tmin, tmax)
    t2 = np.maximum(tmin, tmax)

    if(debug):
        print("tmin", tmin)
        print("tmax", tmax)
        print(t1)
        print(t2)
        
    # We're inside the box, so t_enter < 0 and we want the smallest positive t_exit
    t_exit = np.min(t2)
    
    
    if t_exit <= 0:
        return [-9999, -9999, -9999]  # Line doesn't exit in the direction given

    exit_point = origin + t_exit * direction
    if(debug):
        print('!!!!!!!!!!!!====>>>>> exit point ', exit_point)
        
    return  exit_point    





def compute_time_for_anode_crosser(trk):

    mod_ini, mod_end = trk.module_ini, trk.module_end    
    z_anodes = [cf.anode_z[mod_ini], cf.anode_z[mod_end]]
    trk_z_bounds = [trk.ini_z, trk.end_z]    
    trk_modules  = [mod_ini, mod_end]
    trk_times    = [trk.ini_time/cf.sampling[mod_ini], trk.end_time/cf.sampling[mod_end]]

    """ idx is the track endpoint closest to the anode """
    idx, idx_o = (0,1) if np.argmin([np.fabs(t-z) for t,z in zip(trk_z_bounds, z_anodes)]) == 0 else (1,0)

    z0 = z_anodes[idx] - trk_z_bounds[idx]
    tref = trk_times[idx]
    t0 = tref
    
    exit_point = compute_exit_point(trk, idx, z0)
    
    return tref, t0, z0, exit_point, idx





def compute_missing_time_for_late_cathode_crosser(trk, debug):
    other = dc.tracks3D_list[trk.cathode_crosser_ID - dc.n_tot_trk3d]
    ta, tb = trk, other

    is_horizontal = False
    if(dc.evt_list[-1].det == 'pdhd'):
        is_horizontal = True
        if(ta.ini_x < tb.ini_x):
            ta, tb = tb, ta
            is_horizontal = True

    elif(dc.evt_list[-1].det == 'pdvd'):
        if(ta.module_ini < tb.module_ini):
            ta, tb = tb, ta
            is_horizontal = False



    a2 = np.asarray([ta.end_x, ta.end_y, ta.end_z])    
    b1 = np.asarray([tb.ini_x, tb.ini_y, tb.ini_z])


    dx = np.fabs(a2[0]-b1[0])

    if(debug):
        print('--> dx = ', dx)
    if(is_horizontal):
        dz_a = np.fabs(-dx*np.sin(np.radians(ta.end_phi))*np.tan(np.radians(ta.end_theta)))
        dz_b = np.fabs(-dx*np.sin(np.radians(tb.ini_phi))*np.tan(np.radians(tb.ini_theta)))
    else:
        if(np.tan(np.radians(ta.end_theta)) == 0. or np.tan(np.radians(tb.ini_theta)) == 0):
            return 0.
        
        if(np.cos(np.radians(ta.end_phi)) != 0.):
            dz_a = np.fabs(-dx/np.tan(np.radians(ta.end_theta))/np.cos(np.radians(ta.end_phi)))
        else:
            dz_a = np.fabs(-dx/np.tan(np.radians(ta.end_theta)))
        print('Dz a= ', dz_a)
        
        if(np.cos(np.radians(tb.ini_phi)) != 0):                           
            dz_b = np.fabs(-dx/np.tan(np.radians(tb.ini_theta))/np.cos(np.radians(tb.ini_phi)))
        else:
            dz_b = np.fabs(-dx/np.tan(np.radians(tb.ini_theta)))

    if(ta.ID_3D == trk.ID_3D):
        return dz_a
    else:
        return dz_b

    
def compute_time_for_cathode_crosser(trk, debug=False):    

    ztol = dc.reco['track_3d']['timing']['dz_tol']
    
    mod_ini, mod_end = trk.module_ini, trk.module_end    
    z_anodes = [cf.anode_z[mod_ini], cf.anode_z[mod_end]]

    vdrift = lar.drift_velocity(mod_ini)
    max_drifts = [cf.anode_z[mod_ini] - cf.n_sample[mod_ini]*cf.drift_direction[mod_ini] * vdrift /cf.sampling[mod_ini], cf.anode_z[mod_ini] - cf.n_sample[mod_ini]*cf.drift_direction[mod_ini] * vdrift /cf.sampling[mod_ini]]

    drift_times = [ cf.drift_length[mod_ini]/vdrift, cf.drift_length[mod_end]/vdrift]    
    z_cathodes = [cf.anode_z[mod_ini] - cf.drift_direction[mod_ini]*cf.drift_length[mod_ini], cf.anode_z[mod_end] - cf.drift_direction[mod_end]*cf.drift_length[mod_end]]

    trk_z_bounds = [trk.ini_z, trk.end_z]    
    trk_modules  = [mod_ini, mod_end]
    trk_times    = [trk.ini_time/cf.sampling[mod_ini], trk.end_time/cf.sampling[mod_end]]

    is_too_late = np.any(np.asarray([np.fabs(t-z) < ztol for t,z in zip(trk_z_bounds, max_drifts)], dtype = bool))

    vdrift = lar.drift_velocity(mod_ini)
    
    missing_z = 0.

    if(debug):
        print("max drifts: ", max_drifts, "is too late?", is_too_late)

    """
    if(is_too_late and 
        missing_z = compute_missing_time_for_late_cathode_crosser(trk)
        if(debug):
            print('cathode-crosser is late ! --> missing Deltaz =', missing_z )
    """
    
    if(trk.cathode_crosser_ID >=0): 
        missing_z = compute_missing_time_for_late_cathode_crosser(trk, debug)
    if(debug):
        print('paired cathode-crosser --> missing Deltaz =', missing_z )
    
    """ idx is the trk endpoint closest to the anode"""
    idx = np.argmin([np.fabs(t-z) for t,z in zip(trk_z_bounds, z_anodes)])
    idx_o = 1 if idx == 0 else 0
    z0 = z_cathodes[idx_o] - trk_z_bounds[idx_o] + cf.drift_direction[mod_end]*missing_z/2.
    tref = trk_times[idx_o]-drift_times[idx_o] + missing_z/2./vdrift
      
    t0 = tref
    
    return tref, t0, z0





def test_topologies(trk):
    xtol = dc.reco['track_3d']['timing']['dx_tol']
    ytol = dc.reco['track_3d']['timing']['dy_tol']

    ztol = dc.reco['track_3d']['timing']['dz_tol']
    dtol = dc.reco['track_3d']['timing']['drift_tol']
    
    """ track points are ordered by decreasing vertical axis """
    trk_z_bounds = [trk.ini_z, trk.end_z]
    dz = np.fabs(trk_z_bounds[1]-trk_z_bounds[0])
    is_full_drift = min(cf.drift_length) - dtol < dz < max(cf.drift_length) + dtol

    mod_ini, mod_end = trk.module_ini, trk.module_end

    vdrift = lar.drift_velocity(mod_ini)
    
    z_anodes = [cf.anode_z[mod_ini], cf.anode_z[mod_end]]
    
    """ longest drift possible (depends on TPC acquisition window) """
    max_drifts = [cf.anode_z[mod_ini] - cf.n_sample[mod_ini]*cf.drift_direction[mod_ini] * vdrift /cf.sampling[mod_ini], cf.anode_z[mod_ini] - cf.n_sample[mod_ini]*cf.drift_direction[mod_ini] * vdrift /cf.sampling[mod_ini]]
    drift_times = [ cf.drift_length[mod_ini]/vdrift, cf.drift_length[mod_end]/vdrift]
    
    z_cathodes = [cf.anode_z[mod_ini] - cf.drift_direction[mod_ini]*cf.drift_length[mod_ini], cf.anode_z[mod_end] - cf.drift_direction[mod_end]*cf.drift_length[mod_end]]

           
    is_too_early = np.asarray([np.fabs(t-z) < ztol for t,z in zip(trk_z_bounds, z_anodes)], dtype=bool)
    is_too_late = np.asarray([np.fabs(t-z) < ztol for t,z in zip(trk_z_bounds, max_drifts)], dtype = bool)

    
    inside_ini_x = cf.x_boundaries[mod_ini][0]+xtol[mod_ini][0] <= trk.ini_x <= cf.x_boundaries[mod_ini][1]-xtol[mod_ini][1]
    inside_ini_y = cf.y_boundaries[mod_ini][0]+ytol[mod_ini][0] <= trk.ini_y <= cf.y_boundaries[mod_ini][1]-ytol[mod_ini][1]
    through_wall_ini = np.any([not inside_ini_x, not inside_ini_y])


    inside_end_x = cf.x_boundaries[mod_end][0]+xtol[mod_end][0] <= trk.end_x <= cf.x_boundaries[mod_end][1]-xtol[mod_end][1]
    inside_end_y = cf.y_boundaries[mod_end][0]+ytol[mod_end][0] <= trk.end_y <= cf.y_boundaries[mod_end][1]-ytol[mod_end][1]
    through_wall_end = np.any([not inside_end_x, not inside_end_y])

    return is_full_drift, through_wall_ini, through_wall_end, is_too_early, is_too_late


def adjust_timestamp_range(trk):
    if(trk.ID_3D < trk.cathode_crosser_ID):
        return
    
    trk_timestamp = trk.timestamp
    other = dc.tracks3D_list[trk.cathode_crosser_ID - dc.n_tot_trk3d]
    other_timestamp = other.timestamp
    
    trk.set_timestamp(trk_timestamp, other_timestamp)
    other.set_timestamp(trk_timestamp, other_timestamp)

    """
    trk.dump()
    print('WITH')
    other.dump()

    print('--->>>> TIMESTAMP RANGE ', trk_timestamp, other_timestamp, " === ", trk_timestamp - other_timestamp)
    """
    
def compute_all_track_timing():
    [compute_timing(t) for t in dc.tracks3D_list]

    cathode_crossers = [t for t in dc.tracks3D_list if t.is_cathode_crosser and t.cathode_crosser_ID >= 0]
    [adjust_timestamp_range(t) for t in cathode_crossers]
    
def compute_timing(trk):
    debug = False#True

    is_anode_crosser = trk.is_anode_crosser
    is_cathode_crosser = trk.is_cathode_crosser and trk.cathode_crosser_ID >= 0
    has_already_t0 = trk.t0_corr < 9999.
    
    is_full_drift, through_wall_ini, through_wall_end, is_too_early, is_too_late = test_topologies(trk)

    
    mod_ini, mod_end = trk.module_ini, trk.module_end

        
    trk_z_bounds = [trk.ini_z, trk.end_z]    
    trk_modules  = [mod_ini, mod_end]
    trk_times    = [trk.ini_time/cf.sampling[mod_ini], trk.end_time/cf.sampling[mod_end]]

    vdrift = lar.drift_velocity(mod_ini)
    z_anodes = [cf.anode_z[mod_ini], cf.anode_z[mod_end]]
    z_cathodes = [cf.anode_z[mod_ini] - cf.drift_direction[mod_ini]*cf.drift_length[mod_ini], cf.anode_z[mod_end] - cf.drift_direction[mod_end]*cf.drift_length[mod_end]]
    
    drift_times = [ cf.drift_length[mod_ini]/vdrift, cf.drift_length[mod_end]/vdrift]
    
    """ idx is the track endpoint closest to anode """
    idx, idx_o = (0,1) if np.argmin([np.fabs(t-z) for t,z in zip(trk_z_bounds, z_anodes)]) == 0 else (1,0)


    if(debug):
        trk.dump()
        
        print('\nIs full drift ? ', is_full_drift)
        print('Through wall ini ?', through_wall_ini, ' end ? ', through_wall_end)
        print('Is too early ? ', is_too_early, ' is too late?' , is_too_late)
        print('Is anode crosser ? ', is_anode_crosser, ' Cathode crosser ? ', is_cathode_crosser, ' has already a T0? ', has_already_t0,"\n")


    
    z0 = 9999.
    t0 = 9999.
    
    
    if(not is_full_drift and is_anode_crosser and is_cathode_crosser):
        print("There is a reconstruction issue !!! ")
        print("--> Track ", trk.ID_3D, " is tagged as anode & cathode crosser, but the drift length is ", np.fabs(trk_z_bounds[1]-trk_z_bounds[0]))
        print("!!! PLEASE CROSS CHECK !!! ")



    """ start with the most obvious case: track crossed both anode & cathode """
    if(is_full_drift or (is_anode_crosser and is_cathode_crosser)):
        tref, t0, z0, exit_point, idx_exit = compute_time_for_anode_crosser(trk)

        trk.set_t0_z0(t0, z0)
        trk.compute_timestamp(tref, tref)
        trk.is_anode_crosser = True
        trk.set_anode_crosser(exit_point, idx_exit)
        trk.is_cathode_crosser = True
        
        if(is_cathode_crosser):
            other = dc.tracks3D_list[trk.cathode_crosser_ID - dc.n_tot_trk3d]

            if(other.is_anode_crosser):
                o_tref, o_t0, o_z0, o_exit_point, o_idx_exit = compute_time_for_anode_crosser(other)
                other.compute_timestamp(o_tref, o_tref)
                other.set_t0_z0(o_t0, o_z0)
                other.is_anode_crosser = True
                other.set_anode_crosser(o_exit_point, o_idx_exit)
                other.is_cathode_crosser = True

                ts_trk, ts_other = trk.timestamp, other.timestamp
                trk.set_timestamp(ts_trk, ts_other)
                other.set_timestamp(ts_trk, ts_other)
                if(debug):
                    print('--> A-C-A case !!! Delta timestamp: ', ts_trk, ts_other, '=',ts_trk - ts_other)
                    print('--the other track ', o_t0, o_z0, other.timestamp)
            else:
                other.set_t0_z0_from_timestamp(trk.timestamp, vdrift)

                if(debug):
                    print('--other track ', other.t0_corr, other.z0_corr, other.timestamp)


                
        if(debug):
            print('Full drift on time !')
            print('T0 = ', t0, 'z0', z0)
            print('TS = ', trk.timestamp)
            print('New endpoints = ', [z+z0 for z in trk_z_bounds])            
            print("EXIT point = ", trk.exit_point)
        return



    if(is_anode_crosser):
        tref, t0, z0, exit_point, idx_exit = compute_time_for_anode_crosser(trk)

        trk.set_t0_z0(t0, z0)
        trk.compute_timestamp(tref, tref)
        trk.is_anode_crosser = True
        trk.set_anode_crosser(exit_point, idx_exit)

        if(debug):
            print('Known Anode Crosser!')
            print('T0 = ', t0, 'z0', z0)
            print('TS = ', trk.timestamp)
            print('New endpoints = ', [z+z0 for z in trk_z_bounds])            
            print("EXIT point = ", trk.exit_point)
        return


    
    if(is_cathode_crosser):
        tref, t0, z0 = compute_time_for_cathode_crosser(trk, debug)
        trk.set_t0_z0(t0, z0)
        trk.compute_timestamp(tref, tref)
        trk.is_cathode_crosser = True

        if(debug):
            print('Known Cathode Crosser!')
            print('T0 = ', t0, 'z0', z0)
            print('TS = ', trk.timestamp)
            print('New endpoints = ', [z+z0 for z in trk_z_bounds])            

        return





    
    if(has_already_t0):
        return

    
    """ FC-FC tracks """
    if(through_wall_ini and through_wall_end):
        if(is_cathode_crosser):
            tref, t0, z0 = compute_time_for_cathode_crosser(trk)
            trk.set_t0_z0(t0, z0)
            trk.compute_timestamp(tref, tref)
            
            if(debug):
                print('FC-FC but! cathode crosser')
                print('T0 = ', t0, 'z0', z0)
                print('TS = ', trk.timestamp)
                print('New endpoints = ', [z+z0 for z in trk_z_bounds])
                
            return

            
        """ unresolved case """
        trk.set_t0_z0(t0, z0)        

        """ compute the range of possible timestamps """
        tref_a = trk_times[idx] # latest possible time
        tref_b = trk_times[idx_o]-drift_times[idx_o] #earliest possible time
        trk.compute_timestamp(tref_a, tref_b)
        
        if(debug):
            print('FC to FC: unresolved')
            print('T0 can be from ', tref_a, 'to', tref_b)
            print('TS = ', trk.timestamp, ' to ', trk.timestamp_r)
            if(trk.is_anode_crosser):
                print("EXIT point = ", trk.exit_point)

        return
    
    """ case when one track endpoint is near the FC """
    if(through_wall_ini or through_wall_end):
        
        ''' fc_idx is the track endpoint at FC '''
        fc_idx, fc_idx_o = (0, 1) if through_wall_ini else (1, 0)

        ''' is the other endpoint at the tpc window boundary? '''
        if(is_too_early[fc_idx_o] or is_too_late[fc_idx_o]):

            if(is_cathode_crosser):
                tref, t0, z0 = compute_time_for_cathode_crosser(trk)
                trk.set_t0_z0(t0, z0)
                trk.compute_timestamp(tref, tref)

                if(debug):
                    print('FC-Cathode')
                    print('T0 = ', t0, 'z0', z0)
                    print('TS = ', trk.timestamp)
                    print('New endpoints = ', [z+z0 for z in trk_z_bounds])
                    
                return

            
            """ wall to readout : unresolved case """
            trk.set_t0_z0(t0, z0)
            
            """ compute time range of possibilities """
            tref_a = trk_times[idx] # latest possible time
            tref_b = trk_times[idx_o]-drift_times[idx_o] #earliest possible time

            trk.compute_timestamp(tref_a, tref_b)


            if(debug):
                print('FC to readout: unresolved')
                print('T0 can be from ', tref_a, 'to', tref_b)
                print('TS = ', trk.timestamp, ' to ', trk.timestamp_r)

            return

        else:
            """ then the track did cross anode or cathode """
            """ let's find which track endpoint """
            zdir = 1 if (trk_z_bounds[fc_idx_o]-trk_z_bounds[fc_idx]) > 0 else -1

            if(zdir == cf.drift_direction[trk_modules[fc_idx_o]]):

                if(is_cathode_crosser):                                            
                    tref, t0, z0 = compute_time_for_cathode_crosser(trk)
                    trk.set_t0_z0(t0, z0)
                    trk.compute_timestamp(tref, tref)                
                    
                    if(debug):
                        print('Cathode crosser with FC !')
                        print('T0 = ', t0, 'z0', z0)
                        print('TS = ', trk.timestamp)
                        print('New endpoints = ', [z+z0 for z in trk_z_bounds])
                        
                    return


                """ the track has the same direction as the drift direction: use anode """
                tref, t0, z0, exit_point, idx_exit = compute_time_for_anode_crosser(trk)

                trk.set_t0_z0(t0, z0)
                trk.compute_timestamp(tref, tref)
                trk.is_anode_crosser = True
                trk.set_anode_crosser(exit_point, idx_exit)

                if(debug):
                    print('FC-anode !')
                    print('T0 = ', t0, 'z0', z0)
                    print('TS = ', trk.timestamp)
                    print('New endpoints = ', [z+z0 for z in trk_z_bounds])
                    print("EXIT point = ", trk.exit_point)
        
                            
            else:
                """ the track has opposite direction as the drift direction: use cathode """
                tref, t0, z0 = compute_time_for_cathode_crosser(trk)
                trk.set_t0_z0(t0, z0)
                trk.compute_timestamp(tref, tref)                
                trk.is_cathode_crosser = True

                if(debug):
                    print('FC-cathode !')
                    print('T0 = ', t0, 'z0', z0)
                    print('TS = ', trk.timestamp)
                    print('New endpoints = ', [z+z0 for z in trk_z_bounds])
                
            return




    """ if we're here : the track did not enter nor escaped by the FC """    
    if(np.any(is_too_early)):
        """ then track is early """
        tref, t0, z0 = compute_time_for_cathode_crosser(trk)
        trk.set_t0_z0(t0, z0)
        trk.compute_timestamp(tref, tref)
        trk.is_cathode_crosser = True
        
        if(debug):
            print('Early Cathode crosser!')
            print('T0 = ', t0, 'z0', z0)
            print('TS = ', trk.timestamp)
            print('New endpoints = ', [z+z0 for z in trk_z_bounds])
            
        return


    if(np.any(is_too_late)):
        """ then track is late """
        tref, t0, z0, exit_point, idx_exit = compute_time_for_anode_crosser(trk)

        trk.set_t0_z0(t0, z0)
        trk.compute_timestamp(tref, tref)
        trk.is_anode_crosser = True
        #the exiting point was not recorded: the 'true' exiting point is pointless
        #trk.set_anode_crosser(exit_point, idx_exit)
            
        if(is_cathode_crosser):
            other = dc.tracks3D_list[trk.cathode_crosser_ID - dc.n_tot_trk3d]
            if(other.is_anode_crosser):
                ts_trk, ts_other = trk.timestamp, other.timestamp
                trk.set_timestamp(ts_trk, ts_other)
                other.set_timestamp(ts_trk, ts_other)
            else:
                other.set_t0_z0(t0, z0*cf.drift_direction[other.module_ini])
                other.compute_timestamp(tref, tref)
                


        if(debug):
            print('Late Anode crosser!')
            print('T0 = ', t0, 'z0', z0)
            print('TS = ', trk.timestamp)
            print('New endpoints = ', [z+z0 for z in trk_z_bounds])
            print('---exit point not relevant')

        return
    



    """ what's left is a late track entering from the higher plane """
    highest_plane_idx = np.argmax([z_anodes[0], z_cathodes[0]])
    highest_plane_is_anode = True if  highest_plane_idx == 0 else False

    """ idx is the track endpoint closer to the highest plane """    
    """ by construction, track initial point are the highest point """
    """ NB: check for PDHD """
    if(highest_plane_is_anode):
        
        tref, t0, z0, exit_point, idx_exit = compute_time_for_anode_crosser(trk)
        trk.set_t0_z0(t0, z0)
        trk.compute_timestamp(tref, tref)
        trk.is_anode_crosser = True

        #the exiting point is not relevant in this case
        #trk.set_anode_crosser(exit_point, idx_exit)

        if(is_cathode_crosser):
            other = dc.tracks3D_list[trk.cathode_crosser_ID - dc.n_tot_trk3d]
            if(other.is_anode_crosser):
                ts_trk, ts_other = trk.timestamp, other.timestamp
                trk.set_timestamp(ts_trk, ts_other)
                other.set_timestamp(ts_trk, ts_other)
            else:
                other.set_t0_z0(t0, z0*cf.drift_direction[other.module_ini])
                other.compute_timestamp(tref, tref)



        if(debug):
            print('Anode crosser by default!')
            print('T0 = ', t0, 'z0', z0)
            print('TS = ', trk.timestamp)
            print('New endpoints = ', [z+z0 for z in trk_z_bounds])
            print('---exit point not relevant')

                
    else:
        tref, t0, z0 = compute_time_for_cathode_crosser(trk)
        trk.set_t0_z0(t0, z0)
        trk.compute_timestamp(tref, tref)
        trk.is_cathode_crosser = True

        
        if(debug):
            print('Cathode crosser by default!')
            print('T0 = ', t0, 'z0', z0)
            print('TS = ', trk.timestamp)
            print('New endpoints = ', [z+z0 for z in trk_z_bounds])


