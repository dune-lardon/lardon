import lardon.data_containers as dc
import lardon.config as cf
import lardon.lar_param as lar
import jsonc as json

import numpy as np

def configure(detector):
    the_file = cf.lardon_path+'/settings/'+detector+'/trigger.json'
    try:
        with open(the_file,'r') as f:        
            dc.trig_conf = json.load(f)['default']

    except IOError:
        print("WARNING: Trigger configuration ",the_file," not found.")
        print("       -> no trigger searches can be done.")
        dc.trig_conf = {'trigger_nb':[]}
        #print(dc.trig_conf)
        
def search_trigger_track():
    if(dc.evt_list[-1].trig_type in dc.trig_conf['trigger_nb']):
        trigger_case = dc.trig_conf['trigger_case'][str(dc.evt_list[-1].trig_type)]


        if("laser" in trigger_case):
            return is_laser_event(trigger_case)
        elif("crt" in trigger_case):
            return is_crt_event(trigger_case)
        elif("beam" in trigger_case):
            return is_beam_event(trigger_case)


def track_to_vertex_distance(vertex, track, zcorr, radius):

    # track starting points
    p = np.vstack((track.ini_x, track.ini_y, track.ini_z+zcorr)).T

    # direction vectors from angles
    #WARNING --- CHECK FOR PDHD !!!!! 
    vx = np.sin(np.radians(track.ini_theta)) * np.cos(np.radians(track.ini_phi))
    vy = np.sin(np.radians(track.ini_theta)) * np.sin(np.radians(track.ini_phi))
    vz = np.cos(np.radians(track.ini_theta))

    v = np.vstack((vx, vy, vz)).T

    # normalize
    v = v / np.linalg.norm(v, axis=1)[:, None]

    # vector vertex -> track point
    diff = vertex - p

    # cross product
    cross = np.cross(diff, v)

    # distance
    d = np.linalg.norm(cross, axis=1)

    return d < radius



def is_laser_event(trig_case):    
    laser_point = np.array([float(dc.trig_conf[trig_case]['point_x']),
                            float(dc.trig_conf[trig_case]['point_y']),
                            float(dc.trig_conf[trig_case]['point_z'])])

    search_radius = float(dc.trig_conf[trig_case]['search_radius'])
    search_modules = [int(x) for x in dc.trig_conf[trig_case]['search_modules']]

    vdrift = lar.drift_velocity(imod=search_modules[0])
    trigger_ts = dc.evt_list[-1].event_time
    z_corr = 1e6*(trigger_ts - dc.evt_list[-1].charge_time[search_modules[0]])*vdrift

    trks = [t for t in dc.tracks3D_list if t.module_ini in search_modules and track_to_vertex_distance(laser_point, t, z_corr, search_radius)==True]
    [t.set_trigger_track(0., vdrift) for t in trks]


    #[t.dump() for t in trks]
    
    print('Found ', len(trks), ' laser  tracks!')
    return
def is_beam_event(trig_case):
    return
def is_crt_event(trig_case):
    return
