import lardon.config as cf
import lardon.data_containers as dc
import lardon.channel_mapping as chmap

from tables import *
import numpy as np
import itertools as itt


class Infos(IsDescription):
    run          = UInt64Col()
    sub          = StringCol(6)
    elec         = StringCol(3)
    n_evt        = UInt32Col()
    process_date = UInt32Col()
    n_channels   = UInt16Col()
    sampling     = Float32Col(shape=(cf.n_module_used))
    n_samples    = Float32Col(shape=(cf.n_module_used))
    n_view       = UInt8Col()
    view_nchan   = Float32Col(shape=(cf.n_view))
    e_drift      = Float32Col(shape=(cf.n_module_used))
    t_lar        = Float32Col(shape=(cf.n_module_used))


class Event(IsDescription):
    trigger_nb       = UInt32Col()
    trigger_type     = UInt16Col()
    time_s           = UInt64Col()
    time_ns          = UInt64Col()
    charge_time      = Float64Col(shape=(cf.n_module))
    pds_stream_time  = Float64Col()
    pds_trig_time    = Float64Col()
    n_sample         = UInt32Col(shape=(cf.n_module_used))
    n_hits           = UInt32Col(shape=(cf.n_view, cf.n_module))
    n_tracks2D       = UInt32Col(shape=(cf.n_view))
    n_tracks3D       = UInt32Col()
    n_single_hits    = UInt32Col()
    n_ghosts         = UInt32Col()

    

class Pedestal(IsDescription):
    raw_mean   = Float32Col(shape=(cf.n_tot_channels))
    raw_rms    = Float32Col(shape=(cf.n_tot_channels))
    filt_mean  = Float32Col(shape=(cf.n_tot_channels))
    filt_rms   = Float32Col(shape=(cf.n_tot_channels))


class PDSPedestal(IsDescription):
    raw_mean   = Float32Col(shape=(cf.n_pds_tot_channels))
    raw_rms    = Float32Col(shape=(cf.n_pds_tot_channels))
    filt_mean   = Float32Col(shape=(cf.n_pds_tot_channels))
    filt_rms    = Float32Col(shape=(cf.n_pds_tot_channels))
    
class NoiseStudy(IsDescription):
    delta_mean  = Float32Col(shape=(cf.n_tot_channels))
    rms         = Float32Col(shape=(cf.n_tot_channels))

'''
class FFT(IsDescription):
    """ ADJUST THE NUMBERS OF THE PS SHAPE ACCORDING TO THE RUN CONFIGURATION """
    print('test: ', (cf.module_nchan[0]/2, int(cf.n_sample[0]/2)+1), ' and ', cf.module_nchan[2]/2, int(cf.n_sample[2]/2)+1)
    ps_0 = Float32Col(shape=(cf.module_nchan[0], 3937))
    ps_1 = Float32Col(shape=(cf.module_nchan[1], 3937))
    ps_2 = Float32Col(shape=(cf.module_nchan[2], 4033))
    ps_3 = Float32Col(shape=(cf.module_nchan[3], 4033))
'''
'''
class Corr(IsDescription):
    corr_0 = Float32Col(shape=(cf.module_nchan[0], cf.module_nchan[0]))
    corr_1 = Float32Col(shape=(cf.module_nchan[1], cf.module_nchan[1]))
    corr_2 = Float32Col(shape=(cf.module_nchan[2], cf.module_nchan[2]))
    corr_3 = Float32Col(shape=(cf.module_nchan[3], cf.module_nchan[3]))
'''

class Waveform(IsDescription):
    view        = UInt8Col()
    channel     = UInt16Col()
    daq_channel = UInt16Col()

    pos_mean = Float32Col(shape=(60))
    pos_std  = Float32Col(shape=(60))
    neg_mean = Float32Col(shape=(60))
    neg_std  = Float32Col(shape=(60))


class Pulse(IsDescription):
    event   = UInt32Col()
    trigger = UInt32Col()
    
    view        = UInt8Col()
    channel     = UInt16Col()
    daq_channel = UInt16Col()

    n_pulse_pos = UInt16Col()
    n_pulse_neg = UInt16Col()

class PDSInfos(IsDescription):
    run          = UInt16Col()
    sub          = StringCol(6)
    elec         = StringCol(3)
    n_evt        = UInt32Col()
    process_date = UInt32Col()
    n_channels   = UInt16Col()
    sampling     = Float32Col()
    n_samples    = Float32Col()
    e_drift      = Float32Col(shape=(cf.n_module_used))
    t_lar        = Float32Col(shape=(cf.n_module_used))

class PDSEvent(IsDescription):
    event         = UInt32Col()
    trigger_nb    = UInt32Col()
    stream_time   = Float64Col()
    trig_time     = Float64Col()
    n_sample      = UInt32Col()
    n_peak        = UInt32Col(shape=(cf.n_pds_tot_channels))
    n_cluster     = UInt32Col()
    

class Hits(IsDescription):
    event   = UInt32Col()
    trigger = UInt32Col()
    ID = UInt32Col()

    module      = UInt8Col()
    view        = UInt8Col()
    channel     = UInt16Col()
    daq_channel = UInt16Col()

    is_collection  = BoolCol()

    tdc_start = Int32Col()
    tdc_stop  = Int32Col()

    tdc_max  = Int32Col()
    tdc_min  = Int32Col()
    tdc_zero = Int32Col()

    z       = Float32Col()
    x       = Float32Col()


    fC_max  = Float32Col()
    fC_min  = Float32Col()

    charge_pos = Float32Col()
    charge_neg = Float32Col()

    is_free = BoolCol()

    match_3D = Int32Col()
    match_2D = Int32Col()
    match_dray = Int32Col()
    match_ghost = Int32Col()
    match_sh = Int32Col()

class Tracks2D(IsDescription):
    event   = UInt32Col()
    trigger = UInt32Col()
    
    ID       = UInt32Col()
    match_3D = Int32Col()
    matched  = Int32Col(shape=(cf.n_view))

    view    = UInt8Col()
    pos_ini = Float32Col()
    pos_end = Float32Col()
    z_ini   = Float32Col()
    z_end   = Float32Col()

    chi2_fwd    = Float32Col()
    chi2_bwd    = Float32Col()

    n_hits_track  = UInt16Col()
    n_hits_dray  = UInt16Col()
    dray_total_charge = Float32Col()


    slope_ini = Float32Col()
    slope_end = Float32Col()
    slope_ini_err = Float32Col()
    slope_end_err = Float32Col()

    len_straight = Float32Col()
    len_path     = Float32Col()

    track_total_charge = Float32Col()



class Tracks3D(IsDescription):

    event   = UInt32Col()
    trigger = UInt32Col()

    matched_2D = Int32Col(shape=(cf.n_module, cf.n_view))
    n_matched = UInt32Col()
    
    ID = UInt32Col()
    
    module_ini = Int32Col()
    module_end = Int32Col()
    
    x_ini   = Float32Col()
    y_ini   = Float32Col()
    z_ini   = Float32Col()
    t_ini   = Int32Col()
    x_end   = Float32Col()
    y_end   = Float32Col()
    z_end   = Float32Col()
    t_end   = Int32Col()
    chi2    = Float32Col(shape=(cf.n_view))
    

    z_ini_overlap = Float32Col()
    z_end_overlap = Float32Col()

    theta_ini = Float32Col()
    theta_end = Float32Col()
    phi_ini   = Float32Col()
    phi_end   = Float32Col()

    n_hits       = UInt16Col(shape=(cf.n_view))
    len_straight = Float32Col(shape=(cf.n_view))
    len_path     = Float32Col(shape=(cf.n_view))
    total_charge = Float32Col(shape=(cf.n_view))

    z0_corr = Float64Col()
    t0_corr = Float32Col()

    d_match = Float32Col()
    timestamp  = Float64Col()
    cluster_ID = Int32Col()

    is_cathode_crosser = BoolCol()
    cathode_crossing = Float32Col(shape=(2,6))
    cathode_crosser_ID = Int32Col()
    cathode_crossing_trk_end = Int32Col()
    
    is_module_crosser = BoolCol()
    module_crossing = Float32Col(shape=(2,6))

    is_anode_crosser = BoolCol()
    exit_point = Float32Col(shape=(3))
    exit_trk_end = Int32Col()
    


class Ghost(IsDescription):

    event   = UInt32Col()
    trigger = UInt32Col()

    match_3D = UInt32Col()
    match_2D = UInt32Col()

    x_anode   = Float32Col()
    y_anode   = Float32Col()
    z_anode   = Float32Col()

    theta = Float32Col()

    phi   = Float32Col()

    n_hits       = UInt16Col()
    total_ghost_charge = Float32Col()
    total_track_charge = Float32Col()

    z0_corr = Float64Col()
    t0_corr = Float32Col()

    d_min = Float32Col()

class Hits3D(IsDescription):
    event   = UInt32Col()
    trigger = UInt32Col()
    ID = UInt32Col()

    view        = UInt8Col()
    channel     = UInt16Col()
    daq_channel = UInt16Col()
    module      = UInt8Col()
    
    x = Float64Col()
    y = Float64Col()
    z = Float64Col()
    n_cluster = Int32Col(shape=(cf.n_view))
    match_ID  =  Int32Col(shape=(cf.n_view))

class SingleHits(IsDescription):
    event   = UInt32Col()
    trigger = UInt32Col()

    n_hits = UInt32Col(shape=(cf.n_view))
    hit_IDs = Int32Col(shape=(cf.n_view,dc.reco['single_hit']['max_per_view']))
    
    ID = UInt32Col()
    module = UInt8Col()
    
    charge_pos = Float32Col(shape=(cf.n_view))
    charge_neg = Float32Col(shape=(cf.n_view))

    #charge_extend = Float32Col(shape=(cf.n_view))
    #charge_extend_pos = Float32Col(shape=(cf.n_view))
    #charge_extend_neg = Float32Col(shape=(cf.n_view))

    tdc_start = Int32Col(shape=(cf.n_view))
    tdc_stop  = Int32Col(shape=(cf.n_view))
    tdc_max   = Int32Col(shape=(cf.n_view))
    tdc_zero  = Int32Col(shape=(cf.n_view))
    tdc_min   = Int32Col(shape=(cf.n_view))

    x = Float64Col()
    y = Float64Col()
    z = Float64Col()

    d_bary_max = Float64Col()
    d_track_3D = Float64Col()
    d_track_2D = Float64Col()

    n_veto = Int32Col(shape=(cf.n_view))

    timestamp  = Float64Col()
    cluster_ID = Int32Col()
    Z_light    = Float64Col()

class PDS_Peak(IsDescription):

    event   = UInt32Col()
    trigger = UInt32Col()

    ID = UInt32Col()
    glob_ch = UInt32Col()
    channel = UInt32Col()
    start   = Int32Col()
    stop    = Int32Col()
    
    max_t   = Int32Col()
    charge  = Float64Col()
    max_adc = Float64Col()

    cluster_ID = Int32Col()
    
    
class PDS_Cluster(IsDescription):
    event   = UInt32Col()
    trigger = UInt32Col()

    ID = UInt32Col()

    size = UInt32Col()

    start_t = Int32Col()
    stop_t  = Int32Col()

    timestamp = Float64Col()
    match_trk3D  = Int32Col(cf.n_drift_volumes)
    match_single = Int32Col()

class Debug(IsDescription):
    event   = UInt32Col()
    trigger = UInt32Col()

    read_data      = Float64Col(cf.n_module)
    ped_1          = Float64Col(cf.n_module)
    fft            = Float64Col(cf.n_module)
    ped_2          = Float64Col(cf.n_module)
    cnr            = Float64Col(cf.n_module)
    ped_3          = Float64Col(cf.n_module)
    hit_f          = Float64Col(cf.n_module)
    trk2D_1        = Float64Col(cf.n_module)
    trk2D_2        = Float64Col(cf.n_module)
    stitch2D       = Float64Col(cf.n_module)
    #stitch2D_gap_1 = Float64Col()
    stitch2D_gap_2 = Float64Col()
    trk3D          = Float64Col(cf.n_module)
    stitch3D       = Float64Col()
    single         = Float64Col(cf.n_module)
    output         = Float64Col()
    mem_mod        = Float64Col(cf.n_module)
    time_mod       = Float64Col(cf.n_module)
    mem_tot        = Float64Col()
    time_tot        = Float64Col()

def create_tables(h5file):
    table = h5file.create_table("/", 'infos', Infos, 'Infos')

    table = h5file.create_table("/", 'event', Event, "Event")
    table = h5file.create_table("/", 'pedestals', Pedestal, 'Pedestals')
    table = h5file.create_table("/", 'noisestudy', NoiseStudy, 'Noise Study')
    #table = h5file.create_table("/", 'fft', FFT, 'fft')
    #table = h5file.create_table("/", 'corr', Corr, 'corr')
    table = h5file.create_table("/", 'hits', Hits, 'Hits')
    table = h5file.create_table("/", 'single_hits', SingleHits, 'Single Hits')
    table = h5file.create_table("/", 'ghost', Ghost, 'Ghost Tracks')
    
    table = h5file.create_table("/", 'tracks2d', Tracks2D, 'Tracks2D')
    for i in range(cf.n_view):

        t = h5file.create_vlarray("/", 'trk2d_v'+str(i), Float64Atom(shape=(4)), "2D Path V"+str(i)+" (x, z, q, ID)")

    table = h5file.create_table("/", 'tracks3d', Tracks3D, 'Tracks3D')
    for i in range(cf.n_view):
        t = h5file.create_vlarray("/", 'trk3d_v'+str(i), Float64Atom(shape=(6)), "3D Path V"+str(i)+" (x, y, z, dq, ds, ID)")

    t = h5file.create_vlarray("/", 'ghost_tracks', Float64Atom(shape=(6)), "3D Path (x, y, z, dq, ds, ID)")



def create_table_debug(h5file):
    table = h5file.create_table("/", "debug", Debug, 'Debug')    


def create_tables_pulsing(h5file):
    table = h5file.create_table("/", 'infos', Infos, 'Infos')
    table = h5file.create_table("/", 'chmap', ChanMap, "ChanMap")
    table = h5file.create_table("/", "pulse", Pulse, "Pulse")
    table = h5file.create_table("/", 'pedestals', Pedestal, 'Pedestals')
    table = h5file.create_table("/", 'event', Event, "Event")

    table = h5file.create_table('/','waveform',Waveform,'Waveform')


    t = h5file.create_vlarray("/", 'pos_pulse', Float32Atom(shape=(10)), "Positive Pulses (start, tmax, vmax, A, Aerr, tau, tauerr, area, fit_area, rchi2)")
    t = h5file.create_vlarray("/", 'neg_pulse', Float32Atom(shape=(10)), "Negative Pulses (start, tmin, vmin, A, Aerr, tau, tauerr, area, fit_area, rchi2)")

def create_tables_pds(h5file):
    table = h5file.create_table("/", 'pds_infos', PDSInfos, 'PDSInfos')
    table = h5file.create_table("/", 'pds_event', PDSEvent, 'PDSEvent')
    table = h5file.create_table("/", 'pds_pedestals', PDSPedestal, 'PDSPedestals')
    table = h5file.create_table("/", "pds_peaks", PDS_Peak, "PDSPeak")
    table = h5file.create_table("/", "pds_clusters", PDS_Cluster, "PDSCluster")

    t = h5file.create_vlarray("/", 'pds_peakID_clusters', Float32Atom(shape=(1)), "Peak IDs")
    t = h5file.create_vlarray("/", 'charge_pds_match', Float32Atom(shape=(9)), "(distance, SiPM strip nb, x_impact, y_impact, z_impact, isInside, x_closest, y_closest, z_closest)")
    
    
def store_run_infos(h5file, run, sub, nevent, time):
    inf = h5file.root.infos.row
    inf['run']           = run
    inf['sub']           = sub
    inf['elec']          = cf.elec[0]
    inf['n_evt']         = nevent
    inf['process_date']  = time
    inf['n_channels']    = cf.n_tot_channels
    inf['sampling']      = [cf.sampling[i] for i in cf.module_used]
    inf['n_samples']     = [cf.n_sample[i] for i in cf.module_used]
    inf['n_view']        = cf.n_view
    inf['view_nchan']    = cf.view_nchan
    inf['e_drift']       = [cf.e_drift[i] for i in cf.module_used]
    inf['t_lar']         = [cf.LAr_temperature[i] for i in cf.module_used]
    inf.append()


def store_event(h5file):
    evt = h5file.root.event.row
    evt['trigger_nb']      = dc.evt_list[-1].trigger_nb
    evt['trigger_type']    = dc.evt_list[-1].trig_type
    evt['time_s']          = dc.evt_list[-1].time_s
    evt['time_ns']         = dc.evt_list[-1].time_ns
    evt['charge_time']     = dc.evt_list[-1].charge_time
    evt['pds_stream_time'] = dc.evt_list[-1].pds_stream_time
    evt['pds_trig_time']   = dc.evt_list[-1].pds_trig_time
    evt['n_sample']        = [cf.n_sample[i] for i in cf.module_used]
    evt['n_hits']          = dc.evt_list[-1].n_hits
    evt['n_tracks2D']      = dc.evt_list[-1].n_tracks2D
    evt['n_tracks3D']      = dc.evt_list[-1].n_tracks3D
    evt['n_single_hits']   = dc.evt_list[-1].n_single_hits
    evt['n_ghosts']        = dc.evt_list[-1].n_ghosts

    evt.append()


def store_pedestals(h5file):
    ped = h5file.root.pedestals.row

    if(cf.n_module != cf.n_module_used):
        print("sorry, cannot store noise related info atm when looking at a subset of the detector")
        return
    
    ped['raw_mean']   = chmap.arange_in_glob_channels(list(itt.chain.from_iterable(x.ped_mean for x in dc.evt_list[-1].noise_raw)))
    ped['raw_rms']    = chmap.arange_in_glob_channels(list(itt.chain.from_iterable(x.ped_rms for x in dc.evt_list[-1].noise_raw)))
    
    ped['filt_mean']  = chmap.arange_in_glob_channels(list(itt.chain.from_iterable(x.ped_mean for x in dc.evt_list[-1].noise_filt))) #dc.evt_list[-1].noise_filt.ped_mean
    ped['filt_rms']   = chmap.arange_in_glob_channels(list(itt.chain.from_iterable(x.ped_rms for x in dc.evt_list[-1].noise_filt)))

    ped.append()

def store_pds_pedestals(h5file):
    ped = h5file.root.pds_pedestals.row
    ped['raw_mean']   = dc.evt_list[-1].noise_pds_raw.ped_mean
    ped['raw_rms']    = dc.evt_list[-1].noise_pds_raw.ped_rms
    ped['filt_mean']   = dc.evt_list[-1].noise_pds_filt.ped_mean
    ped['filt_rms']    = dc.evt_list[-1].noise_pds_filt.ped_rms
    ped.append()

def store_noisestudy(h5file):
    ped = h5file.root.noisestudy.row

    if(dc.evt_list[-1].noise_study==None):
        return

    ped['delta_mean'] = chmap.arange_in_glob_channels(list(itt.chain.from_iterable(x.ped_mean for x in dc.evt_list[-1].noise_study)))

    ped['rms']  = chmap.arange_in_glob_channels(list(itt.chain.from_iterable(x.ped_rms for x in dc.evt_list[-1].noise_study)))
    ped.append()

'''
def store_fft(h5file, ps):
    #print('fft shapes: ', ps[0].shape, " ", ps[1].shape, " ", ps[2].shape, " ", ps[3].shape)
    print("WARNING !!! FFT PS PLOTS TAKE A LOT OF SPACE !!!")

    fft = h5file.root.fft.row
    if(ps[0].shape[1] == 3937):
        fft['ps_0']   = ps[0]
    if(ps[1].shape[1] == 3937):
        fft['ps_1']   = ps[1]
    if(ps[2].shape[1] == 4033):
        fft['ps_2']   = ps[2]
    if(ps[3].shape[1] == 4033):
        fft['ps_3']   = ps[3]
    fft.append()
'''
'''
def store_corr(h5file, corr):
    #print('STORE corr ', len(corr))    
    print("WARNING !! CORRELATION PLOTS TAKE A LOT OF SPACE !!!")
    cnr = h5file.root.corr.row
    cnr['corr_0']   = corr[0]
    cnr['corr_1']   = corr[1]
    cnr['corr_2']   = corr[2]
    cnr['corr_3']   = corr[3]
    cnr.append()
'''

def store_hits(h5file):
    hit = h5file.root.hits.row

    for ih in dc.hits_list:
       hit['event']   = dc.evt_list[-1].evt_nb
       hit['trigger'] = dc.evt_list[-1].trigger_nb
       hit['ID']= ih.ID

       hit['daq_channel'] = ih.daq_channel

       hit['module']  = ih.module
       hit['view']    = ih.view
       hit['channel'] = ih.channel

       hit['is_collection'] = ih.signal == "Collection"
       
       hit['tdc_start']  = ih.start
       hit['tdc_stop']   = ih.stop

       hit['tdc_max']  = ih.max_t
       hit['tdc_min']  = ih.min_t
       hit['tdc_zero'] = ih.zero_t

       hit['z']       = ih.Z
       hit['x']       = ih.X
       
       
       hit['fC_max']  = ih.max_fC
       hit['fC_min']  = ih.min_fC
       

       hit['charge_pos'] = ih.charge_pos
       hit['charge_neg'] = ih.charge_neg


       hit['is_free'] = ih.is_free
       


       hit['match_3D'] = ih.match_3D
       hit['match_2D'] = ih.match_2D
       hit['match_dray'] = ih.match_dray
       hit['match_ghost'] = ih.match_ghost
       hit['match_sh'] = ih.match_sh

       hit.append()

def store_single_hits(h5file):
    sh = h5file.root.single_hits.row
    
    for it in dc.single_hits_list:
        sh['event'] = dc.evt_list[-1].evt_nb
        sh['trigger'] = dc.evt_list[-1].trigger_nb 

        sh['n_hits'] = it.n_hits

        sh['ID'] = it.ID_SH
        sh['module'] = it.module

        id_np = np.zeros((cf.n_view, dc.reco['single_hit']['max_per_view']), dtype=int)
        id_np.fill(-1)
        for i,j in enumerate(it.IDs):
            id_np[i][0:len(j)] = j

        sh['hit_IDs'] = id_np
        sh['charge_pos'] =  it.charge_pos
        sh['charge_neg'] =  it.charge_neg

        sh['tdc_start'] =  it.start
        sh['tdc_stop'] =  it.stop

        sh['tdc_max'] =  it.max_t
        sh['tdc_zero'] =  it.min_t
        sh['tdc_min'] =  it.min_t

        sh['x'] =  it.X
        sh['y'] =  it.Y
        sh['z'] =  it.Z

        sh['d_bary_max'] = it.d_bary_max
        sh['d_track_3D'] = it.d_track_3D
        sh['d_track_2D'] = it.d_track_2D
        sh['n_veto'] = it.n_veto

        #sh['charge_extend'] = it.charge_extend
        #sh['charge_extend_pos'] = it.charge_extend_pos
        #sh['charge_extend_neg'] = it.charge_extend_neg

        sh['timestamp']  = it.timestamp
        sh['cluster_ID'] = it.match_pds_cluster
        sh['Z_light']    = it.Z_from_light

        sh.append()

def store_tracks2D(h5file):
    t2d = h5file.root.tracks2d.row
    vl_h = [h5file.get_node('/trk2d_v'+str(i)) for i in range(cf.n_view)]


    for it in dc.tracks2D_list:
       t2d['event'] = dc.evt_list[-1].evt_nb
       t2d['trigger'] = dc.evt_list[-1].trigger_nb

       t2d['ID'] = it.trackID

       t2d['match_3D'] = it.match_3D
       t2d['matched']  = it.matched


       t2d['view'] = it.view
       t2d['pos_ini'] = it.path[0][0]
       t2d['pos_end'] = it.path[-1][0]
       t2d['z_ini'] = it.path[0][1]
       t2d['z_end'] = it.path[-1][1]
       t2d['n_hits_track'] = it.n_hits
       t2d['n_hits_dray'] = it.n_hits_dray

       t2d['chi2_fwd'] = it.chi2_fwd
       t2d['chi2_bwd'] = it.chi2_bkwd

       t2d['slope_ini'] = it.ini_slope
       t2d['slope_end'] = it.end_slope
       t2d['slope_ini_err'] = it.ini_slope_err
       t2d['slope_end_err'] = it.end_slope_err

       t2d['len_straight'] = it.len_straight
       t2d['len_path'] = it.len_path

       t2d['track_total_charge'] = it.tot_charge
       t2d['dray_total_charge'] = it.dray_charge

       pts = [[p[0], p[1], q, r] for p,q,r in zip(it.path,it.dQ,it.hits_ID)]
       vl_h[it.view].append(pts)
       t2d.append()



def store_tracks3D(h5file):
    t3d = h5file.root.tracks3d.row
    vl_h = [h5file.get_node('/trk3d_v'+str(i)) for i in range(cf.n_view)]

    for it in dc.tracks3D_list:
       t3d['event'] = dc.evt_list[-1].evt_nb
       t3d['trigger'] = dc.evt_list[-1].trigger_nb

       t3d['ID'] = it.ID_3D
       t3d['matched_2D'] = it.match_ID#[it.match_ID[i] for i in range(cf.n_view)]
       t3d['n_matched'] = it.n_matched#sum([it.match_ID[i] >= 0 for i in range(cf.n_view)])

       t3d['module_ini'] = it.module_ini
       t3d['module_end'] = it.module_end
       
       t3d['x_ini'] = it.ini_x
       t3d['y_ini'] = it.ini_y
       t3d['z_ini'] = it.ini_z
       t3d['t_ini'] = it.ini_time
       t3d['x_end'] = it.end_x
       t3d['y_end'] = it.end_y
       t3d['z_end'] = it.end_z
       t3d['t_end'] = it.end_time
       t3d['chi2']  = it.chi2


       t3d['z_ini_overlap'] = it.ini_z_overlap
       t3d['z_end_overlap'] = it.end_z_overlap

       t3d['theta_ini'] = it.ini_theta
       t3d['theta_end'] = it.end_theta
       t3d['phi_ini']   = it.ini_phi
       t3d['phi_end']   = it.end_phi

       t3d['n_hits']        = it.n_hits
       t3d['len_straight']  = it.len_straight
       t3d['len_path']      = it.len_path
       t3d['total_charge']  = it.tot_charge

       t3d['z0_corr']   = it.z0_corr
       t3d['t0_corr']   = it.t0_corr

       t3d['d_match']  = it.d_match

       t3d['timestamp'] = it.timestamp
       t3d['cluster_ID'] = it.match_pds_cluster


       
       t3d['is_cathode_crosser'] = it.is_cathode_crosser
       t3d['cathode_crossing'] = it.cathode_crossing
       t3d['cathode_crosser_ID'] = it.cathode_crosser_ID
       t3d['cathode_crossing_trk_end'] = it.cathode_crossing_trk_end
              
       t3d['is_module_crosser'] = it.is_module_crosser
       t3d['module_crossing'] = it.module_crossing

       t3d['is_anode_crosser'] = it.is_anode_crosser
       t3d['exit_point'] = it.exit_point
       t3d['exit_trk_end'] = it.exit_trk_end

       for i in range(cf.n_view):
           pts = [[p[0], p[1], p[2], q, s, r] for p,q,s,r in zip(it.path[i], it.dQ[i], it.ds[i], it.hits_ID[i])]
           vl_h[i].append(pts)
       t3d.append()


def store_ghost(h5file):
    tgh = h5file.root.ghost.row
    vl_h = h5file.get_node('/ghost_tracks')

    for it in dc.ghost_list:
       tgh['event'] = dc.evt_list[-1].evt_nb
       tgh['trigger'] = dc.evt_list[-1].trigger_nb

       tgh['match_3D'] = it.trk3D_ID
       tgh['match_2D'] = it.trk2D_ID

       tgh['x_anode'] = it.anode_x
       tgh['y_anode'] = it.anode_y
       tgh['z_anode'] = it.anode_z
       
       tgh['theta'] = it.theta
       tgh['phi'] = it.phi
       
       tgh['n_hits'] = it.n_hits
       tgh['total_ghost_charge'] = it.ghost_charge
       tgh['total_track_charge'] = it.trk_charge
       
       tgh['z0_corr'] = it.z0_corr
       tgh['t0_corr'] = it.t0_corr
       
       tgh['d_min'] = it.min_dist
       
       pts = [[p[0], p[1], p[2], q, s, r] for p,q,s, r in zip(it.path, it.dQ, it.ds, it.hits_ID)]
       vl_h.append(pts)
       tgh.append()


def store_hits_3d(h5file):
    hit = h5file.root.hits3D.row

    for ih in dc.hits_list:
        if(ih.has_3D == False):
            continue
        
        hit['event']   = dc.evt_list[-1].evt_nb
        hit['trigger'] = dc.evt_list[-1].trigger_nb
        hit['ID']= ih.ID
        
        hit['daq_channel'] = ih.daq_channel
        hit['module']  = ih.module
        
        hit['view']    = ih.view
        hit['channel'] = ih.channel


        hit['x']        = ih.x_3D
        hit['y']        = ih.y_3D
        hit['z']        = ih.Z
        hit['match_ID'] = ih.ID_match_3D
                
        hit['n_cluster'] = [dc.hits_cluster_list[dc.hits_list[h-dc.n_tot_hits].cluster-dc.n_tot_hits_clusters].n_hits for h in ih.ID_match_3D]
            
        hit.append()
       
def store_avf_wvf(h5file):
    twvf = h5file.root.waveform.row
    

    for i in range(cf.n_tot_channels):
        twvf['view'] = dc.chmap[i].view
        twvf['channel'] = dc.chmap[i].vchan
        twvf['daq_channel'] = i


        w_pos = np.asarray(dc.wvf_pos[i])
        if(len(w_pos)>0):
            twvf['pos_mean'] = np.mean(w_pos, axis=0)
            twvf['pos_std'] = np.std(w_pos, axis=0)
        else:
            twvf['pos_mean'] = [-1 for x in range(60)]
            twvf['pos_std']  = [-1 for x in range(60)]

        w_neg = np.asarray(dc.wvf_neg[i])
        if(len(w_neg)>0):
            twvf['neg_mean'] = np.mean(w_neg, axis=0)
            twvf['neg_std'] = np.std(w_neg, axis=0)
        else:
            twvf['neg_mean'] = [-1 for x in range(60)]
            twvf['neg_std'] = [-1 for x in range(60)]

        twvf.append()

def store_pulse(h5file):
    tpul = h5file.root.pulse.row
    vl_pos = h5file.get_node('/pos_pulse')
    vl_neg = h5file.get_node('/neg_pulse')

    for p in dc.pulse_fit_res:
        tpul['event'] = dc.evt_list[-1].evt_nb 
        tpul['trigger'] = dc.evt_list[-1].trigger_nb     
        tpul['view'] = p.view
        tpul['channel'] = p.channel
        tpul['daq_channel'] = p.daq_channel
        tpul['n_pulse_pos'] = p.n_pulse_pos
        tpul['n_pulse_neg'] = p.n_pulse_neg

        vl_pos.append(p.fit_pos)
        vl_neg.append(p.fit_neg)
    
        tpul.append()
    

def store_pds_infos(h5file, run, sub, nevent, time):
    inf = h5file.root.pds_infos.row
    inf['run']           = run
    inf['sub']           = sub
    inf['elec']          = cf.elec[0]
    inf['n_evt']         = nevent
    inf['process_date']  = time
    inf['n_channels']    = cf.n_pds_tot_channels
    inf['sampling']      = cf.pds_sampling
    inf['n_samples']     = max([cf.n_pds_stream_sample, cf.n_pds_trig_sample])
    inf['e_drift']       = [cf.e_drift[i] for i in cf.module_used]
    inf['t_lar']         = [cf.LAr_temperature[i] for i in cf.module_used]
    inf.append()

def store_pds_event(h5file):
    evt = h5file.root.pds_event.row

    evt['event']       = dc.evt_list[-1].evt_nb
    evt['trigger_nb']  = dc.evt_list[-1].trigger_nb
    evt['stream_time'] = dc.evt_list[-1].pds_stream_time
    evt['trig_time']   = dc.evt_list[-1].pds_trig_time
    evt['n_sample']    = max([cf.n_pds_stream_sample, cf.n_pds_trig_sample])
    evt['n_peak']      = dc.evt_list[-1].n_pds_peaks
    evt['n_cluster']   = dc.evt_list[-1].n_pds_clusters

    evt.append()

def store_pds_peak(h5file):
    pds = h5file.root.pds_peaks.row
    
    for p in dc.pds_peak_list:
        pds['event']   = dc.evt_list[-1].evt_nb
        pds['trigger'] = dc.evt_list[-1].trigger_nb

        pds['ID'] = p.ID
        
        pds['glob_ch'] = p.glob_ch
        pds['channel'] = p.channel
        pds['start']   = p.start
        pds['stop']    = p.stop
        
        pds['max_t']   = p.max_t
        pds['charge']  = p.charge
        pds['max_adc'] = p.max_adc
        
        pds['cluster_ID'] = p.cluster_ID
        pds.append()


def store_pds_cluster(h5file):
    clu = h5file.root.pds_clusters.row

    vl_ids = h5file.get_node('/pds_peakID_clusters')
    vl_match = h5file.get_node('/charge_pds_match')

    
    for c in dc.pds_cluster_list:
        clu['event']   = dc.evt_list[-1].evt_nb
        clu['trigger'] = dc.evt_list[-1].trigger_nb

        clu['ID'] = c.ID
        clu['size'] = c.size
        clu['start_t'] = c.t_start
        clu['stop_t'] = c.t_stop
        clu['timestamp'] = c.timestamp
        clu['match_trk3D'] = c.match_trk3D
        clu['match_single'] = c.match_single

        vl_ids.append([[i] for i in c.peak_IDs])

        pts = [[d, idx, p[0], p[1], p[2], tf, h[0], h[1], h[2]] for d, idx, p, tf,h in zip(c.dist_closest_strip, c.id_closest_strip, c.point_impact, c.point_closest_above, c.point_closest)]
        vl_match.append(pts)
        clu.append()

def store_debug(h5file, debug):
    deb  = h5file.root.debug.row
    deb['event']   = dc.evt_list[-1].evt_nb
    deb['trigger'] = dc.evt_list[-1].trigger_nb



    deb['read_data']      = debug.read_data
    deb['ped_1']          = debug.ped_1
    deb['fft']            = debug.fft
    deb['ped_2']          = debug.ped_2
    deb['cnr']            = debug.cnr
    deb['ped_3']          = debug.ped_3
    deb['hit_f']          = debug.hit_f
    deb['trk2D_1']        = debug.trk2D_1
    deb['trk2D_2']        = debug.trk2D_2
    deb['stitch2D']       = debug.stitch2D
    #deb['stitch2D_gap_1'] = debug.stitch2D_gap_1
    #deb['stitch2D_gap_2'] = debug.stitch2D_gap_2
    deb['trk3D']        = debug.trk3D
    #deb['trk3D_2']        = debug.trk3D_2
    deb['stitch3D']       = debug.stitch3D
    deb['single']         = debug.single
    deb['output']         = debug.output
    deb['mem_mod']        = debug.memory_mod
    deb['time_mod']       = debug.time_mod
    deb['mem_tot']        = debug.memory_tot
    deb['time_tot']       = debug.time_tot

    deb.append()
    
        
def dictToGroup(f, parent, groupname, dictin, force=False, recursive=True):
    """
    From https://stackoverflow.com/questions/18071075/saving-dictionaries-to-file-numpy-and-python-2-3-friendly
    
    Take a dict, shove it into a PyTables HDF5 file as a group. Each item in
    the dict must have a type and shape compatible with PyTables Array.

    If 'force == True', any existing child group of the parent node with the
    same name as the new group will be overwritten.

    If 'recursive == True' (default), new groups will be created recursively
    for any items in the dict that are also dicts.
    """
    try:
        g = f.create_group(parent, groupname)
    except tables.NodeError as ne:
        if force:
            pathstr = parent._v_pathname + '/' + groupname
            f.removeNode(pathstr, recursive=True)
            g = f.create_group(parent, groupname)
        else:
            raise ne
    for key, item in dictin.items():
        if(key=="plot" or "store"):
            continue
        if isinstance(item, dict):
            if recursive:
                dictToGroup(f, g, key, item, recursive=True)
        else:
            if item is None:
                item = '_None'
            f.create_array(g, key, item)
    return g

def save_reco_param(h5file):
    g = dictToGroup(h5file, "/", "reco", dc.reco, force=False, recursive=True)
