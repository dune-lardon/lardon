import config as cf
from tables import *
import numpy as np
import data_containers as dc



class Infos(IsDescription):
    run          = UInt16Col()
    sub          = StringCol(6)
    elec         = StringCol(3)
    n_evt        = UInt8Col()
    process_date = UInt32Col()
    n_channels   = UInt16Col()
    sampling     = Float32Col()
    n_samples    = Float32Col()
    n_view       = UInt8Col()
    view_nchan   = Float32Col(shape=(cf.n_view))
    e_drift      = Float32Col()


class ChanMap(IsDescription):
    view    = Int8Col(shape=(cf.n_tot_channels))
    channel = Int16Col(shape=(cf.n_tot_channels))

class Event(IsDescription):
    trigger_nb    = UInt32Col()
    time_s        = UInt32Col()
    time_ns       = UInt32Col()
    n_hits        = UInt32Col(shape=(cf.n_view))
    n_tracks2D    = UInt32Col(shape=(cf.n_view))
    n_tracks3D    = UInt32Col()


class Pedestal(IsDescription):
    raw_mean   = Float32Col(shape=(cf.n_tot_channels))
    raw_rms    = Float32Col(shape=(cf.n_tot_channels))
    filt_mean  = Float32Col(shape=(cf.n_tot_channels))
    filt_rms   = Float32Col(shape=(cf.n_tot_channels))

class FFT(IsDescription):
    ps = Float32Col(shape=(cf.n_tot_channels, int(cf.n_sample/2)+1))


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


class Hits(IsDescription):
    event   = UInt32Col()
    trigger = UInt32Col()
    ID = UInt32Col()


    view        = UInt8Col()
    channel     = UInt16Col()
    daq_channel = UInt16Col()

    is_collection  = BoolCol()

    tdc_max  = UInt16Col()
    tdc_min  = UInt16Col()
    tdc_zero = UInt16Col()

    z       = Float32Col()
    x       = Float32Col()


    fC_max  = Float32Col()
    fC_min  = Float32Col()

    charge_pos = Float32Col()
    charge_neg = Float32Col()



class Tracks2D(IsDescription):
    event   = UInt32Col()
    trigger = UInt32Col()

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

    matched = BoolCol(shape=(cf.n_view))
    n_matched = UInt32Col()

    x_ini   = Float32Col()
    y_ini   = Float32Col()
    z_ini   = Float32Col()
    x_end   = Float32Col()
    y_end   = Float32Col()
    z_end   = Float32Col()
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


def create_tables(h5file):
    table = h5file.create_table("/", 'infos', Infos, 'Infos')
    table = h5file.create_table("/", 'chmap', ChanMap, "ChanMap")
    table = h5file.create_table("/", 'event', Event, "Event")
    table = h5file.create_table("/", 'pedestals', Pedestal, 'Pedestals')
    table = h5file.create_table("/", 'fft', FFT, 'fft')
    table = h5file.create_table("/", 'hits', Hits, 'Hits')

    table = h5file.create_table("/", 'tracks2d', Tracks2D, 'Tracks2D')
    for i in range(cf.n_view):

        t = h5file.create_vlarray("/", 'trk2d_v'+str(i), Float64Atom(shape=(4)), "2D Path V"+str(i)+" (x, z, q, ID)")

    table = h5file.create_table("/", 'tracks3d', Tracks3D, 'Tracks3D')
    for i in range(cf.n_view):
        t = h5file.create_vlarray("/", 'trk3d_v'+str(i), Float64Atom(shape=(6)), "3D Path V"+str(i)+" (x, y, z, dq, ds, ID)")



def create_tables_pulsing(h5file):
    table = h5file.create_table("/", 'infos', Infos, 'Infos')
    table = h5file.create_table("/", 'chmap', ChanMap, "ChanMap")
    table = h5file.create_table("/", "pulse", Pulse, "Pulse")
    table = h5file.create_table("/", 'pedestals', Pedestal, 'Pedestals')
    table = h5file.create_table("/", 'event', Event, "Event")

    table = h5file.create_table('/','waveform',Waveform,'Waveform')


    t = h5file.create_vlarray("/", 'pos_pulse', Float32Atom(shape=(8)), "Positive Pulses (start, tmax, vmax, A, tau, area, fit_area, chi2)")
    t = h5file.create_vlarray("/", 'neg_pulse', Float32Atom(shape=(8)), "Negative Pulses (start, tmin, vmin, A, tau, area, fit_area, chi2)")


def store_run_infos(h5file, run, sub, elec, nevent, time):
    inf = h5file.root.infos.row
    inf['run']           = run
    inf['sub']           = sub
    inf['elec']          = elec
    inf['n_evt']         = nevent
    inf['process_date']  = time
    inf['n_channels']    = cf.n_tot_channels
    inf['sampling']      = cf.sampling
    inf['n_samples']     = cf.n_sample
    inf['n_view']        = cf.n_view
    inf['view_nchan']    = cf.view_nchan
    inf['e_drift']       = cf.e_drift
    inf.append()


def store_chan_map(h5file):
    chm = h5file.root.chmap.row

    chm['view']    = [-1 if(x.view == cf.n_view) else x.view for x in dc.chmap]
    chm['channel'] = [x.vchan for x in dc.chmap]

    chm.append()

def store_event(h5file):
    evt = h5file.root.event.row
    evt['trigger_nb'] = dc.evt_list[-1].trigger_nb
    evt['time_s']     = dc.evt_list[-1].time_s
    evt['time_ns']    = dc.evt_list[-1].time_ns
    evt['n_hits']     = dc.evt_list[-1].n_hits
    evt['n_tracks2D'] = dc.evt_list[-1].n_tracks2D
    evt['n_tracks3D'] = dc.evt_list[-1].n_tracks3D

    evt.append()


def store_pedestals(h5file):
    ped = h5file.root.pedestals.row
    ped['raw_mean']   = dc.evt_list[-1].noise_raw.ped_mean
    ped['raw_rms']    = dc.evt_list[-1].noise_raw.ped_rms
    ped['filt_mean']  = dc.evt_list[-1].noise_filt.ped_mean
    ped['filt_rms']   = dc.evt_list[-1].noise_filt.ped_rms
    ped.append()

def store_fft(h5file, ps):
    print("WARNING THIS TAKES A LOT OF SPACE !!!")
    fft = h5file.root.fft.row
    fft['ps']   = ps
    fft.append()


def store_hits(h5file):
    hit = h5file.root.hits.row

    for ih in dc.hits_list:
       hit['event'] = dc.evt_list[-1].evt_nb
       hit['trigger'] = dc.evt_list[-1].trigger_nb
       hit['ID']= ih.ID

       hit['daq_channel'] = ih.daq_channel
       hit['view']    = ih.view
       hit['channel'] = ih.channel

       hit['is_collection'] = ih.signal == "Collection"
       
       hit['tdc_max']  = ih.max_t
       hit['tdc_min']  = ih.min_t
       hit['tdc_zero'] = ih.zero_t

       hit['z']       = ih.Z
       hit['x']       = ih.X
       
       
       hit['fC_max']  = ih.max_fC
       hit['fC_min']  = ih.min_fC
       

       hit['charge_pos'] = ih.charge_pos
       hit['charge_neg'] = ih.charge_neg


       hit.append()




def store_tracks2D(h5file):
    t2d = h5file.root.tracks2d.row
    vl_h = [h5file.get_node('/trk2d_v'+str(i)) for i in range(cf.n_view)]


    for it in dc.tracks2D_list:
       t2d['event'] = dc.evt_list[-1].evt_nb
       t2d['trigger'] = dc.evt_list[-1].trigger_nb

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

       t3d['matched'] = [it.match_ID[i] >= 0 for i in range(cf.n_view)]
       t3d['n_matched'] = sum([it.match_ID[i] >= 0 for i in range(cf.n_view)])

       t3d['x_ini'] = it.ini_x
       t3d['y_ini'] = it.ini_y
       t3d['z_ini'] = it.ini_z
       t3d['x_end'] = it.end_x
       t3d['y_end'] = it.end_y
       t3d['z_end'] = it.end_z
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

       for i in range(cf.n_view):
           pts = [[p[0], p[1], p[2], q, s, r] for p,q,s,r in zip(it.path[i], it.dQ[i], it.ds[i], it.hits_ID[i])]
           vl_h[i].append(pts)
       t3d.append()




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
    
