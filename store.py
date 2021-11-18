import config as cf
from tables import *
import numpy as np
import data_containers as dc



class Infos(IsDescription):
    run          = UInt16Col()
    sub          = UInt16Col()
    elec         = StringCol(3)
    n_evt        = UInt8Col()
    process_date = UInt32Col()
    n_channels   = UInt16Col()
    sampling     = Float32Col()
    n_samples    = Float32Col()
    n_view       = Float32Col()
    view_nchan   = Float32Col(shape=(cf.n_view))



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


def create_tables(h5file):
    table = h5file.create_table("/", 'infos', Infos, 'Infos')
    table = h5file.create_table("/", 'chmap', ChanMap, "ChanMap")
    table = h5file.create_table("/", 'event', Event, "Event")
    table = h5file.create_table("/", 'pedestals', Pedestal, 'Pedestals')
    table = h5file.create_table("/", 'fft', FFT, 'fft')
    
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
    fft = h5file.root.fft.row
    fft['ps']   = ps
    fft.append()
