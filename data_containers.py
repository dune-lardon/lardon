import numpy as np
import config as cf
import time

chmap = []
evt_list = []
hits_list = []
tracks2D_list = []
tracks3D_list = []


data_daq = np.zeros((cf.n_tot_channels, cf.n_sample), dtype=np.float32) #view, vchan
mask_daq  = np.ones((cf.n_tot_channels, cf.n_sample), dtype=bool)
alive_chan = np.ones((cf.n_tot_channels, cf.n_sample), dtype=bool)


data = np.zeros((cf.n_view, max(cf.view_nchan), cf.n_sample), dtype=np.float32) #view, vchan
"""
the mask will be used to differentiate background (True for noise processing) from signal (False for noise processing)
at first everything is considered background (all at True)
"""
mask = np.ones((cf.n_view, max(cf.view_nchan), cf.n_sample), dtype=bool)

"""
alive_chan mask intends to not take into account broken channels
True : not broken
False : broken
"""



def reset_event():
    data[:,:,:] = 0.
    data_daq[:,:] = 0.
    mask[:,:,:] = True

    hits_list.clear()
    tracks2D_list.clear()
    tracks3D_list.clear()
    evt_list.clear()

class channel:
    def __init__(self, daqch, globch, view, vchan):
        self.daqch = daqch
        self.globch = globch
        self.view = view
        self.vchan = vchan
        
    def get_ana_chan(self):
        return self.view, self.vchan
        
    def get_daqch(self):
        return self.daqch
        
    def get_globch(self):
        return self.globch

class noise:
    def __init__(self, ped, rms):
        self.ped_mean = ped
        self.ped_rms  = rms

class event:
    def __init__(self, elec, run, sub, evt, trigger, t_s, t_ns):
        self.elec = elec
        self.run_nb  = run
        self.sub  = sub
        self.evt_nb  = evt
        self.trigger_nb = trigger
        self.time_s = t_s
        self.time_ns = t_ns
        self.n_hits = np.zeros((cf.n_view), dtype=int)
        self.n_tracks2D = np.zeros((cf.n_view), dtype=int)
        self.n_tracks3D = 0

    
    def set_noise_raw(self, noise):
        self.noise_raw = noise

    def set_noise_filt(self, noise):
        self.noise_filt = noise

    def dump(self):
        print("RUN ",self.run_nb, " of ", self.elec, " EVENT ", self.evt_nb, " / ", self.trigger_nb,)
        print("Taken at ", time.ctime(self.time_s), " + ", self.time_ns, " ns ")




""" not sure this is useful """
class run:
    def __init__(self, period, elec):
        self.period = period
        self.elec = elec

    def set_elec(n_view, view_angle, view_name, view_type, n_tot_channels, view_nchan, calib):
        """ to add when we know : channel pitches """
        self.n_view = n_view
        self.view_angle = view_angle
        self.view_name = view_name
        self.view_type = view_type
        self.n_tot_channels = n_tot_channels #nb of channels written in the files
        self.view_nchan = view_nchan
        self.ADC_to_fC = calib

    def set_run_infos(self, run_nb, n_events, t_start, delta_t, sampling, n_sample):
        self.run_nb = run_nb
        
        """ separate total nb of events the file has with nb of events processed ?"""
        self.n_events = n_events
        self.t_start = t_start
        self.delta_t = delta_t

        self.sampling = sampling
        self.n_sample = n_sample

    def dump_run(self):
        print(' * * Run Informations * *')
        print('Data Taking period ', self.period)
        print('Electronics is ', self.elec, ', calibration is ', self.calib, ' ADC per fC')
        print('Number of channels written in the raw file ', self.n_tot_channels)
        print('Detector has ', self.n_view, ' views : ')
        for i in range(self.n_view):
            print('\t View ', i, ' is ', self.view_name[i], '(',self.view_type[i],') at an angle of ', self.view_angle[i], ' with ', self.view_nchan, ' channels per view')
        print('Reading run ', self.run_nb, ' with ', self.n_events, ' events')
        print('Taken on ', self.t_start, ' for ', self.delta_t, ' seconds')
        print('At a sampling of ', self.sampling, ' MHz with ', self.n_sample, ' time samples')
        print('\n')
        
