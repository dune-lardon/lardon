import lardon.config as cf
import lardon.utils.enum_type as et

import numpy as np
import time
import math
from rtree import index

chmap = []
chmap_daq_pds = []
chmap_pds = []


evt_list = []
hits_list = []
hits_cluster_list = []
tracks2D_list = []
tracks3D_list = []
single_hits_list = []
ghost_list = []
pulse_fit_res = []
pds_peak_list = []
pds_cluster_list = []

wvf_pos = []
wvf_neg = []

data_daq = np.zeros((1,1), dtype=np.float32) #view, vchan

"""
the mask will be used to differentiate background (True for noise processing) from signal (False for noise processing)
at first everything is considered background (all at True)
"""

mask_daq  = np.ones((1,1), dtype=bool)

"""
alive_chan mask intends to not take into account broken channels
True : not broken
False : broken
"""

alive_chan = np.ones(1, dtype=bool)


""" for PDS data in full stream mode """
data_stream_pds = np.zeros((1,1), dtype=np.float32)
mask_stream_pds = np.ones((1,1), dtype=bool)

""" for PDS data in self-trigger mode """
data_trig_pds = np.zeros((1,1), dtype=np.float32)
mask_trig_pds = np.ones((1,1), dtype=bool)





#NB : do not store all modules
data = np.zeros((1,1,1), dtype=np.float32)


n_tot_hits, n_tot_pds_peaks = 0, 0
n_tot_trk2d, n_tot_trk3d = 0, 0
n_tot_ghosts, n_tot_sh = 0, 0
n_tot_pds_clusters, n_tot_hits_clusters = 0, 0

def reset_evt():
    evt_list.clear()
        
    hits_list.clear()
    hits_cluster_list.clear()
    tracks2D_list.clear()
    tracks3D_list.clear()

    single_hits_list.clear()
    ghost_list.clear()
    
    pulse_fit_res.clear()
        
    pds_peak_list.clear()
    pds_cluster_list.clear()

pties = index.Property()
pties.dimension = 4

''' create an rtree index for hits (module, view, X, Z)'''
rtree_hit_idx = index.Index(properties=pties)

    
def reset_containers_trk():
    
    global data_daq, mask_daq, data, alive_chan

    mask_daq[:,:] = True
    data_daq[:,:] = 0.
    data[:,:,:] = 0.
    alive_chan[:] = True
    #reset_rtree()

    
    if(data_daq.shape != (cf.module_nchan[cf.imod], cf.n_sample[cf.imod])):

        new_shape = (cf.module_nchan[cf.imod], cf.n_sample[cf.imod])
        data_daq = np.resize(data_daq, new_shape)
        mask_daq = np.resize(mask_daq, new_shape)
        alive_chan = np.resize(alive_chan, cf.module_nchan[cf.imod])#new_shape)
        data = np.resize(data, (cf.n_view, max(cf.view_nchan), cf.n_sample[cf.imod]))

    


def reset_containers_pds():
    global data_stream_pds, mask_stream_pds, data_trig_pds, mask_trig_pds
    if(data_stream_pds.shape != (cf.n_pds_stream_channels, cf.n_pds_trig_sample)):

        new_shape = (cf.n_pds_stream_channels, cf.n_pds_stream_sample)
        data_stream_pds = np.resize(data_stream_pds, new_shape)
        mask_stream_pds = np.resize(mask_stream_pds, new_shape)


    data_stream_pds[:,:] = 0.
    mask_stream_pds[:,:] = True

    if(data_trig_pds.shape != (cf.n_pds_trig_channels, cf.n_pds_trig_sample)):

        new_shape = (cf.n_pds_trig_channels, cf.n_pds_trig_sample)
        data_trig_pds = np.resize(data_trig_pds, new_shape)
        mask_trig_pds = np.resize(mask_trig_pds, new_shape)


    data_trig_pds[:,:] = 0.
    mask_trig_pds[:,:] = True


    
def set_waveforms():
    wvf_pos = [[] for x in range(cf.n_tot_channels)]
    wvf_neg = [[] for x in range(cf.n_tot_channels)]

    
class channel:
    def __init__(self, daqch, globch, module, view, vchan, length, capa, gain, pos, card=-1):
        self.daqch = daqch
        self.globch = globch
        self.module = module
        self.view = view
        self.vchan = vchan
        self.length = length
        self.capa   = capa
        self.gain = gain
        self.pos  = pos
        self.prev_daqch = -1
        self.next_daqch = -1
        self.card = card


    def __str__(self):
        return "DAQ "+str(self.daqch)+"-> M"+str(self.module)+" V"+str(self.view)+" ch "+str(self.vchan)+ " - prev : "+str(self.prev_daqch)+" next : "+str(self.next_daqch)

    def set_prev_next(self, prev, nxt):
        self.prev_daqch = prev
        self.next_daqch = nxt

    def get_ana_chan(self):
        return self.module, self.view, self.vchan

    def get_daqch(self):
        return self.daqch

    def get_globch(self):
        return self.globch

    def get_card_nb(self):
        return self.card

class channel_pds:
    def __init__(self, daqch, globch, det, chan, mod, data_type):
        self.daqch = daqch
        self.globch = globch
        self.det = det
        self.chan = chan
        self.module = mod
        self.data_type = data_type

    def __str__(self):
        return "DAQ "+str(self.daqch)+"-> "+self.det+" global "+str(self.globch)+ ", "+str(self.chan)+" data type "+self.data_type

    
class fit_pulse:
    def __init__(self, idaq, v, chan, np_pos, np_neg, fit_pos, fit_neg):
        self.daq_channel = idaq
        self.view = v
        self.channel = chan
        self.n_pulse_pos = np_pos
        self.n_pulse_neg = np_neg
        
        self.fit_pos = fit_pos
        self.fit_neg = fit_neg

        
class noise:
    def __init__(self, ped, rms):
        #keep noise-related stuff arange in global channel numbering

        self.ped_mean = ped
        self.ped_rms  = rms

    def add(self, ped, rms):
        self.ped_mean = np.append(self.ped_mean, ped)
        self.ped_rms  = np.append(self.ped_rms, rms)

class event:
    def __init__(self, det, elec, run, sub, evt, trigger, timestamp, t_s, t_ns, trig_type):
        self.det = det
        self.elec = elec
        self.run_nb  = run
        self.sub  = sub
        self.evt_nb  = evt
        self.trigger_nb = trigger
        self.trig_type = trig_type #as number, see utils/enum_type for words
        """ time of the event, as written in the TriggerRecordHeader"""
        self.event_time = timestamp    #in unix timestamp (s)
        self.time_s = t_s
        self.time_ns = t_ns
        self.charge_time = [0 for x in range(cf.n_module)]#earliest wib unix timestamp (s)
        self.pds_stream_time = 0
        self.pds_trig_time = 0

        self.n_hits = np.zeros((cf.n_view, cf.n_module), dtype=int)
        self.n_tracks2D = np.zeros((cf.n_view), dtype=int)
        self.n_tracks3D = 0
        self.n_single_hits = 0
        self.n_ghosts = 0
        self.n_hits_clusters = 0
        self.noise_raw = [[] for x in range(cf.n_module_used)]
        self.noise_filt = [[] for x in range(cf.n_module_used)]
        self.noise_study = None 
        self.noise_pds_raw = None
        self.noise_pds_filt = None
        self.n_pds_peaks = np.zeros((cf.n_pds_tot_channels), dtype=int)
        self.n_pds_clusters = 0


    def set_file_infos(self, dataflow, datawriter, daqserver):
        self.dataflow = dataflow
        self.datawriter = datawriter
        self.daqserver = daqserver
                
    def set_noise_raw(self, noise):
        self.noise_raw[cf.imod] = noise

    def set_noise_filt(self, noise):
        self.noise_filt[cf.imod] = noise

    def set_noise_study(self, noise):
        self.noise_study[cf.imod] = noise

    def set_noise_pds_raw(self, noise):
        self.noise_pds_raw = noise
        """ in case no PDS noise filtering is done, still fill with something"""
        self.noise_pds_filt = noise

    def set_noise_pds_filt(self, noise):
        self.noise_pds_filt = noise

    def set_pds_stream_timestamp(self, time):
        self.pds_stream_time = time

    def set_pds_trig_timestamp(self, time):
        self.pds_trig_time = time
                
    def set_charge_timestamp(self, module, time):
        self.charge_time[module] = time

        
    def dump(self):
        print("RUN ",self.run_nb, " of ", self.elec, " EVENT ", self.evt_nb, " TRIGGER ", self.trigger_nb, ":", et.TRIGGER_TYPES[self.trig_type], '(',self.trig_type,')')
        print("Taken at ", time.ctime(self.time_s), " + ", self.time_ns, " ns ")




class hits:
    def __init__(self, module, view, daq_channel, start, stop, max_t, max_adc, min_t, min_adc, zero_t, signal):
        self.view    = view
        self.module  = module

        """Each hit should have a unique ID per event"""
        self.ID = -1
        self.daq_channel = daq_channel #+ cf.module_daqch_start[self.module]
        self.channel = chmap[daq_channel].vchan
        self.glob_channel = chmap[daq_channel].globch
        self.start   = start
        self.stop    = stop
        self.pad_start = start
        self.pad_stop  = stop        
        self.Z_start = -1
        self.Z_stop  = -1
        self.signal = signal


        """ time is in time bin number """
        self.max_t   = max_t
        self.min_t   = min_t
        self.zero_t  = zero_t
        self.t = 0.

        self.charge_pos  = 0.
        self.charge_neg  = 0.        
        self.charge = 0.

        
        self.max_adc = max_adc
        self.min_adc = min_adc

        self.max_fC = self.max_adc*chmap[self.daq_channel].gain
        self.min_fC = self.min_adc*chmap[self.daq_channel].gain


        self.X       = chmap[self.daq_channel].pos
        self.Z       = -1


        self.match_2D = -9999
        self.match_3D = -9999
        self.match_dray = -9999
        self.match_ghost = -9999
        self.match_sh = -9999
        self.is_free = True

        self.has_3D  = False
        
        self.x_3D = -9999
        self.y_3D = -9999
        self.cluster = -1
        self.cluster_3D = -1
        self.ID_match_3D = []
        self.cluster_SH = -1

        
    def __lt__(self,other):
        """ sort hits by decreasing Z and increasing channel """
        #return (self.view,  self.X, self.Z) < (other.view, other.X, other.Z)
        return (self.view < other.view) or (self.view==other.view and self.Z > other.Z) or (self.view==other.view and self.Z == other.Z and self.X < other.X)

    def set_index(self, idx):
        self.ID = idx + n_tot_hits

    def hit_positions(self, v):
        if(self.signal == "Induction"):
            self.t = self.zero_t
        else:
            self.t = self.max_t if self.max_t >= 0 else self.min_t

        """ transforms time bins into distance from the anode """
        """ for CB it does not mean someting concrete """
        """ correct Z with PCB positions """

        self.Z = cf.anode_z[self.module] - cf.drift_direction[self.module]*(v * self.t / cf.sampling[self.module] - cf.view_z_offset[self.module][self.view])
        self.Z_start = cf.anode_z[self.module] - cf.drift_direction[self.module]*(v * self.start /cf.sampling[self.module] -  cf.view_z_offset[self.module][self.view])
        self.Z_stop  = cf.anode_z[self.module] - cf.drift_direction[self.module]*(v * self.stop / cf.sampling[self.module] - cf.view_z_offset[self.module][self.view])


    def set_X(self, x):
        self.X = x

    def shift_X(self, u):
        self.X += u

        
    def hit_charge(self):        
        self.charge_pos *= chmap[self.daq_channel].gain
        self.charge_neg *= chmap[self.daq_channel].gain


        """ I'm not sure this is correct for the induction hits """
        self.charge = self.charge_pos if self.signal == "Collection" else self.charge_pos + np.fabs(self.charge_neg)


    def reset_match(self):
        self.match_2D = -9999
        self.match_3D = -9999
        self.match_dray = -9999
        self.match_ghost = -9999
        self.match_sh = -9999
        self.is_free = True

        
    def set_match_2D(self, ID):
        self.match_dray = -9999
        self.match_2D = ID
        self.is_free = False

    def set_match_dray(self, ID):
        self.match_2D = -9999
        self.match_dray = ID
        self.is_free = False


    def set_match_3D(self, ID):
        self.match_3D = ID
        self.is_free = False


    def set_match_ghost(self, ID):    
        self.match_3D = -9999
        self.match_dray = -9999
        self.match_ghost = ID
        self.is_free = False

    def set_match_sh(self, ID):
        self.match_sh = ID
        self.is_free = False

    def set_3D(self, x, y, matchID):#, n=0):
        self.has_3D  = True
        self.x_3D = x
        self.y_3D = y
        self.ID_match_3D = matchID
        #self.ncoll_3D = n
        
    def set_cluster(self, clus):
        self.cluster = clus
        
    def set_cluster_3D(self, clus):
        self.cluster_3D = clus

    def set_cluster_SH(self, clus):
        self.cluster_SH = clus

        
    def get_charges(self):
        return (self.charge, self.charge_pos, self.charge_neg)

    
    def mini_dump(self):
        print(f"Hit {self.ID} (free:{self.is_free}, cluster {self.cluster}, trk {self.match_2D}/{self.match_dray}) v{self.view} ch{self.channel} :: {self.glob_channel} ({self.daq_channel}/{self.module}) t: {self.start} to {self.stop} max {self.max_t} min {self.min_t} maxADC: {self.max_adc:.2f} minADC: {self.min_adc:.2f}, at ({self.X:.2f}, {self.Z:.2f}) [{self.Z_start:.2f},{self.Z_stop:.2f}] Q- {self.charge_neg:.2f} Q+ {self.charge_pos:.2f}")   
        
    def dump(self):

        print("\n**View ", self.view, " Channel ", self.channel, " ID: ", self.ID)
        print("Type ", self.signal)
        print("channel gain ", chmap[self.daq_channel].gain)

        print(" from t ", self.start, " to ", self.stop, " dt = ", self.stop-self.start)
        print(" padded t ", self.pad_start, " to ", self.pad_stop, " dt = ", self.pad_stop-self.pad_start)
        print(" tmax ", self.max_t, " tmin ", self.min_t, ' dt = ', self.min_t-self.max_t)
        print("zero time ", self.zero_t)
        print(" ref time ", self.t)

        print(" positions : ", self.X, ", ", self.Z)
        print(" adc max ", self.max_adc, " adc min ", self.min_adc)
        print(" fC max ", self.max_fC, " fC min ", self.min_fC)
        print(" charges pos : ", self.charge_pos, " neg : ", self.charge_neg)
        print(" Is free ?", self.is_free)
        print(" Match in 2D with trk ", self.match_2D)
        print(" Match in 3D with trk ", self.match_3D)
        print(" Match as dray with trk ", self.match_dray)
        print(" Match as ghost with trk ", self.match_ghost)
        print(" Match as SH with hits ", self.match_sh)


class hits_clusters:
    def __init__(self, ID, view):
        self.ID      = ID
        self.view    = view
        self.n_hits  = 0
        self.hits_ID = []

    def add_hit(self, hit_ID):
        self.n_hits  += 1
        self.hits_ID.append(hit_ID)
    
        
class singleHits:
    def __init__(self, ID_SH, module, n_hits, IDs, x, y, z, d_max_bary, d_min_3D, d_min_2D):
        self.ID_SH = ID_SH
        self.n_hits = n_hits
        self.IDs = IDs
        self.charge_pos = [0.,0.,0.]
        self.charge_neg = [0.,0.,0.]

        self.start   = [-1, -1, -1]
        self.stop   = [-1, -1, -1]

        self.max_t   = [-1, -1, -1]
        self.zero_t   = [-1, -1, -1]
        self.min_t   = [-1, -1, -1]
        self.X = x
        self.Y = y
        self.Z = z
        self.module = module
        
        self.d_bary_max = d_max_bary
        self.d_track_3D = d_min_3D
        self.d_track_2D = d_min_2D

        self.n_veto = [0,0,0]#False, False, False]
        self.timestamp = 0.
        self.match_pds_cluster = -1
        self.Z_from_light = -9999
        

    def set_veto(self, view, nb):
        self.n_veto[view] = nb

    def set_view(self, view, charge_pos, charge_neg, start, stop, max_t, zero_t, min_t):
        self.charge_pos[view] = charge_pos
        self.charge_neg[view] = charge_neg

        self.start[view] = start
        self.stop[view] = stop

        self.max_t[view] = max_t
        self.zero_t[view] = zero_t
        self.min_t[view] = min_t

    def set_timestamp(self):
        delta_time_ref = evt_list[-1].charge_time[self.module] - evt_list[-1].event_time
        delta_time_ref *= 1e6 #in mus
        self.timestamp =  delta_time_ref+ min(self.start)/cf.sampling[self.module]



    def dump(self):
        print('\n****')        
        print('Single Hit at ', self.X, ', ', self.Y, ', ', self.Z)
        print('Max hit distance to barycenter ', self.d_bary_max)
        print('Nb of hits/view ', self.n_hits)
        print('IDs ', self.IDs)
        print('Charge pos ', self.charge_pos)
        print('Charge neg ', self.charge_neg)
        print('Starts ', self.start)
        print('Stops ', self.stop)
        print('Time max ', self.max_t)
        print('Time zero ', self.zero_t)
        print('Time min ', self.min_t)
        print('Distance to closest track in 2D:', self.d_track_2D, ' in 3D ', self.d_track_3D)
        print('Nb in veto ', self.n_veto)
        print('Timestamp : ', self.timestamp, ' mus')
        print('Matched with light cluster : ', self.match_pds_cluster)
        print('Z estimated from light cluster : ', self.Z_from_light)

        for iv in range(3):
            hits_list[self.IDs[iv][0]-n_tot_hits].mini_dump()
        
class trk2D:
    def __init__(self, ID, view, ini_slope, ini_slope_err, x0, y0, t0, q0, hit_ID, chi2):
        self.trackID = ID
        self.view    = view
        self.module_ini = -1
        self.module_end  = -1

        self.ini_slope       = ini_slope
        self.ini_slope_err   = ini_slope_err
        self.end_slope       = ini_slope
        self.end_slope_err   = ini_slope_err

        self.n_hits      = 1
        self.n_hits_dray = 0
        self.hits_ID = [hit_ID]
 
        self.path    = [(x0,y0)]
        self.dQ      = [q0]
        self.dz      = -1
        self.dx      = -1
        
        self.chi2_fwd    = chi2
        self.chi2_bkwd   = chi2

        self.drays   = []
        self.drays_ID   = []

        self.tot_charge = q0
        self.dray_charge = 0.

        self.len_straight = 0.
        self.len_path = 0.

        self.matched = [-1 for x in range(cf.n_view)]
        self.match_3D = -1
        
        self.ini_time = t0
        self.end_time = t0

        self.ghost = False

        self.matched_tracks = [[] for x in range(cf.n_view)]
        self.label3D = -1
        
    def __lt__(self,other):
        print('sorting 2D tracks!!!')
        """ sort tracks by decreasing Z and increasing channel """
        return (self.path[0][1] > other.path[0][1]) or (self.path[0][1] == other.path[0][1] and self.path[0][0] < other.path[0][0])

    def set_ID(self, ID):
        self.trackID = ID

    def add_drays(self, x, y, q, ID):
        self.drays.append((x,y,q))
        self.drays_ID.append(ID)
        self.dray_charge += q
        self.n_hits_dray += 1
        try:
            self.hits_ID.remove(ID)
        except ValueError:
            #print('weird ... hit', ID, 'is not in the list?')
            a = 0

            
    def add_hit(self, x, y, q, t, hID):
        self.n_hits += 1

        self.len_path += math.sqrt( pow(self.path[-1][0]-x, 2) + pow(self.path[-1][1]-y,2) )
        #beware to append (x,y) after !
        self.path.append((x,y))
        self.dQ.append(q)
        self.hits_ID.append(hID)
        self.tot_charge += q
        self.len_straight = math.sqrt( pow(self.path[0][0]-self.path[-1][0], 2) + pow(self.path[0][1]-self.path[-1][1], 2) )
        self.end_time = t

    def add_hit_update(self, slope, slope_err, x, y, t, q, hID, chi2):
        self.end_slope = slope
        self.end_slope_err = slope_err
        self.n_hits += 1
        self.len_path += math.sqrt( pow(self.path[-1][0]-x, 2) + pow(self.path[-1][1]-y,2) )

        #beware to append (x,y) after !
        self.path.append((x,y))
        self.dQ.append(q)
        self.hits_ID.append(hID)
        self.chi2 = chi2
        self.tot_charge += q
        self.len_straight = math.sqrt( pow(self.path[0][0]-self.path[-1][0], 2) + pow(self.path[0][1]-self.path[-1][1], 2) )
        self.end_time = t

    def update_forward(self, chi2, slope, slope_err):
        self.chi2_fwd = chi2
        self.end_slope = slope
        self.end_slope_err = slope_err

    def update_backward(self, chi2, slope, slope_err):
        self.chi2_bkwd = chi2
        self.ini_slope = slope
        self.ini_slope_err = slope_err


    def remove_all_drays(self):
        [self.path.append((i,j)) for i,j,k in self.drays]

        self.hits_ID += self.drays_ID
        [self.dQ.append(k) for i,j,k in self.drays]
        
        for x in self.drays_ID:
            hits_list[x-n_tot_hits].set_match_2D(self.trackID)

        self.drays   = []
        self.drays_ID   = []
        
        self.finalize_track()
        
    def reset_path(self, path, dQ, ID):
        self.path = path
        self.dQ = dQ
        self.hits_ID = ID
        self.finalize_track()
        

    def finalize_track(self):
        if(self.path[-1][1] > self.path[0][1]):

            self.path.reverse()
            self.dQ.reverse()
            self.hits_ID.reverse()
            self.ini_slope, self.end_slope = self.end_slope, self.ini_slope
            self.ini_slope_err, self.end_slope_err = self.end_slope_err, self.ini_slope_err

            self.chi2_fwd, self.chi2_bkwd = self.chi2_bkwd, self.chi2_fwd
            #print(self.trackID, " : wrong order check :", self.path[0][1], " to ", self.path[-1][1])
            self.ini_time, self.end_time = self.end_time, self.ini_time

        self.n_hits = len(self.path)
        self.tot_charge = sum(self.dQ)

        self.n_hits_dray = len(self.drays)
        self.dray_charge = sum(k for i,j,k in self.drays)
        
        self.len_straight = math.sqrt( pow(self.path[0][0]-self.path[-1][0], 2) + pow(self.path[0][1]-self.path[-1][1], 2) )
        self.len_path = 0.
        for i in range(self.n_hits-1):
            self.len_path +=  math.sqrt( pow(self.path[i][0]-self.path[i+1][0], 2) + pow(self.path[i][1]-self.path[i+1][1],2) )
        
        self.dz = np.fabs(self.path[-1][1]-self.path[0][1])
        self.dx = np.fabs(self.path[-1][0]-self.path[0][0])
        
        self.module_ini = hits_list[self.hits_ID[0]-n_tot_hits].module
        self.module_end = hits_list[self.hits_ID[-1]-n_tot_hits].module


    def dist(self, other, i=-1, j=0):
        return math.sqrt(pow( self.path[i][0] - other.path[j][0], 2) + pow(self.path[i][1] - other.path[j][1], 2))



    def slope_comp(self, other):#, sigcut):
        """ check if both tracks have the same slope direction """
        if(self.end_slope * other.ini_slope < 0.):
            return 9999. #False

        """ if slope error is too low, re-assign it to 5 percent """
        if(self.end_slope_err == 0 or math.fabs(self.end_slope_err/self.end_slope) < 0.05):
            end_err = math.fabs(self.end_slope*0.05)
        else:
            end_err = self.end_slope_err

        if(other.ini_slope_err == 0 or math.fabs(other.ini_slope_err/other.ini_slope) < 0.05):
            ini_err = math.fabs(other.ini_slope*0.05)
        else:
            ini_err = other.ini_slope_err

        #return (math.fabs( self.end_slope - other.ini_slope) < (sigcut*end_err + sigcut*ini_err))

        return math.fabs( self.end_slope - other.ini_slope) / (end_err + ini_err)


    def x_extrapolate(self, other, rcut):
        view = self.view

        xa = self.path[-1][0]
        za = self.path[-1][1]
        xb = other.path[0][0]
        zb = other.path[0][1]
        return ( math.fabs( xb - (xa+(zb-za)*self.end_slope)) < rcut) and (math.fabs( xa-(xb+(za-zb)*other.ini_slope)) < rcut)

    def z_extrapolate(self, other, rcut):
        view = self.view

        xa = self.path[-1][0]
        za = self.path[-1][1]
        xb = other.path[0][0]
        zb = other.path[0][1]

        if(self.end_slope == 0 and other.ini_slope == 0) :
            return True

        if(self.end_slope == 0):
            return (math.fabs(za - zb - (xa-xb)/other.ini_slope) < rcut)
        elif( other.ini_slope == 0):
            return ( math.fabs(zb - za - (xb-xa)/self.end_slope) < rcut)
        else:
            return ( math.fabs(zb - za - (xb-xa)/self.end_slope) < rcut) and (math.fabs(za - zb - (xa-xb)/other.ini_slope) < rcut)


    def joinable(self, other, dcut, sigcut, rcut):
        if(self.view != other.view):
            return False
        if( self.dist(other) < dcut and self.slope_comp(other) <  sigcut and self.x_extrapolate(other, rcut) and self.z_extrapolate(other, rcut)):
            return True


    def merge(self, other):
        self.n_hits += other.n_hits
        self.n_hits_dray += other.n_hits_dray
        self.chi2_fwd += other.chi2_fwd #should be refiltered though
        self.chi2_bkwd += other.chi2_bkwd #should be refiltered though
        self.tot_charge += other.tot_charge
        self.dray_charge += other.dray_charge
        self.len_path += other.len_path
        self.len_path += self.dist(other)
        self.matched = [-1 for x in range(cf.n_view)]
        self.drays.extend(other.drays)
        self.drays_ID.extend(other.drays_ID)
        self.match_3D = -1
        self.ghost = False
        assert self.label3D == other.label3D
        other.label3D = -1

        
        
        if(self.path[0][1] > other.path[0][1]):
            self.ini_slope = self.ini_slope
            self.ini_slope_err = self.ini_slope_err
            self.ini_time = self.ini_time
        else:
            self.ini_slope = self.end_slope
            self.ini_slope_err = self.end_slope_err
            self.ini_time = other.ini_time

        if(self.path[-1][1] > other.path[-1][1]):
            self.end_slope = other.end_slope
            self.end_slope_err = other.end_slope_err
            self.end_time = other.end_time
        else:
            self.end_slope = self.end_slope
            self.end_slope_err = self.end_slope_err
            self.end_time = self.end_time
                         

        self.path.extend(other.path)
        self.dQ.extend(other.dQ)
        self.hits_ID.extend(other.hits_ID)


        self.set_match_hits_2D(self.trackID)
        self.finalize_track()
        
    def shift_x_coordinates(self, u):
        self.path = [ (p[0]+u, p[1]) for p in self.path]
        self.drays = [(p[0]+u, p[1], p[2]) for p in self.drays]

        [hits_list[i-n_tot_hits].shift_X(u) for i in self.hits_ID]
        [hits_list[i-n_tot_hits].shift_X(u) for i in self.drays_ID]
        
    def charge_in_z_interval(self, start, stop):
        return sum([q for q, (x, z) in zip(self.dQ, self.path) if z >= start and z <= stop])


    def set_match_hits_2D(self, ID):
        self.trackID = ID
        
        for x in self.hits_ID:
            hits_list[x-n_tot_hits].set_match_2D(ID)
            
        for x in self.drays_ID:
            hits_list[x-n_tot_hits].set_match_dray(ID)

    
    def set_match_hits_3D(self, ID):
        self.match_3D = ID
        
        for x in self.hits_ID:
            hits_list[x-n_tot_hits].set_match_3D(ID)
        for x in self.drays_ID:
            hits_list[x-n_tot_hits].set_match_3D(ID)

    def set_match_hits_ghost(self, ID):
        for x in self.hits_ID:
            hits_list[x-n_tot_hits].set_match_ghost(ID)
        for x in self.drays_ID:
            hits_list[x-n_tot_hits].set_match_ghost(ID)

    def set_label(self,ID):
        self.label3D = ID

    def mini_dump(self):
        print("2D track", self.trackID,"mod", self.module_ini, " to ", self.module_end, "view:", self.view, "from (%.1f,%.1f)"%(self.path[0][0], self.path[0][1]), "to (%.1f, %.1f)"%(self.path[-1][0], self.path[-1][1]), "N =", self.n_hits, "L= %.1f/%.1f"%(self.len_straight, self.len_path), "Q = %.1f"%(self.tot_charge), "Dray N=", self.n_hits_dray, "Qdray = %.1f"%(self.dray_charge), "3D MATCH :", self.match_3D, '/', self.label3D,'slopes: (%.2f, %.2f)'%(self.ini_slope, self.end_slope), 'err: (%.2f, %.2f)'%(self.ini_slope_err, self.end_slope_err), 'times ', self.ini_time, ' to ', self.end_time)#, 'NIDs', len(self.hits_ID), 'NdrIDs', len(self.drays_ID), 'p', len(self.path))
        #print('track hits ID ', self.hits_ID)
        #print('test', [hits_list[x-n_tot_hits].match_2D for x in self.hits_ID])
        #print('matches : ', self.matched)
        #print('3D match : ', self.match_3D)
        

class trk3D:
    def __init__(self):

        self.match_ID  =  [[-1]*cf.n_view for i in range(cf.n_module)]
        self.ID_3D      = -1

        
        self.n_matched = 0
        
        self.chi2    = [-1]*cf.n_view
        self.momentum = -1

        self.d_match = -1

        self.n_hits   = [-1]*cf.n_view #t.n_hits for t in trks]

        self.len_straight = [-1]*cf.n_view
        self.len_path = [-1]*cf.n_view
        self.dz = -1
        
        self.tot_charge = [-1]*cf.n_view
        self.dray_charge = [-1]*cf.n_view


        self.ini_theta = -1
        self.end_theta = -1
        self.ini_phi = -1
        self.end_phi = -1

        self.t0_corr = 9999.
        self.z0_corr = 9999.
        

        self.ini_time = -1#cf.n_sample[cf.imod]+1
        self.end_time = -1


        ''' track boundaries '''
        self.path = [[] for x in range(cf.n_view)]
        self.dQ = [[] for x in range(cf.n_view)]
        self.hits_ID = [[] for x in range(cf.n_view)]

        self.ds = [[] for x in range(cf.n_view)]

        self.module_ini = -1
        self.module_end = -1

        self.n_module_crossed = 1
        
        self.is_cathode_crosser = False
        self.cathode_crosser_ID = -1
        self.is_module_crosser  = False
        self.is_anode_crosser = False
        self.exit_point = [-1, -1, -1]
        self.exit_trk_end = -1
        self.cathode_crossing_trk_end = -1

        
        """ (module, x, y, z, theta, phi) """
        self.cathode_crossing   = [(-1, -1, -1, -1, -1, -1), (-1, -1, -1, -1, -1, -1)]
        self.module_crossing   = [(-1, -1, -1, -1, -1, -1), (-1, -1, -1, -1, -1, -1)]
        
        self.timestamp = 0.
        self.timestamp_r = 0.
        
        self.match_pds_cluster = -1

    def set_ID(self, ID):
        self.ID_3D = ID

    def reset_anode_crosser(self):
        self.is_anode_crosser = False
        self.exit_point = [-1, -1, -1]
        self.exit_trk_end = -1

    def set_anode_crosser(self, exit_point, idx):
        self.is_anode_crosser = True
        self.exit_point = exit_point
        self.exit_trk_end = idx

    def set_module_crosser(self, crossing_points):
        self.n_module_crossed += 1
        self.is_module_crosser = True
        self.module_crossing = crossing_points

        
    def set_cathode_crosser(self, crossing_points, ID, idx):
        self.is_cathode_crosser = True
        self.cathode_crossing = crossing_points
        self.cathode_crosser_ID = ID
        self.cathode_crossing_trk_end = idx
        
    def set_view(self, trk, path, dq, ds, hits_id, isFake=False):
        view = trk.view
        trk.match_3D = self.ID_3D
        if(isFake == False):
            path, dq, ds, hits_id = self.check_descending(path, dq, ds, hits_id)

        
        self.path[view]  = path
        self.dQ[view]   = dq
        self.ds[view]   = ds
        self.hits_ID[view] = hits_id
        self.tot_charge[view] = sum(q/s for q,s in zip(dq,ds))

        if(isFake == True):
            self.len_straight[view] = 0.
            self.len_path[view] = 0.
            self.n_hits[view] = 0
            self.chi2[view] = 0.

            
        else:
            self.n_hits[view] = len(path)
            self.chi2[view] = trk.chi2_fwd
            self.match_ID[trk.module_ini][view] = trk.trackID#llll
            self.match_ID[trk.module_end][view] = trk.trackID#llll

            self.len_straight[view] = math.sqrt( sum([pow(path[0][i]-path[-1][i], 2) for i in range(3)]))
            self.len_path[view] = 0.

            if(self.ini_time < 0 and self.end_time < 0):
                self.ini_time = trk.ini_time
                self.end_time = trk.end_time
            
            if(cf.drift_direction[trk.module_ini]>0):
                self.ini_time = trk.ini_time if trk.ini_time < self.ini_time else self.ini_time
                self.end_time = trk.end_time if trk.end_time > self.end_time else self.end_time
            else:
                self.ini_time = trk.ini_time if trk.ini_time > self.ini_time else self.ini_time
                self.end_time = trk.end_time if trk.end_time < self.end_time else self.end_time
                
            self.n_matched += 1
            
            for i in range(len(path)-1):
                self.len_path[view] +=  math.sqrt( pow(path[i][0]-path[i+1][0], 2) + pow(path[i][1]-path[i+1][1],2)+ pow(path[i][2]-path[i+1][2],2) )


            
    def remove_hit(self, hit_ID, view, module):
        
        idx = self.hits_ID[view].index(hit_ID)
        
        self.hits_ID[view].pop(idx)
        self.path[view].pop(idx)
        self.dQ[view].pop(idx)
        self.ds[view].pop(idx)

        self.n_hits[view] -= 1

        if(self.n_hits[view] > 1):
            self.len_straight[view] = math.sqrt( sum([pow(self.path[view][0][i]-self.path[view][-1][i], 2) for i in range(3)]))

            for i in range(len(self.path[view])-1):
                self.len_path[view] +=  math.sqrt( pow(self.path[view][i][0]-self.path[view][i+1][0], 2) + pow(self.path[view][i][1]-self.path[view][i+1][1],2)+ pow(self.path[view][i][2]-self.path[view][i+1][2],2) )


        else:
            self.n_matched -= 1            
            self.match_ID[module][view] = -1
            self.len_straight[view] = 0.
            self.len_path[view] = 0.

        
        
    def check_descending(self,path, dq, ds, hits_id):
        if(cf.tpc_orientation == 'Horizontal'):
            if(path[0][0] < path[-1][0]):
                return path[::-1], dq[::-1], ds[::-1], hits_id[::-1]

        return path, dq, ds, hits_id


    def check_descending_internal(self):
        if(cf.tpc_orientation == 'Horizontal'):
            
            if(np.any([self.path[iv][0][0] < self.path[iv][-1][0] for iv in range(cf.n_view)])):
                print('upside down')
                for iv in range(cf.n_view):
                    self.path[iv] = self.path[iv][::-1]
                    self.dQ[iv] = self.dQ[iv][::-1]
                    self.ds[iv] = self.ds[iv][::-1]
                self.hits_ID = self.hits_ID[::-1]
                self.ini_theta, self.end_theta = self.end_theta, self.ini_theta
                self.ini_phi, self.end_phi = self.end_phi, self.ini_phi
            
    def check_views(self):
        n_fake = 0
        for i in range(cf.n_view):
            #if(np.all(self.match_ID[:][i] == -1)):
            if(sum([self.match_ID[k][i] for k in range(cf.n_module)]) == -cf.n_module):
                tfake = trk2D(-1, i, -1, -1, -9999., -9999., -9999., 0, -1,0)

                self.set_view(tfake, [(-9999.,-9999.,-9999), (9999., 9999., 9999.)], [0., 0.], [1., 1.],[-1, -1], isFake=True)
                n_fake += 1

        self.n_matched = cf.n_view - n_fake
        return n_fake

    def set_modules(self, ini, end):
        self.module_ini = ini
        self.module_end = end

    def boundaries(self):
        sum_match = [sum([self.match_ID[k][i] for k in range(cf.n_module)]) for i in range(cf.n_view)]

        
        inis = np.asarray([list(self.path[i][0]) if sum_match[i]>-1*cf.n_module else [np.nan, np.nan, np.nan] for i in range(cf.n_view)])
        ends = np.asarray([list(self.path[i][-1])  if sum_match[i]>-1*cf.n_module  else [np.nan, np.nan, np.nan] for i in range(cf.n_view)])

        
        lengths = np.asarray([[np.linalg.norm(e-i) for e in ends] for i in inis])
        imaxs = np.unravel_index(np.nanargmax(lengths, axis=None), lengths.shape)
        imins = np.unravel_index(np.nanargmin(lengths, axis=None), lengths.shape)
        
        ''' begining '''
        #v_higher = np.argmax([self.path[i][0][2] for i in range(cf.n_view)])
        self.ini_x = self.path[imaxs[0]][0][0]
        self.ini_y = self.path[imaxs[0]][0][1]
        self.ini_z = self.path[imaxs[0]][0][2]

        self.ini_z_overlap = self.path[imins[0]][0][2]
        #min([self.path[i][0][2] if k >=0 else 99999. for i,k in zip(range(cf.n_view),self.match_ID)])

        ''' end '''
        #v_lower = np.argmax([self.path[i][-1][2] for i in range(cf.n_view)])
        self.end_x = self.path[imaxs[1]][-1][0]
        self.end_y = self.path[imaxs[1]][-1][1]
        self.end_z = self.path[imaxs[1]][-1][2]

        self.end_z_overlap = self.path[imins[1]][-1][2]
        #max([self.path[i][-1][2] if k >= 0 else -9999. for i,k in zip(range(cf.n_view),self.match_ID)])
        
        self.dz = self.end_z - self.ini_z
        
    def set_t0_z0(self, t0, z0):

        self.t0_corr = t0
        self.z0_corr = z0


    def set_timestamp(self, tini, tend):
        
        delta_time_ref = evt_list[-1].charge_time[self.module_ini] - evt_list[-1].event_time
        delta_time_ref *= 1e6 #in mus

            
        self.timestamp = delta_time_ref + tini #ref_trk_time/cf.sampling[self.module_ini]
        self.timestamp_r = delta_time_ref + tend

        
    def set_angles(self, theta_ini, phi_ini, theta_end, phi_end):
        self.ini_phi = phi_ini
        self.ini_theta = theta_ini
        self.end_phi = phi_end
        self.end_theta = theta_end



    def merge(self, other, idx_merge):        
        other.ID_3D = -1
        
        if(self.module_ini == self.module_end):
            imod = self.module_ini
            for iv in range(cf.n_view):
                if(self.match_ID[imod][iv] < 0):
                    self.path[iv].clear()
                    self.dQ[iv].clear()
                    self.ds[iv].clear()
                    self.hits_ID[iv].clear()

        else:
            print('WHAAAT?')
            

        self.momentum = -1

        """ will be recomputed anyway """
        self.d_match += other.d_match
        self.d_match /= 2

        if(evt_list[-1].det == 'pdhd'):                
            if(self.ini_y > other.ini_y):
                self.module_end = other.module_end
            
            else:
                self.module_ini = other.module_ini


        elif(evt_list[-1].det == 'pdvd'):                
                self.module_end = other.module_end

        print('merge ', self.ini_time, 'to',self.end_time)
        print('with ', other.ini_time, ' with', other.end_time)
        print('idx merge ', idx_merge)
        
        if(idx_merge[0] == 1):
            self.ini_time = self.ini_time
            self.end_time = other.end_time
            self.end_theta = other.end_theta
            self.end_phi = other.end_phi
        else:
            self.ini_time = other.ini_time
            self.end_time = self.end_time
            self.ini_theta = other.ini_theta
            self.ini_phi = other.ini_phi
        print('---->>>> ', self.ini_time, 'to',self.end_time)
        """
        if(evt_list[-1].det == 'pdvd' and cf.drift_direction[self.module_ini] == -1):
                self.ini_time = max(self.ini_time, other.ini_time)
                self.end_time = min(self.end_time, other.end_time)
            else:
                self.ini_time = min(self.ini_time, other.ini_time)
                self.end_time = max(self.end_time, other.end_time)
        """
        #self.ini_time = min(self.ini_time, other.ini_time)
        #self.end_time = max(self.end_time, other.end_time)
        
            
        self.timestamp = min(self.timestamp, other.timestamp)
        self.timestamp_r = max(self.timestamp_r, other.timestamp_r)


        

        for iv in range(cf.n_view):
            for imod in range(cf.n_module):
                #
                if(self.match_ID[imod][iv]>=0 and other.match_ID[imod][iv]>=0):
                    print('wait, what ? that is not possible')
                
                elif(self.match_ID[imod][iv]<0 and other.match_ID[imod][iv]>=0):
                    self.match_ID[imod][iv] = other.match_ID[imod][iv]
                    self.path[iv].extend(other.path[iv])
                    self.dQ[iv].extend(other.dQ[iv])
                    self.ds[iv].extend(other.ds[iv])
                    self.hits_ID[iv].extend(other.hits_ID[iv])
                
                    self.chi2[iv] += other.chi2[iv]
                    self.n_hits[iv] += other.n_hits[iv]


                    self.tot_charge[iv] += other.tot_charge[iv]
                    self.dray_charge[iv] += other.dray_charge[iv]
            
        self.check_descending_internal()
        self.check_views()
        
        for iv in range(cf.n_view):
            self.len_straight[iv] = math.sqrt( sum([pow(self.path[iv][0][i]-self.path[iv][-1][i], 2) for i in range(3)]))
                
            for i in range(len(self.path[iv])-1):
                self.len_path[iv] +=  math.sqrt( pow(self.path[iv][i][0]-self.path[iv][i+1][0], 2) + pow(self.path[iv][i][1]-self.path[iv][i+1][1],2)+ pow(self.path[iv][i][2]-self.path[iv][i+1][2],2) )
                
        

        self.boundaries()
        
        
        
    def dump(self):
        print('\n----')
        print('Track with ID ', self.ID_3D)
        print('match IDs ', self.match_ID, ' n view matched', self.n_matched)
        #print('test ', [len(self.path[iv]) for iv in range(cf.n_view)])
        print(" From (%.2f, %.2f, %.2f) to (%.2f, %.2f, %.2f)"%(self.ini_x, self.ini_y, self.ini_z, self.end_x, self.end_y, self.end_z))
        print(" module ini ", self.module_ini, " module end ", self.module_end)
        print(" Time start ", self.ini_time, ' stop ', self.end_time)
        print(" Timestamp", self.timestamp, "us possibly up to ", self.timestamp_r,"us")
        print(' z-overlap ', self.ini_z_overlap, ' to ', self.end_z_overlap)
        print(" N Hits ", self.n_hits)
        print(" theta, phi: [ini] %.2f ; %.2f"%(self.ini_theta, self.ini_phi), " -> [end] %.2f ; %.2f "%( self.end_theta, self.end_phi))
        print(" Straight Lengths :  ", self.len_straight)
        print(" Path Lengths ", self.len_path)
        print(" Total charges ", self.tot_charge)
        print(" z0 ", self.z0_corr, " t0 ", self.t0_corr)
        print(" MATCHING DISTANCE SCORE : ", self.d_match)
        #print(" timestamp ", self.timestamp, ' mus')
        print(" matched with light cluster ", self.match_pds_cluster)
        print("Cathode?", self.is_cathode_crosser, " with ", self.cathode_crosser_ID)
        print("Anode? ", self.is_anode_crosser, " exit ", self.exit_point)
        print('----\n')

class ghost:
    def __init__(self, ghost_id, t2d_id, min_dist, ghost_charge, trk_charge, nhits, mod):#, t3d_id, min_dist, xstart, ystart, zstart):
        self.ghost_ID = ghost_id
        self.trk2D_ID = t2d_id

        self.module_ini = mod[0]
        self.module_end = mod[1]
        self.trk3D_ID = -1
        self.anode_x = -9999
        self.anode_y = -9999
        self.anode_z = -9999
        self.min_dist = min_dist

        self.n_hits = nhits
        self.path = []
        self.dQ = []
        self.hits_ID = []
        self.ds = []

        self.theta = -9999
        self.phi = -9999
        
        self.ghost_charge = ghost_charge
        self.trk_charge = trk_charge

        self.t0_corr = -9999
        self.z0_corr = -9999

    def set_3D_ghost(self, t3d_id, path, dQ, ds, hits, x, y, z, theta, phi, t0, z0):
        self.trk3D_ID = t3d_id
        self.anode_x = x
        self.anode_y = y
        self.anode_z = z

        self.path = path
        self.dQ = dQ
        self.hits_ID = hits
        self.ds = ds

        self.theta = theta
        self.phi = phi

        self.t0_corr = t0
        self.z0_corr = z0

        
class pds_peak:
    def __init__(self, glob_ch, chan, module, start, stop, max_t, max_adc, timestamp):

        """Each hit should have a unique ID per event"""
        self.ID = -1
        self.glob_ch = glob_ch
        self.daq_ch  = chmap_pds[self.glob_ch].daqch
        self.channel = chan
        self.module  = module
        self.start   = start #time in tick nb
        self.stop    = stop
        self.pad_start   = start
        self.pad_stop    = stop
        self.timestamp = timestamp
        
        """ time is in time bin number """
        self.max_t   = max_t
        self.charge  = 0.        
        self.max_adc = max_adc

        self.cluster_ID = -1
        self.match_3D   = -9999
        self.match_sh   = -9999
        
    def set_charge(self, q):
        self.charge = q
            
    def set_index(self, idx):
        self.ID = idx + n_tot_pds_peaks

    def set_cluster_ID(self, idx):
        self.cluster_ID = idx
        
    def dump(self):
        print(f"{self.ID} Glob Channel {self.glob_ch}/DAPHNE chan {self.channel}: peak from {self.start} to {self.stop} max at {self.max_t}; ADC {self.max_adc:.1f} integral {self.charge:.1f} Cluster {self.cluster_ID} Timestamp {self.timestamp}")

class pds_cluster:
    def __init__(self, ID, peak_IDs, glob_chans, channels, t_starts, t_maxs, t_stops, max_adcs, charges, timestamp):

        self.ID   = ID #the ID of the cluster
        self.size = len(peak_IDs)
        self.peak_IDs  = peak_IDs
        self.glob_chans = glob_chans
        self.channels = channels
        self.t_starts = t_starts
        self.t_maxs = t_maxs
        self.t_stops = t_stops
        self.max_adcs = max_adcs
        self.charges = charges

        self.t_start  = min(self.t_starts)
        self.t_stop  = max(self.t_stops)
        self.timestamp = timestamp

        self.match_trk3D  = [-1 for x in range(cf.n_drift_volumes)] #for double-sided PDS
        self.match_single = -1

        self.dist_closest_strip  = []
        self.id_closest_strip    = []
        self.point_closest = []
        self.point_impact = []
        self.point_closest_above = []

    def set_ID(self, idx):
        self.ID = idx

    def set_timestamp(self, t):
        self.timestamp = t
        
    def merge(self, other):
        self.ID = min(self.ID, other.ID)
        self.size += other.size

        self.peak_IDs.extend(other.peak_IDs)
        self.glob_chans.extend(other.glob_chans)
        self.channels.extend(other.channels)
        self.t_starts.extend(other.t_starts)
        self.t_maxs.extend(other.t_maxs)
        self.t_stops.extend(other.t_stops)
        self.max_adcs.extend(other.max_adcs)
        self.charges.extend(other.charges)

        self.t_start = min(self.t_starts)
        self.t_stop  = max(self.t_stops)
        self.timestamp = min([self.timestamp, other.timestamp])
        
    def dump(self):
        print('\n')
        print('Cluster ', self.ID, ' has ', self.size, ' peaks')
        print('start at ', self.t_start, ' stop ', self.t_stop)
        print('Peak IDS ', self.peak_IDs)
        print('Glob Channels ', self.glob_chans)
        print('Max ADC ', self.max_adcs)
        print('Timestamp = ', self.timestamp, 'mus')
        print('matched with trk ',self.match_trk3D, ' or single hit ', self.match_single)
        print('closest dist. : ', self.dist_closest_strip)
        print('closest SiPM strip. : ', self.id_closest_strip)
        print('closest point : ', self.point_closest)
        print('closest point inside ?', self.point_closest_above)


class debug:
    def __init__(self):
        
        self.read_data = [-1 for x in range(cf.n_module)]
        self.ped_1 = [-1 for x in range(cf.n_module)]
        self.fft = [-1 for x in range(cf.n_module)]
        self.ped_2 = [-1 for x in range(cf.n_module)]
        self.cnr = [-1 for x in range(cf.n_module)]
        self.ped_3 = [-1 for x in range(cf.n_module)]
        self.hit_f = [-1 for x in range(cf.n_module)]
        self.trk2D_1 = [-1 for x in range(cf.n_module)]
        self.trk2D_2 = [-1 for x in range(cf.n_module)]
        self.stitch2D = [-1 for x in range(cf.n_module)]
        #self.stitch2D_gap_1 = -1
        #self.stitch2D_gap_2 = -1
        self.trk3D    = [-1 for x in range(cf.n_module)]
        self.stitch3D  = -1
        self.single    = [-1 for x in range(cf.n_module)]
        self.output    = -1
        self.time_mod  = [-1 for x in range(cf.n_module)]
        self.time_tot  = -1
        self.memory_mod = [-1 for x in range(cf.n_module)]
        self.memory_tot = -1
        
        
    def dump(self):
        print(f"read data : "+', '.join('{:.2f}'.format(f) for f in self.read_data))
        print(f"1st pass ped : "+', '.join('{:.2f}'.format(f) for f in self.ped_1))
        print(f"FFT : "+', '.join('{:.2f}'.format(f) for f in self.fft))
        print(f"2nd pass ped : "+', '.join('{:.2f}'.format(f) for f in self.ped_2))
        print(f"CNR : "+', '.join('{:.2f}'.format(f) for f in self.cnr))
        print(f"3rd pass ped : "+', '.join('{:.2f}'.format(f) for f in self.ped_3))
        print(f"Hit finder : "+', '.join('{:.2f}'.format(f) for f in self.hit_f))
        print(f"Track2D 1 : "+', '.join('{:.2f}'.format(f) for f in self.trk2D_1))
        print(f"Track2D_2 : "+', '.join('{:.2f}'.format(f) for f in self.trk2D_2))
        print(f"Stitch 2D : "+', '.join('{:.2f}'.format(f) for f in self.stitch2D))
        #print(f"Stitch 2D gap 1: {self.stitch2D_gap_1:.2f}")
        #print(f"Stitch 2D gap 2: {self.stitch2D_gap_2:.2f}")
        print(f"Track 3D :  "+', '.join('{:.2f}'.format(f) for f in self.trk3D))
        print(f"Stitch 3D  : {self.stitch3D:.2f}")
        print(f"Single  :  "+', '.join('{:.2f}'.format(f) for f in self.single))
        print(f"Output  : {self.output:.2f}")
        print('-*-*-*-*-*-*-*-*-*-*')
        print(f"Time per module "+', '.join('{:.2f}'.format(f) for f in self.time_mod))
        print(f"Memory per module  :  "+', '.join('{:.2f}'.format(f*1e-6) for f in self.memory_mod))
        print(f"Total time   : {self.time_tot:.2f}")
        print(f"Total Memory : {self.memory_tot*1e-6:.2f} MB")
