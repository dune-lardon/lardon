import numpy as np
import config as cf
import time
import math

chmap = []
evt_list = []
hits_list = []
tracks2D_list = []
tracks3D_list = []
single_hits_list = []
ghost_list = []
pulse_fit_res = []
pds_peak_list = []
wvf_pos = [[] for x in range(cf.n_tot_channels)]
wvf_neg = [[] for x in range(cf.n_tot_channels)]

data_daq = np.zeros((cf.n_tot_channels, cf.n_sample), dtype=np.float32) #view, vchan

"""
the mask will be used to differentiate background (True for noise processing) from signal (False for noise processing)
at first everything is considered background (all at True)
"""
mask_daq  = np.ones((cf.n_tot_channels, cf.n_sample), dtype=bool)
"""
alive_chan mask intends to not take into account broken channels
True : not broken
False : broken
"""
alive_chan = np.ones((cf.n_tot_channels, cf.n_sample), dtype=bool)

data_pds = np.zeros((cf.n_pds_channels, cf.n_pds_sample), dtype=np.float32)
mask_pds = np.ones((cf.n_pds_channels, cf.n_pds_sample), dtype=bool)
chmap_daq_pds = []
chmap_pds = []


data = np.zeros((cf.n_module, cf.n_view, max(cf.view_nchan), cf.n_sample), dtype=np.float32)


n_tot_hits = 0
def reset_event():
    mask_daq[:,:] = True
    data_daq[:,:] = 0.
    data_pds[:,:] = 0.
    mask_pds[:,:] = True
    
    data[:,:,:,:] = 0.

    hits_list.clear()
    tracks2D_list.clear()
    tracks3D_list.clear()
    evt_list.clear()
    single_hits_list.clear()
    ghost_list.clear()
    pds_peak_list.clear()
    
    pulse_fit_res.clear()

class channel:
    def __init__(self, daqch, globch, module, view, vchan, length, capa, gain, pos):
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


class channel_pds:
    def __init__(self, daqch, globch, det, chan):
        self.daqch = daqch
        self.globch = globch
        self.det = det
        self.chan = chan


    def __str__(self):
        return "DAQ "+str(self.daqch)+"-> "+self.det+" global "+str(self.globch)+ ", "+str(self.chan)

    
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
        self.ped_mean = ped
        self.ped_rms  = rms

class event:
    def __init__(self, det, elec, run, sub, evt, trigger, t_s, t_ns):
        self.det = det
        self.elec = elec
        self.run_nb  = run
        self.sub  = sub
        self.evt_nb  = evt
        self.trigger_nb = trigger
        self.time_s = t_s
        self.time_ns = t_ns
        self.charge_time_s = t_s
        self.charge_time_ns = t_ns        
        self.n_hits = np.zeros((cf.n_view), dtype=int)
        self.n_tracks2D = np.zeros((cf.n_view), dtype=int)
        self.n_tracks3D = 0
        self.n_single_hits = 0
        self.n_ghosts = 0
        self.noise_raw = None
        self.noise_filt = None
        self.noise_study = None
        self.noise_pds = None

        self.pds_time_s = t_s
        self.pds_time_ns = t_ns
        self.n_pds_peak = np.zeros((cf.n_pds_channels), dtype=int)
        
    def set_noise_raw(self, noise):
        self.noise_raw = noise

    def set_noise_filt(self, noise):
        self.noise_filt = noise

    def set_noise_study(self, noise):
        self.noise_study = noise

    def set_noise_pds(self, noise):
        self.noise_pds = noise

    def set_pds_timestamp(self, t_s, t_ns):
        self.pds_time_s = t_s
        self.pds_time_ns = t_ns

    def set_charge_timestamp(self, t_s, t_ns):
        self.charge_time_s = t_s
        self.charge_time_ns = t_ns

        
    def dump(self):
        print("RUN ",self.run_nb, " of ", self.elec, " EVENT ", self.evt_nb, " TRIGGER ", self.trigger_nb,)
        print("Taken at ", time.ctime(self.time_s), " + ", self.time_ns, " ns ")




class hits:
    def __init__(self, module, view, daq_channel, start, stop, max_t, max_adc, min_t, min_adc, zero_t, signal):
        self.view    = view
        self.module  = module

        """Each hit should have a unique ID per event"""
        self.ID = -1
        self.daq_channel = daq_channel
        self.channel = chmap[daq_channel].vchan
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

    def __lt__(self,other):
        """ sort hits by decreasing Z and increasing channel """
        return (self.Z > other.Z) or (self.Z == other.Z and self.X < other.X)

    def set_index(self, idx):
        self.ID = idx + n_tot_hits

    def hit_positions(self, v):
        self.t = self.max_t if self.signal == "Collection" else self.zero_t
        

        """ transforms time bins into distance from the anode """
        """ for CB it does not mean someting concrete """
        """ correct Z with PCB positions """

        self.Z = cf.anode_z[self.module] - cf.drift_direction[self.module]*(v * self.t / cf.sampling - cf.view_z_offset[self.view])
        self.Z_start = cf.anode_z[self.module] - cf.drift_direction[self.module]*(v * self.start /cf.sampling -  cf.view_z_offset[self.view])
        self.Z_stop  = cf.anode_z[self.module] - cf.drift_direction[self.module]*(v * self.stop / cf.sampling - cf.view_z_offset[self.view])



    def hit_charge(self):
        
        self.charge_pos *= chmap[self.daq_channel].gain
        self.charge_neg *= chmap[self.daq_channel].gain


        """ I'm not sure this is correct for the induction hits """
        self.charge = self.charge_pos if self.signal == "Collection" else self.charge_pos + np.fabs(self.charge_neg)


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


    def get_charges(self):
        return (self.charge, self.charge_pos, self.charge_neg)


    def mini_dump(self):
        print('Hit ',self.ID, '(free:',self.is_free,') v',self.view,' ch ', self.channel, ' t:', self.start, ' to ', self.stop, 'max ',self.max_t, ' min ', self.min_t, ' maxADC ', self.max_adc, ' minADc ', self.min_adc, 'matched 2D ', self.match_2D, ' 3D ', self.match_3D, ' dray ', self.match_dray, 'ghost ', self.match_ghost, 'SH ', self.match_SH)

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



class singleHits:
    def __init__(self, ID_SH, n_hits, IDs, x, y, z, d_max_bary, d_min_3D, d_min_2D):
        self.ID_SH = ID_SH
        self.n_hits = n_hits
        self.IDs = IDs
        self.charge_pos = [0.,0.,0.]
        self.charge_extend = [0., 0, 0]
        self.charge_extend_pos = [0., 0, 0]
        self.charge_extend_neg = [0., 0, 0]
        self.charge_neg = [0.,0.,0.]

        self.start   = [-1, -1, -1]
        self.stop   = [-1, -1, -1]

        self.max_t   = [-1, -1, -1]
        self.zero_t   = [-1, -1, -1]
        self.min_t   = [-1, -1, -1]
        self.X = x
        self.Y = y
        self.Z = z
        self.d_bary_max = d_max_bary
        self.d_track_3D = d_min_3D
        self.d_track_2D = d_min_2D

        self.veto = [False, False, False]
        
    def set_veto(self, view, veto, q, p, n):
        self.veto[view] = veto
        self.charge_extend[view] = q
        self.charge_extend_pos[view] = p
        self.charge_extend_neg[view] = n

    def set_view(self, view, charge_pos, charge_neg, start, stop, max_t, zero_t, min_t):
        self.charge_pos[view] = charge_pos
        self.charge_neg[view] = charge_neg

        self.start[view] = start
        self.stop[view] = stop

        self.max_t[view] = max_t
        self.zero_t[view] = zero_t
        self.min_t[view] = min_t


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
        print('Is vetoed ', self.veto)
        print('Charge extended ', self.charge_extend)
        print('Charge extended pos ', self.charge_extend_pos)
        print('Charge extended neg ', self.charge_extend_neg)

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

    def __lt__(self,other):
        """ sort tracks by decreasing Z and increasing channel """
        return (self.path[0][1] > other.path[0][1]) or (self.path[0][1] == other.path[0][1] and self.path[0][0] < other.path[0][0])


    def add_drays(self, x, y, q, ID):
        self.drays.append((x,y,q))
        self.drays_ID.append(ID)
        self.dray_charge += q
        self.n_hits_dray += 1
        self.remove_hit(x, y, q)


    def remove_hit(self, x, y, q):
        pos = -1
        for p,t in enumerate(self.path):
            #The hit ID could be used to locate the corresponding hit instead of x,y,Q - To be implemented
            if(t[0] == x and t[1] == y and self.dQ[p]==q):
                pos = p
                break

        if(pos >= 0):
            self.path.pop(pos)
            self.dQ.pop(pos)
            self.hits_ID.pop(pos)
            self.n_hits -= 1
            self.tot_charge -= q
        else:
            print("?! cannot remove hit ", x, " ", y, " ", q, " pos ", pos)


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

    def reset_path(self, path, dQ):
        self.path = path
        self.dQ = dQ
        self.finalize_track()


    def finalize_track(self):
        if(self.path[-1][1] > self.path[0][1]):

            self.path.reverse()
            self.dQ.reverse()
            self.hits_ID.reverse()
            self.ini_slope, self.end_slope = self.end_slope, self.ini_slope
            self.ini_slope_err, self.end_slope_err = self.end_slope_err, self.ini_slope_err

            self.chi2_fwd, self.chi2_bkwd = self.chi2_bkwd, self.chi2_fwd
            print(self.trackID, " : wrong order check :", self.path[0][1], " to ", self.path[-1][1])
            self.ini_time, self.end_time = self.end_time, self.ini_time

        self.n_hits = len(self.path)
        self.tot_charge = sum(self.dQ)

        self.n_hits_dray = len(self.drays)
        self.dray_charge = sum(k for i,j,k in self.drays)
        
        self.len_straight = math.sqrt( pow(self.path[0][0]-self.path[-1][0], 2) + pow(self.path[0][1]-self.path[-1][1], 2) )
        self.len_path = 0.
        for i in range(self.n_hits-1):
            self.len_path +=  math.sqrt( pow(self.path[i][0]-self.path[i+1][0], 2) + pow(self.path[i][1]-self.path[i+1][1],2) )
        

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

        if(self.path[0][1] > other.path[0][1]):
               self.ini_slope = self.ini_slope
               self.ini_slope_err = self.ini_slope_err
               self.end_slope = other.end_slope
               self.end_slope_err = other.end_slope_err

               self.path.extend(other.path)
               self.dQ.extend(other.dQ)
               self.hits_ID.extend(other.hits_ID)
               self.len_straight = math.sqrt( pow(self.path[0][0]-self.path[-1][0], 2) + pow(self.path[0][1]-self.path[-1][1], 2) )
               self.end_time = other.end_time

        else:
               self.ini_slope = other.ini_slope
               self.ini_slope_err = other.ini_slope_err
               self.end_slope = self.end_slope
               self.end_slope_err = self.end_slope_err

               self.path = other.path + self.path
               self.dQ = other.dQ + self.dQ
               self.hits_ID = other.hits_ID + self.hits_ID
               self.len_straight = math.sqrt( pow(self.path[0][0]-self.path[-1][0], 2) + pow(self.path[0][1]-self.path[-1][1],2) )
               self.ini_time = other.ini_time

    def charge_in_z_interval(self, start, stop):
        return sum([q for q, (x, z) in zip(self.dQ, self.path) if z >= start and z <= stop])

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

    

    def mini_dump(self):
        print("2D track ", self.trackID," in view : ", self.view, " from (%.1f,%.1f)"%(self.path[0][0], self.path[0][1]), " to (%.1f, %.1f)"%(self.path[-1][0], self.path[-1][1]), " N = ", self.n_hits, " L = %.1f/%.1f"%(self.len_straight, self.len_path), " Q = ", self.tot_charge, " Dray N = ", self.n_hits_dray, " Qdray ", self.dray_charge, "3D MATCH : ", self.match_3D)
        #print('track hits ID ', self.hits_ID)
        #print('matches : ', self.matched)
        #print('3D match : ', self.match_3D)


class trk3D:
    def __init__(self):

        self.match_ID  =  [-1]*cf.n_view
        self.ID_3D      = -1

        self.chi2    = [-1]*cf.n_view
        self.momentum = -1

        self.d_match = -1

        self.n_hits   = [-1]*cf.n_view #t.n_hits for t in trks]

        self.len_straight = [-1]*cf.n_view
        self.len_path = [-1]*cf.n_view


        self.tot_charge = [-1]*cf.n_view
        self.dray_charge = [-1]*cf.n_view


        self.ini_theta = -1
        self.end_theta = -1
        self.ini_phi = -1
        self.end_phi = -1

        self.t0_corr = 0.
        self.z0_corr = 0.
        
        
        self.ini_time = cf.n_sample+1
        self.end_time = -1


        ''' track boundaries '''
        self.path = [[] for x in range(cf.n_view)]
        self.dQ = [[] for x in range(cf.n_view)]
        self.hits_ID = [[] for x in range(cf.n_view)]

        self.ds = [[] for x in range(cf.n_view)]

        self.module_ini = -1
        self.module_end = -1

    def set_view(self, trk, path, dq, ds, hits_id, isFake=False):
        view = trk.view
        trk.match_3D = self.ID_3D

        self.path[view]  = path
        self.dQ[view]   = dq
        self.ds[view]   = ds
        self.hits_ID[view] = hits_id
        self.tot_charge[view] = sum(q/s for q,s in zip(dq,ds))

        if(isFake == True):
            self.len_straight[view] = 0.
            self.len_path[view] = 0.
            self.n_hits[view] = 0.
            self.chi2[view] = 0.

        else:
            self.n_hits[view] = len(path)
            self.chi2[view] = trk.chi2_fwd
            self.match_ID[view] = trk.trackID

            self.len_straight[view] = math.sqrt( sum([pow(path[0][i]-path[-1][i], 2) for i in range(3)]))
            self.len_path[view] = 0.

            self.ini_time = trk.ini_time if trk.ini_time < self.ini_time else self.ini_time
            self.end_time = trk.end_time if trk.end_time > self.end_time else self.end_time
            
            for i in range(len(path)-1):
                self.len_path[view] +=  math.sqrt( pow(path[i][0]-path[i+1][0], 2) + pow(path[i][1]-path[i+1][1],2)+ pow(path[i][2]-path[i+1][2],2) )



    def check_views(self):
        n_fake = 0
        for i in range(cf.n_view):
            if(self.match_ID[i] == -1):
                tfake = trk2D(-1, i, -1, -1, -9999., -9999., -9999., 0, -1,0)

                self.set_view(tfake, [(-9999.,-9999.,-9999), (9999., 9999., 9999.)], [0., 0.], [1., 1.],[-1, -1], isFake=True)
                n_fake += 1
        return n_fake

    def set_modules(self, ini, end):
        self.module_ini = ini
        self.module_end = end

    def boundaries(self):
        ''' begining '''
        v_higher = np.argmax([self.path[i][0][2] for i in range(cf.n_view)])
        self.ini_x = self.path[v_higher][0][0]
        self.ini_y = self.path[v_higher][0][1]
        self.ini_z = self.path[v_higher][0][2]

        self.ini_z_overlap = min([self.path[i][0][2] if k >=0 else 99999. for i,k in zip(range(cf.n_view),self.match_ID)])

        ''' end '''
        v_lower = np.argmin([self.path[i][-1][2] for i in range(cf.n_view)])
        self.end_x = self.path[v_lower][-1][0]
        self.end_y = self.path[v_lower][-1][1]
        self.end_z = self.path[v_lower][-1][2]

        self.end_z_overlap = max([self.path[i][-1][2] if k >= 0 else -9999. for i,k in zip(range(cf.n_view),self.match_ID)])

    def set_t0_z0(self, t0, z0):

        self.t0_corr = t0
        self.z0_corr = z0

    def set_angles(self, theta_ini, phi_ini, theta_end, phi_end):
        self.ini_phi = phi_ini
        self.ini_theta = theta_ini
        self.end_phi = phi_end
        self.end_theta = theta_end

    def dump(self):
        print('\n----')
        print('Track with ID ', self.ID_3D)
        print(" From (%.2f, %.2f, %.2f) to (%.2f, %.2f, %.2f)"%(self.ini_x, self.ini_y, self.ini_z, self.end_x, self.end_y, self.end_z))
        print("Time start ", self.ini_time, ' stop ', self.end_time)
        print('z-overlap ', self.ini_z_overlap, ' to ', self.end_z_overlap)
        print("N Hits ", self.n_hits)
        print(" theta, phi: [ini] %.2f ; %.2f"%(self.ini_theta, self.ini_phi), " -> [end] %.2f ; %.2f "%( self.end_theta, self.end_phi))
        print(" Straight Lengths :  ", self.len_straight)
        print(" Path Lengths ", self.len_path)
        print(" Total charges ", self.tot_charge)
        print(" z0 ", self.z0_corr, " t0 ", self.t0_corr)
        print(" MATCHING DISTANCE SCORE : ", self.d_match)
        print('----\n')

class ghost:
    def __init__(self, ghost_id, t2d_id, min_dist, ghost_charge, trk_charge, nhits):#, t3d_id, min_dist, xstart, ystart, zstart):
        self.ghost_ID = ghost_id
        self.trk2D_ID = t2d_id

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
    def __init__(self, glob_ch, chan, start, stop, max_t, max_adc):

        """Each hit should have a unique ID per event"""
        self.ID = -1
        self.glob_ch = glob_ch
        self.channel = chan
        self.start   = start
        self.stop    = stop
        self.pad_start   = start
        self.pad_stop    = stop

        """ time is in time bin number """
        self.max_t   = max_t
        self.charge = 0.        
        self.max_adc = max_adc


    def dump(self):
        print('Glob Channel ', self.glob_ch, ' detector : ', self.channel, ' peak from ', self.start, ' to ', self.stop, ' max at ', self.max_t, ' ADC : ', self.max_adc)
