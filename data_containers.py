import numpy as np
import config as cf
import time
import math

chmap = []
evt_list = []
hits_list = []
tracks2D_list = []
tracks3D_list = []


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



# Maybe these two containers are completely useless ?
data = np.zeros((cf.n_view, max(cf.view_nchan), cf.n_sample), dtype=np.float32)
mask = np.ones((cf.n_view, max(cf.view_nchan), cf.n_sample), dtype=bool)




def reset_event():
    mask_daq[:,:] = True
    data_daq[:,:] = 0.

    data[:,:,:] = 0.
    mask[:,:,:] = True

    hits_list.clear()
    tracks2D_list.clear()
    tracks3D_list.clear()
    evt_list.clear()

class channel:
    def __init__(self, daqch, globch, view, vchan, length, capa, tot_length, tot_capa):
        self.daqch = daqch
        self.globch = globch
        self.view = view
        self.vchan = vchan
        self.length = length
        self.tot_length = tot_length
        self.capa   = capa
        self.tot_capa   = tot_capa

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
        print("RUN ",self.run_nb, " of ", self.elec, " EVENT ", self.evt_nb, " TRIGGER ", self.trigger_nb,)
        print("Taken at ", time.ctime(self.time_s), " + ", self.time_ns, " ns ")




class hits:
    def __init__(self, view, daq_channel, start, stop, charge_int, max_t, max_adc, min_t, min_adc):

        # self.idx = -1
        self.view    = view

        """Each hit should have a unique ID per event"""
        self.ID = -1
        self.daq_channel = daq_channel
        self.channel = chmap[daq_channel].vchan
        self.start   = start
        self.stop    = stop
        self.Z_start = -1
        self.Z_stop  = -1


        """ time is in time bin number """
        self.max_t   = max_t
        self.min_t   = min_t

        self.t = 0.

        self.charge_int  = charge_int
        self.charge_max  = 0.
        self.charge_min  = 0.
        self.charge_pv   = 0. #peak-valley

        self.charge = 0.

        self.max_adc = max_adc
        self.min_adc = min_adc
        self.adc = 0.


        self.cluster = -1
        self.X       = -1
        self.Z       = -1
        self.matched = -9999

        self.ped_bef = -1
        self.ped_aft = -1

    def __lt__(self,other):
        """ sort hits by decreasing Z and increasing channel """
        return (self.Z > other.Z) or (self.Z == other.Z and self.X < other.X)

    def set_index(self, idx):
        self.ID = idx

    def hit_positions(self, v):
        self.t = self.max_t if cf.view_type[self.view] == "Collection" else self.min_t

        """ trick because view Y is separated into 2 sub-volumes in CB """
        self.X = self.channel%cf.view_chan_repet[self.view] * cf.view_pitch[self.view] + cf.view_offset[self.view]

        """ transforms time bins into distance from the anode """
        """ for CB it does not mean someting concrete """
        self.Z = cf.anode_z - v * self.t / cf.sampling

        self.Z_start = cf.anode_z - v * self.start /cf.sampling
        self.Z_stop = cf.anode_z - v * self.stop / cf.sampling


    def hit_charge(self):
        self.adc = self.max_adc if cf.view_type[self.view] == "Collection" else self.min_adc

        """ difference in TDE and BDE has to be understood """

        self.charge_int = self.charge_int / cf.sampling / cf.ADC_per_fC

        self.charge_max = (self.max_adc) / cf.sampling / cf.ADC_per_fC
        self.charge_min = (self.min_adc) / cf.sampling / cf.ADC_per_fC
        self.charge_pv  = (self.max_adc - self.min_adc) / cf.sampling / cf.ADC_per_fC

        self.charge = self.charge_int  if cf.view_type[self.view] == "Collection" else self.charge_min




    def set_match(self, ID):
        self.matched = ID

    def set_cluster(self, ID):
        self.cluster = ID

    def set_ped(self, bef, aft):
        self.ped_bef = bef
        self.ped_aft = aft

    def get_charges(self):
        return (self.charge_int, self.charge_max, self.charge_min, self.charge_pv)


    def dump(self):
        print("\n**View ", self.view, " Channel ", self.channel, " ID: ", self.ID)

        print(" from t ", self.start, " to ", self.stop, " dt = ", self.stop-self.start)
        print(" tmax ", self.max_t, " tmin ", self.min_t, ' dt = ', self.min_t-self.max_t)
        print(" ref time ", self.t)

        print(" positions : ", self.X, ", ", self.Z)
        print(" adc max ", self.max_adc, " adc min ", self.min_adc)
        print(" charges ", self.charge_int, " ", self.charge_max, " ", self.charge_min, " ", self.charge_pv)




class trk2D:
    def __init__(self, ID, view, ini_slope, ini_slope_err, x0, y0, t0, q0, chi2, hit_ID, cluster):
        self.trackID = ID
        self.view    = view

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

        self.tot_charge = q0
        self.dray_charge = 0.

        self.len_straight = 0.
        self.len_path = 0.

        self.matched = [-1 for x in range(cf.n_view)]
        self.cluster = cluster

        self.ini_time = t0
        self.end_time = t0

    def __lt__(self,other):
        """ sort tracks by decreasing Z and increasing channel """
        return (self.path[0][1] > other.path[0][1]) or (self.path[0][1] == other.path[0][1] and self.path[0][0] < other.path[0][0])


    def add_drays(self, x, y, q):
        self.drays.append((x,y,q))
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


    def add_hit(self, x, y, q, t, id):
        self.n_hits += 1

        self.len_path += math.sqrt( pow(self.path[-1][0]-x, 2) + pow(self.path[-1][1]-y,2) )
        #beware to append (x,y) after !
        self.path.append((x,y))
        self.dQ.append(q)
        self.hits_ID.append(id)
        self.tot_charge += q
        self.len_straight = math.sqrt( pow(self.path[0][0]-self.path[-1][0], 2) + pow(self.path[0][1]-self.path[-1][1], 2) )
        self.end_time = t

    def add_hit_update(self, slope, slope_err, x, y, t, q, id, chi2):
        self.end_slope = slope
        self.end_slope_err = slope_err
        self.n_hits += 1
        self.len_path += math.sqrt( pow(self.path[-1][0]-x, 2) + pow(self.path[-1][1]-y,2) )

        #beware to append (x,y) after !
        self.path.append((x,y))
        self.dQ.append(q)
        self.hits_ID.append(id)
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

    def mini_dump(self):
        print("view : ", self.view, " from (%.1f,%.1f)"%(self.path[0][0], self.path[0][1]), " to (%.1f, %.1f)"%(self.path[-1][0], self.path[-1][1]), " N = ", self.n_hits, " L = %.1f/%.1f"%(self.len_straight, self.len_path), " Q = ", self.tot_charge, " Dray N = ", self.n_hits_dray, " Qdray ", self.dray_charge)



class trk3D:
    def __init__(self):

        self.match_ID  =  [-1]*cf.n_view#[t.ID for t in trks]
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

        #self.ini_time = min([t.ini_time for t in trks])
        #self.end_time = max([t.end_time for t in trks])


        ''' track boundaries '''
        self.path = [[] for x in range(cf.n_view)]
        self.dQ = [[] for x in range(cf.n_view)]
        self.hits_ID = [[] for x in range(cf.n_view)]

        self.ds = [[] for x in range(cf.n_view)]


    def set_view(self, trk, path, dq, ds, hits_id, isFake=False):
        view = trk.view

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

            for i in range(len(path)-1):
                self.len_path[view] +=  math.sqrt( pow(path[i][0]-path[i+1][0], 2) + pow(path[i][1]-path[i+1][1],2)+ pow(path[i][2]-path[i+1][2],2) )



    def check_views(self):
        n_fake = 0
        for i in range(cf.n_view):
            if(self.match_ID[i] == -1):
                tfake = trk2D(-1, i, -1, -1, -9999., -9999., -9999., 0, -1,0, -1)
                self.set_view(tfake, [(-9999.,-9999.,-9999), (9999., 9999., 9999.)], [0., 0.], [1., 1.],[-1, -1], isFake=True)
                n_fake += 1
        return n_fake

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
        print(" From (%.2f, %.2f, %.2f) to (%.2f, %.2f, %.2f)"%(self.ini_x, self.ini_y, self.ini_z, self.end_x, self.end_y, self.end_z))
        print('z-overlap ', self.ini_z_overlap, ' to ', self.end_z_overlap)
        print("N Hits ", self.n_hits)
        print(" theta, phi: [ini] %.2f ; %.2f"%(self.ini_theta, self.ini_phi), " -> [end] %.2f ; %.2f "%( self.end_theta, self.end_phi))
        print(" Straight Lengths :  ", self.len_straight)
        print(" Path Lengths ", self.len_path)
        print(" Total charges ", self.tot_charge)
        print(" z0 ", self.z0_corr, " t0 ", self.t0_corr)
        print(" MATCHING DISTANCE SCORE : ", self.d_match)
        print('----\n')
