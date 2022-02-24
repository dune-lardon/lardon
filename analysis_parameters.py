import config as cf
import json as json
import sys

# To validate JSON file: https://jsonlint.com

class params:
    def __init__(self):
       self.ped_amp_sig_fst = 1          # default amplitude trigger threshold for 1st pass signal mask - in RMS 
       self.ped_amp_sig_oth = 1          # default amplitude trigger threshold for other pass signal mask - in RMS 
       self.ped_rise_thr = [5,5,30]      # consecutive positive samples threshold
       self.ped_ramp_thr = [1,1,2]       # consecutive increasing sample threshold
       self.ped_amp_thr  = [2,2,3]       # RMS amplitude threshold
       self.ped_dt_thr   = 100           # trigger window duration in which a signal is looked for
       self.ped_zero_cross_thr = 15      # minimal number of samples after downward zero-crossing to look for upward zero-crossing

       
       self.noise_coh_group = [32]        # coherent noise channel grouping
       self.noise_coh_per_view = False    # apply the per card per view case
       self.noise_coh_capa_weight = False # account for strip length
       self.noise_coh_calibrated = False      # account for channel calibration
       self.noise_fft_freq  = -1          # specific frequency removal (-1 is none)
       self.noise_fft_lcut  = 0.6         # low-pass filter frequency cut


       self.hit_amp_sig     = [3,6,2]    # default amplitude trigger threshold for hit search - in RMS
       self.hit_dt_min      = [10,10,10] # minimal delta t for hit search - in bins
       self.hit_pad_left    = 6          # hit lower side band
       self.hit_pad_right   = 10         # hit upper side band



       self.trk2D_nhits     = 5     #min nb of hits to make a 2D track
       self.trk2D_rcut      = 6.    #search radius (cm) when looking for nearby hits
       self.trk2D_chi2cut   = 8.    #chi2 cut to accept new hit in track
       self.trk2D_yerr      = 0.5   #assumed error on 'y' i.e. position in cm
       self.trk2D_slope_err = 1.    #assumed slope error for begining of track
       self.trk2D_pbeta     = 3.    #assumed pbeta of the track

       self.trk3D_ztol     = 3.    #min z diff of two tracks boundaries when matched
       self.trk3D_qfrac    = 5.    #max charge balance between two trk when matched (unused now)
       self.trk3D_len_min  = 2.    #min trk length to be considered in matching

       self.trk3D_dx_tol   = [2.,2.]    #distance tolerance to detector x-boundaries for timing computation
       self.trk3D_dy_tol   = [2.,2.]    #distance tolerance to detector y-boundaries for timing computation
       self.trk3D_dz_tol   = 2.    #distance tolerance to detector z-upper boundary for timing computation

       self.plt_noise_show        = 0
       self.plt_evt_disp_daq_show = 0
       self.plt_evt_disp_vch_show = 0

       self.plt_noise_zrange= [0,900]                # color scale for noise plots
       self.plt_evt_disp_daq_zrange = [-1000,1000]   # color scale for DAQ channels    event display plots
       self.plt_evt_disp_vch_ind_zrange = [-100,100] # color scale for induction  view event display plots
       self.plt_evt_disp_vch_col_zrange = [-50,50]   # color scale for collection view event display plots

    def read(self,elec="top",config="1"):
       with open(cf.lardon_path+'/settings/analysis_parameters.json','r') as f:
             #print("AnalysisParameters: Loading analysis setting file: settings/analysis_parameters.json ... ", end='')
             data = json.load(f)
             #print("done")
             if(config not in data):
                print("WARNING: Analysis configuration ",config," not found.")
                print("         Default thresholds will be applied.")
             else:
                self.ped_amp_sig_fst = data[config][elec]['pedestal']['first_pass_thrs']
                self.ped_amp_sig_oth = data[config][elec]['pedestal']['other_pass_thrs']
                self.ped_rise_thr    = data[config][elec]['pedestal']['rise_thr']
                self.ped_ramp_thr    = data[config][elec]['pedestal']['ramp_thr']
                self.ped_amp_thr     = data[config][elec]['pedestal']['amp_thr']
                self.ped_dt_thr      = data[config][elec]['pedestal']['dt_thr']
                self.ped_zero_cross_thr = data[config][elec]['pedestal']['zero_cross_thr']

                self.noise_coh_group = data[config][elec]['noise']['coherent']['groupings']
                self.noise_coh_per_view = bool(data[config][elec]['noise']['coherent']['per_view'])
                self.noise_coh_capa_weight = bool(data[config][elec]['noise']['coherent']['capa_weight'])
                self.noise_coh_calibrated = bool(data[config][elec]['noise']['coherent']['calibrated'])

                self.noise_fft_freq  = data[config][elec]['noise']['fft']['freq']
                self.noise_fft_lcut  = data[config][elec]['noise']['fft']['low_cut']

                self.hit_amp_sig     = data[config][elec]['hit_finder']['amp_sig_thrs']
                self.hit_dt_min      = data[config][elec]['hit_finder']['dt_min']
                self.hit_pad_left    = data[config][elec]['hit_finder']['pad']['left']
                self.hit_pad_right   = data[config][elec]['hit_finder']['pad']['right']


                self.trk2D_nhits     = data[config][elec]['track_2d']['min_nb_hits']
                self.trk2D_rcut      = data[config][elec]['track_2d']['rcut']
                self.trk2D_chi2cut   = data[config][elec]['track_2d']['chi2cut']
                self.trk2D_yerr      = data[config][elec]['track_2d']['y_error']
                self.trk2D_slope_err = data[config][elec]['track_2d']['slope_error']
                self.trk2D_pbeta     = data[config][elec]['track_2d']['pbeta']


                
                self.trk3D_ztol     = data[config][elec]['track_3d']['ztol']
                self.trk3D_qfrac    = data[config][elec]['track_3d']['qfrac']
                self.trk3D_len_min  = data[config][elec]['track_3d']['len_min']

                self.trk3D_dx_tol   = data[config][elec]['track_3d']['dx_tol']
                self.trk3D_dy_tol   = data[config][elec]['track_3d']['dy_tol']
                self.trk3D_dz_tol   = data[config][elec]['track_3d']['dz_tol']

                self.plt_noise_show              = data[config][elec]['plot']['noise']['show']
                self.plt_noise_zrange            = data[config][elec]['plot']['noise']['zrange']
                self.plt_evt_disp_daq_show       = data[config][elec]['plot']['evt_display']['daqch']['show']
                self.plt_evt_disp_daq_zrange     = data[config][elec]['plot']['evt_display']['daqch']['zrange']
                self.plt_evt_disp_vch_show       = data[config][elec]['plot']['evt_display']['viewch']['show']
                self.plt_evt_disp_vch_ind_zrange = data[config][elec]['plot']['evt_display']['viewch']['ind_zrange']
                self.plt_evt_disp_vch_col_zrange = data[config][elec]['plot']['evt_display']['viewch']['col_zrange']

    # setters and getters (potentially not useful now)
    def set_ped_amp_sig_fst(self,value):
      self.ped_amp_sig_fst = values 

    def set_ped_amp_sig_oth(self,value):
      self.ped_amp_sig_oth = values 

    def get_ped_amp_fst(self):
      return self.ped_amp_sig_fst 

    def get_ped_amp_oth(self):
      return self.ped_amp_oth_fst 

    def set_hit_amp_sig(self,values):
      self.hit_amp_sig = values

    def set_hit_dt_min(self,values):
      self.hit_dt_min = values

    def set_hit_pad_left(self,value):
      self.hit_pad_left = value

    def set_hit_pad_right(self,value):
      self.hit_pad_right = value
    
    def get_hit_amp_sig_thrs(self):
      return self.hit_amp_sig  
    
    def get_hit_dt_min_thrs(self):
      return self.hit_dt_min

    def get_hit_pad_left(self):
      return self.hit_pad_left

    def get_hit_pad_right(self):
      return self.hit_pad_right


    def dump(self):
        print("\n~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ")
        print(" \t ~Reconstruction Parameters used~ ")
        print("~Pedestal/Masking~ ")
        print("    1st pass rms threshold : ", self.ped_amp_sig_fst, " 2nd pass : ", self.ped_amp_sig_oth)
        print("    Nb of consecutive positive sample over threshold ", self.ped_rise_thr)
        print("    Nb of consecutive increasing samples over threshold ", self.ped_ramp_thr)
        print("    Amplitude threshold ", self.ped_amp_thr)
        print("    Window duration ", self.ped_dt_thr)
        print("    Nb of sample in between zero cross ", self.ped_zero_cross_thr)

        print("\n~Noise Removal~ ")
        print("    FFT low pass cut ", self.noise_fft_lcut)
        print("    FFT frequency cut ", self.noise_fft_freq)
        print("    Coherent groups : ", self.noise_coh_group)
        print("    Coh per view case : ", self.noise_coh_per_view)
        print("    Coh with capa weight  : ", self.noise_coh_capa_weight)
        print("    Coh with calibrated ch : ", self.noise_coh_calibrated)

        print("\n~Hit Finder~ ")
        print("    Amplitude RMS threshold ", self.hit_amp_sig)
        print("    Minimum Hit duration in sample ", self.hit_dt_min)
        print("    Hit signal pad left ", self.hit_pad_left, " right ", self.hit_pad_right)

        print("\n~2D Track Finder~ ")
        print("    Min Nb of Hits ", self.trk2D_nhits)
        print("    Hit Search radius ", self.trk2D_rcut)
        print("    Hit chi2 cut when added to the track ", self.trk2D_chi2cut)
        print("    PFilter initial error estimate y: ", self.trk2D_yerr, " slope: ", self.trk2D_slope_err, " pbeta : ", self.trk2D_pbeta)

        print("\n~3D Track Finder~ ")
        print("    Max Z difference at 2D track boundaries : ", self.trk3D_ztol)
        print("    Max track charge balance : ", self.trk3D_qfrac)
        print("    Min track 2D length : ", self.trk3D_len_min)
        print("    Distance to detector x-boundaries : ", self.trk3D_dx_tol, 'y-boundaries: ', self.trk3D_dy_tol, " z-boundary ", self.trk3D_dz_tol)
        print("~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ \n")

