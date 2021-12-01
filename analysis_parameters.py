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

       self.noise_coh_group = [32]       # coherent noise channel grouping
       self.noise_fft_freq  = -1         # ??
       self.noise_fft_lcut  = 0.0225     # low-pass filter frequency cut

       self.hit_amp_sig     = [3,6,2]    # default amplitude trigger threshold for hit search - in RMS
       self.hit_dt_min      = [10,10,10] # minimal delta t for hit search - in bins
       self.hit_pad_left    = 6          # hit lower side band
       self.hit_pad_right   = 10         # hit upper side band

       self.plt_noise_show        = 0
       self.plt_evt_disp_daq_show = 0
       self.plt_evt_disp_vch_show = 0

       self.plt_noise_zrange= [0,900]                # color scale for noise plots
       self.plt_evt_disp_daq_zrange = [-1000,1000]   # color scale for DAQ channels    event display plots
       self.plt_evt_disp_vch_ind_zrange = [-100,100] # color scale for induction  view event display plots
       self.plt_evt_disp_vch_col_zrange = [-50,50]   # color scale for collection view event display plots

    def read(self,elec="top",config="1"):
       with open('settings/analysis_parameters.json','r') as f:
             print("AnalysisParameters: Loading analysis setting file: settings/analysis_parameters.json ... ", end='')
             data = json.load(f)
             print("done")
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
                self.noise_fft_freq  = data[config][elec]['noise']['fft']['freq']
                self.noise_fft_lcut  = data[config][elec]['noise']['fft']['low_cut']

                self.hit_amp_sig     = data[config][elec]['hit_finder']['amp_sig_thrs']
                self.hit_dt_min      = data[config][elec]['hit_finder']['dt_min']
                self.hit_pad_left    = data[config][elec]['hit_finder']['pad']['left']
                self.hit_pad_right   = data[config][elec]['hit_finder']['pad']['right']

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


