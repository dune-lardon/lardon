import config as cf
import data_containers as dc
import json as json
import sys

# To validate JSON file: https://jsonlint.com

def build_default_reco():
    dc.reco = {
        'pedestal' : {
            "first_pass_thrs":3,
            "other_pass_thrs":4,
            "rise_thr":[5 if cf.view_type[x] == 'Induction' else 30 for x in range(cf.n_view)],
            "ramp_thr":[1 if cf.view_type[x] == 'Induction' else 2 for x in range(cf.n_view)],
            "amp_thr":[2 if cf.view_type[x] == 'Induction' else 3 for x in range(cf.n_view)],
            "dt_thr":100,
            "zero_cross_thr":15
        },    

        'mask':{
            'coll' : {
                "min_dt":15,
                "low_thr":[2.0,3.0],
                "high_thr":[5.0,4.0],
                "min_rise":3,
                "min_fall":10,
                "pad_bef":10,
                "pad_aft":15
            },
            
            'ind':{
                "pad_bef":10,
                "pad_aft":15,
                "max_dt_pos_neg":20,
                "pos" :{
                    "min_dt":15,
                    "low_thr":[2.0,3.0],
                    "high_thr":[5.0,4.0],
                    "min_rise":3,
                    "min_fall":10 #just for testing                    
                },
                "neg":{
                    "min_dt":10,
                    "low_thr":[-1.8,-2.0],
                    "high_thr":[-2.5,-3.5],
                    "min_rise":3,
                    "min_fall":3
                }
            }
        },

        "noise":{
           "coherent":{
              "groupings":[32],
	       "per_view":0,
	       "capa_weight":0,
	       "calibrated":0
           },

           "fft":{
              "freq":-1,
              "low_cut":0.6
           },
           "microphonic":{
              "window":-1
           }

        },

        "hit_finder":{
            "coll":{
                "amp_sig": [3,6],
                "dt_min": 10
                },
            "ind":{
                "amp_sig": [2,3],
                "dt_min": 10
            },
            "pad":{
                "left": 6,
                "right": 10
            }
        },

	"track_2d":{
	    "min_nb_hits": 5,
	    "rcut": 6.0,
	    "chi2cut": 8.0,
	    "y_error": 0.5,
	    "slope_error": 1.0,
	    "pbeta":3.0
	},

	"track_3d":{
	    "ztol":3.0,
	    "qfrac":5.0, 
	    "len_min":2.0, 
	    "dx_tol": [3.0,0.5],
	    "dy_tol": [3.0,3.0],
	    "dz_tol": 2.0
	},

        "ghost":{
            "dmin":10
        },

        "single_hit":{
            "max_per_view":3,
            "outlier_dmax":2,
            "veto_nchan":16,
            "veto_nticks":150,
            "int_nchan":3,
            "int_nticks":50
        },
        
        "plot":{
           "noise":{
              "show": 0,
              "zrange": [0,15]
           },
           "evt_display":{
              "daqch":{
                "show":0,
                "zrange":[-50,50]
              },
              "viewch":{
                "show":0,
                "ind_zrange":[-50,50],
                "col_zrange":[-50,50]
              }
           }
        }
    }
    


def set_param(key, val, ref):

    if(key in ref):
        if(type(val)==dict):
            for k,v in val.items():
                set_param(k,v, ref[key])
        else:
            ref[key] = val
    else:
        dc.reco[key]=val
        print('reco parameter ', key, ' is not an official parameter')


def configure(detector, elec, custom=""):

    the_file = cf.lardon_path+'/settings/'+detector+'_'+elec+'/reco_parameters.json' if custom == "" else custom

    with open(the_file,'r') as f:        
        data = json.load(f)['default']
        
        for k, v in data.items():
            set_param(k,v, dc.reco)


        if(False):#config not in data):
            print("WARNING: Analysis configuration ",config," not found.")
            print("         Default thresholds will be applied.")
            




def dump():
        print("\n~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ")
        print(" \t ~Reconstruction Parameters used~ ")
        print("~Pedestal~ ")
        print("    1st pass rms threshold : ", dc.reco['pedestal']['first_pass_thrs'], " Next pass : ", dc.reco['pedestal']['other_pass_thrs'])
        print("    Nb of consecutive positive sample over threshold ", dc.reco['pedestal']['rise_thr'])
        print("    Nb of consecutive increasing samples over threshold ", dc.reco['pedestal']['ramp_thr'])
        print("    Amplitude rms threshold ", dc.reco['pedestal']['amp_thr'])
        print("    Window duration ", dc.reco['pedestal']['dt_thr'])
        print("    Nb of sample in between zero cross ", dc.reco['pedestal']['zero_cross_thr'])

        print("\n~Masking~")
        print(" - - Collection - -  ")
        print("    Min duration (tick) :", dc.reco['mask']['coll']['min_dt'])
        print("    Low RMS Threshold :", dc.reco['mask']['coll']['low_thr'])
        print("    High RMS Threshold :", dc.reco['mask']['coll']['high_thr'])
        print("    Min Rise Time (tick) :", dc.reco['mask']['coll']['min_rise'])
        print("    Min Fall Time (tick) :", dc.reco['mask']['coll']['min_fall'])
        print("    Pads (tick) : before :", dc.reco['mask']['coll']['pad_bef'], "after ", dc.reco['mask']['coll']['pad_aft'])

        print(" - - Induction - -  ")

        print("    Min duration (tick) :", dc.reco['mask']['ind']['pos']['min_dt'],"/",dc.reco['mask']['ind']['neg']['min_dt'])
        print("    Low RMS Threshold :", dc.reco['mask']['ind']['pos']['low_thr'],"/",dc.reco['mask']['ind']['neg']['low_thr'])
        print("    High RMS Threshold :", dc.reco['mask']['ind']['pos']['high_thr'],"/",dc.reco['mask']['ind']['neg']['high_thr'])
        print("    Min Rise Time (tick) :", dc.reco['mask']['ind']['pos']['min_rise'],"/",dc.reco['mask']['ind']['neg']['min_rise'])
        print("    Min Fall Time (tick) :", dc.reco['mask']['ind']['pos']['min_fall'],"/",dc.reco['mask']['ind']['neg']['min_fall'])        
        print("    Max Pos-Neg dt (tick) :", dc.reco['mask']['ind']['max_dt_pos_neg'])
        print("    Pads (tick) : before :", dc.reco['mask']['ind']['pad_bef'], "after ", dc.reco['mask']['ind']['pad_aft'])

        print("\n~Noise Removal~ ")
        print("    FFT low pass cut ", dc.reco['noise']['fft']['low_cut'])
        print("    FFT frequency cut ", dc.reco['noise']['fft']['freq'])
        print("    Coherent groups : ", dc.reco['noise']['coherent']['groupings'])
        print("    Coh per view case : ", dc.reco['noise']['coherent']['per_view'])
        print("    Coh with capa weight  : ",  dc.reco['noise']['coherent']['capa_weight'])
        print("    Coh with calibrated ch : ",  dc.reco['noise']['coherent']['calibrated'])
        print("    Microphonic window : ", dc.reco['noise']['microphonic']['window'])

        print("\n~Hit Finder~ ")
        print("    Amplitude RMS threshold Coll:", dc.reco['hit_finder']['coll']['amp_sig'], " Ind:", dc.reco['hit_finder']['ind']['amp_sig'])
        print("    Minimum Hit duration in sample Coll: ", dc.reco['hit_finder']['coll']['dt_min'], ', Ind:', dc.reco['hit_finder']['ind']['dt_min'])
        print("    Hit signal pad left ", dc.reco['hit_finder']['pad']['left'], " right ", dc.reco['hit_finder']['pad']['right'])

        print("\n~2D Track Finder~ ")
        print("    Min Nb of Hits ", dc.reco['track_2d']['min_nb_hits'])
        print("    Hit Search radius ", dc.reco['track_2d']['rcut'])
        print("    Hit chi2 cut when added to the track ", dc.reco['track_2d']['chi2cut'])
        print("    PFilter initial error estimate y: ", dc.reco['track_2d']['y_error'], " slope: ", dc.reco['track_2d']['slope_error'], " pbeta : ", dc.reco['track_2d']['pbeta'])

        print("\n~3D Track Finder~ ")
        print("    Max Z difference at 2D track boundaries : ", dc.reco['track_3d']['ztol'])
        print("    Max track charge balance : ", dc.reco['track_3d']['qfrac'])
        print("    Min track 2D length : ", dc.reco['track_3d']['len_min'])
        print("    Distance to detector x-boundaries : ", dc.reco['track_3d']['dx_tol'], 'y-boundaries: ', dc.reco['track_3d']['dy_tol'], " z-boundary ", dc.reco['track_3d']['dz_tol'])

        print("\n~3D Ghost Finder~ ")
        print("    Ghost-Track min distance : ", dc.reco["ghost"]["dmin"])

        print("\n~Single Hit Finder~ ")
        #print("    Time Tolerance ", dc.reco["single_hit"]["time_tol"])
        print("    Max nb of SH hits/view ", dc.reco["single_hit"]["max_per_view"])
        print("    Outlier Dmax ", dc.reco["single_hit"]["outlier_dmax"])
        print("    Veto nchannel ", dc.reco["single_hit"]["veto_nchan"])
        print("    Veto nticks ", dc.reco["single_hit"]["veto_nticks"])
        print("    Integral nchannel ", dc.reco["single_hit"]["int_nchan"])
        print("    Integral nticks ", dc.reco["single_hit"]["int_nticks"])

        print("~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ \n")



class params:
    def __init__(self):
       self.ped_amp_sig_fst = 1          # default amplitude trigger threshold for 1st pass signal mask - in RMS 
       self.ped_amp_sig_oth = 1          # default amplitude trigger threshold for other pass signal mask - in RMS 
       self.ped_rise_thr = [5 if cf.view_type(x) == 'Induction' else 30 for x in range(cf.n_view)]      # consecutive positive samples threshold
       self.ped_ramp_thr =  [1 if cf.view_type(x) == 'Induction' else 2 for x in range(cf.n_view)]       # consecutive increasing sample threshold
       self.ped_amp_thr  =  [2 if cf.view_type(x) == 'Induction' else 3 for x in range(cf.n_view)]        # RMS amplitude threshold
       self.ped_dt_thr   = 100           # trigger window duration in which a signal is looked for
       self.ped_zero_cross_thr = 15      # minimal number of samples after downward zero-crossing to look for upward zero-crossing
       
       
       self.mask_coll_min_dt = 10
       self.mask_coll_low_thr = [2,3]
       self.mask_coll_high_thr = [4,5]
       self.mask_coll_min_rise = 3
       self.mask_coll_min_fall = 10
       self.mask_coll_pad_bef = 10
       self.mask_coll_pad_aft = 15

       self.mask_ind_pos_min_dt = 10
       self.mask_ind_pos_low_thr = [1.8, 2.]
       self.mask_ind_pos_high_thr = [2.5, 3.5]
       self.mask_ind_pos_min_rise = 3
       self.mask_ind_pos_min_fall = 3

       self.mask_ind_neg_min_dt = 10
       self.mask_ind_neg_low_thr = [1.8, 2.]
       self.mask_ind_neg_high_thr = [2.5, 3.5]
       self.mask_ind_neg_min_rise = 3
       self.mask_ind_neg_min_fall = 3


       self.mask_ind_max_dt_pos_neg = 20
       self.mask_ind_pad_bef = 10
       self.mask_ind_pad_aft = 15


       self.noise_coh_group = [32]        # coherent noise channel grouping
       self.noise_coh_per_view = False    # apply the per card per view case
       self.noise_coh_capa_weight = False # account for strip length
       self.noise_coh_calibrated = False      # account for channel calibration
       self.noise_fft_freq  = -1          # specific frequency removal (-1 is none)
       self.noise_fft_lcut  = 0.6         # low-pass filter frequency cut
       self.noise_micro_window  = -1         # microphonic window size


       self.hit_amp_sig     = [3,6,2]    # default amplitude trigger threshold for hit search - in RMS
       self.hit_dt_min      = [10 for x in range(cf.n_view)] # minimal delta t for hit search - in bins
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

       self.ghost_dmin = 10. #min distance between potential track/ghost pair
       
       self.sh_time_tol     = 9. #
       self.sh_outlier_dmax = 5. #
       self.sh_veto_nchan   = 9 #
       self.sh_veto_nticks  = 100 #
       self.sh_int_nchan    = 3 #
       self.sh_int_nticks   = 150 #
       

       self.plt_noise_show        = 0
       self.plt_evt_disp_daq_show = 0
       self.plt_evt_disp_vch_show = 0

       self.plt_noise_zrange= [0,900]                # color scale for noise plots
       self.plt_evt_disp_daq_zrange = [-1000,1000]   # color scale for DAQ channels    event display plots
       self.plt_evt_disp_vch_ind_zrange = [-100,100] # color scale for induction  view event display plots
       self.plt_evt_disp_vch_col_zrange = [-50,50]   # color scale for collection view event display plots


    def read(self,detector='cb',elec='top'):
       with open(cf.lardon_path+'/settings/'+detector+'_'+elec+'/reco_parameters.json','r') as f:

             data = json.load(f)

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



                self.mask_coll_min_dt = data[config][elec]['mask']['coll']['min_dt']
                self.mask_coll_low_thr = data[config][elec]['mask']['coll']['low_thr']
                self.mask_coll_high_thr = data[config][elec]['mask']['coll']['high_thr']
                self.mask_coll_min_rise = data[config][elec]['mask']['coll']['min_rise']
                self.mask_coll_min_fall = data[config][elec]['mask']['coll']['min_fall']
                self.mask_coll_pad_bef = data[config][elec]['mask']['coll']['pad_bef']
                self.mask_coll_pad_aft = data[config][elec]['mask']['coll']['pad_aft']
                
                self.mask_ind_pos_min_dt = data[config][elec]['mask']['ind']['pos']['min_dt']
                self.mask_ind_pos_low_thr = data[config][elec]['mask']['ind']['pos']['low_thr']
                self.mask_ind_pos_high_thr = data[config][elec]['mask']['ind']['pos']['high_thr']
                self.mask_ind_pos_min_rise = data[config][elec]['mask']['ind']['pos']['min_rise']
                self.mask_ind_pos_min_fall = data[config][elec]['mask']['ind']['pos']['min_fall']
                
                self.mask_ind_neg_min_dt = data[config][elec]['mask']['ind']['neg']['min_dt']
                self.mask_ind_neg_low_thr = data[config][elec]['mask']['ind']['neg']['low_thr']
                self.mask_ind_neg_high_thr = data[config][elec]['mask']['ind']['neg']['high_thr']
                self.mask_ind_neg_min_rise = data[config][elec]['mask']['ind']['neg']['min_rise']
                self.mask_ind_neg_min_fall = data[config][elec]['mask']['ind']['neg']['min_fall']


                self.mask_ind_max_dt_pos_neg = data[config][elec]['mask']['ind']['max_dt_pos_neg']
                self.mask_ind_pad_bef = data[config][elec]['mask']['ind']['pad_bef']
                self.mask_ind_pad_aft = data[config][elec]['mask']['ind']['pad_aft']



                self.noise_coh_group = data[config][elec]['noise']['coherent']['groupings']
                self.noise_coh_per_view = bool(data[config][elec]['noise']['coherent']['per_view'])
                self.noise_coh_capa_weight = bool(data[config][elec]['noise']['coherent']['capa_weight'])
                self.noise_coh_calibrated = bool(data[config][elec]['noise']['coherent']['calibrated'])

                self.noise_fft_freq  = data[config][elec]['noise']['fft']['freq']
                self.noise_fft_lcut  = data[config][elec]['noise']['fft']['low_cut']
                self.noise_micro_window  = data[config][elec]['noise']['microphonic']['window']

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


                self.ghost_dmin = data[config][elec]['ghost']['dmin']


                self.sh_time_tol     = data[config][elec]['single_hit']['time_tol']
                self.sh_outlier_dmax = data[config][elec]['single_hit']['outlier_dmax']
                self.sh_veto_nchan   = data[config][elec]['single_hit']['veto_nchan']
                self.sh_veto_nticks  = data[config][elec]['single_hit']['veto_nticks']
                self.sh_int_nchan    = data[config][elec]['single_hit']['int_nchan']
                self.sh_int_nticks   = data[config][elec]['single_hit']['int_nticks']


                self.plt_noise_show              = data[config][elec]['plot']['noise']['show']
                self.plt_noise_zrange            = data[config][elec]['plot']['noise']['zrange']
                self.plt_evt_disp_daq_show       = data[config][elec]['plot']['evt_display']['daqch']['show']
                self.plt_evt_disp_daq_zrange     = data[config][elec]['plot']['evt_display']['daqch']['zrange']
                self.plt_evt_disp_vch_show       = data[config][elec]['plot']['evt_display']['viewch']['show']
                self.plt_evt_disp_vch_ind_zrange = data[config][elec]['plot']['evt_display']['viewch']['ind_zrange']
                self.plt_evt_disp_vch_col_zrange = data[config][elec]['plot']['evt_display']['viewch']['col_zrange']



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

        print("\n~Masking~")
        print(" - - Collection - -  ")
        print("    Min duration (tick) :", self.mask_coll_min_dt)
        print("    Low RMS Threshold :", self.mask_coll_low_thr)
        print("    High RMS Threshold :", self.mask_coll_high_thr)
        print("    Min Rise Time (tick) :", self.mask_coll_min_rise)
        print("    Min Fall Time (tick) :", self.mask_coll_min_fall)
        print("    Pads (tick) : before :", self.mask_coll_pad_bef, "after ", self.mask_coll_pad_aft)
        print(" - - Induction - -  ")
        print("    Min duration (tick) :", self.mask_ind_pos_min_dt,"/",self.mask_ind_neg_min_dt)
        print("    Low RMS Threshold :", self.mask_ind_pos_low_thr,"/",self.mask_ind_neg_low_thr)
        print("    High RMS Threshold :", self.mask_ind_pos_high_thr,"/",self.mask_ind_neg_high_thr)
        print("    Min Rise Time (tick) :", self.mask_ind_pos_min_rise,"/",self.mask_ind_neg_min_rise)
        print("    Min Fall Time (tick) :", self.mask_ind_pos_min_fall,"/",self.mask_ind_neg_min_fall)
        print("    Max Pos-Neg dt (tick) :", self.mask_ind_max_dt_pos_neg)
        print("    Pads (tick) : before :", self.mask_ind_pad_bef, "after ", self.mask_ind_pad_aft)


        print("\n~Noise Removal~ ")
        print("    FFT low pass cut ", self.noise_fft_lcut)
        print("    FFT frequency cut ", self.noise_fft_freq)
        print("    Coherent groups : ", self.noise_coh_group)
        print("    Coh per view case : ", self.noise_coh_per_view)
        print("    Coh with capa weight  : ", self.noise_coh_capa_weight)
        print("    Coh with calibrated ch : ", self.noise_coh_calibrated)
        print("    Microphonic window : ", self.noise_micro_window)

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

        print("\n~3D Ghost Finder~ ")
        print("    Ghost-Track min distance : ", self.ghost_dmin)

        print("\n~Single Hit Finder~ ")
        #print("    Time Tolerance ", self.sh_time_tol)
        #print("    Max nb of hit/view ", 
        print("    Outlier Dmax ", self.sh_outlier_dmax)
        print("    Veto nchannel ", self.sh_veto_nchan)
        print("    Veto nticks ", self.sh_veto_nticks)
        print("    Integral nchannel ", self.sh_int_nchan)
        print("    Integral nticks ", self.sh_int_nticks)

        print("~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ \n")

