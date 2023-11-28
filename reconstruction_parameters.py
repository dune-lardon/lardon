import config as cf
import data_containers as dc
import jsonc as json
import sys

# To validate JSON file: https://jsonlint.com

def build_default_reco():
    try:
        with open(cf.lardon_path+'/settings/default_reco_parameters.json') as f:
            dc.reco = json.load(f)
    except IOError:
        print('ERROR default reconstruction parameters file is missing: ')
        print(cf.lardon_path+'/settings/default_reco_parameters.json')
        print('Please check your lardon!')
        exit()


def set_param(key, val, ref):

    if(key in ref):
        if(type(val)==dict):
            for k,v in val.items():
                set_param(k,v, ref[key])
        else:
            ref[key] = val
    else:
        dc.reco[key]=val
        print('FIY, the reco parameter ', key, ' is not an official parameter ...')


def configure(detector, custom=""):

    the_file = cf.lardon_path+'/settings/'+detector+'/reco_parameters.json' if custom == "" else custom

    try:
        with open(the_file,'r') as f:        
            data = json.load(f)['default']
        
            for k, v in data.items():
                set_param(k,v, dc.reco)


    except IOError:
        print("WARNING: Analysis configuration ",the_file," not found.")
        print("       -> Default thresholds will be applied.")
        




def dump():
        print("\n~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ")
        print(" \t ~Reconstruction Parameters used~ ")
        print("~Pedestal~ ")
        print("    1st pass rms threshold : ", dc.reco['pedestal']["raw_rms_thr"])
  
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


