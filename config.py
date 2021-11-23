data_path = "/eos/experiment/neutplatform/protodune/rawdata/"

""" User's specific """
store_path = "./reco"
plot_path  = "./results"

""" to add in the json file ?? """

LAr_temperature = 87. #K - to check in the slow control 
e_drift = 0.500 #kV/cm - could be run specific ?

drift_length = 23. #cm # ????


#to be discussed & checked!
anode_z = 11.5 #cm
len_det_x = 300. #cm
len_det_y = 300. #cm


""" default values overwritten by the json file """
n_view = -1
view_name = []
view_type = []
view_angle = []
view_pitch = []
view_nchan = []
n_tot_channels = -1
sampling = 0
n_sample = 0
ADC_to_fC = 0
elec = "none"
channel_map = ""
broken_channels = []
view_offset = []




