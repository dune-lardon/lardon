data_path = "/eos/experiment/neutplatform/protodune/rawdata/"

""" User's specific """
store_path = "./reco"
plot_path  = "./results"

""" to add in the json file ?? """

LAr_temperature = 87. #K - to check in the slow control 
e_drift = 0.500 #kV/cm - could be run specific ?

""" should go in the json file """



""" default values overwritten by the json file """
n_view = -1
view_name = []
view_type = []
view_angle = []
view_pitch = []
view_nchan = []
view_capa = []
n_tot_channels = -1
sampling = 0
n_sample = 0
ADC_to_fC = 0
elec = "none"
channel_map = ""
broken_channels = []
view_offset = []
view_chan_repet = []

drift_length = 0.
anode_z = 0.
view_length = []

