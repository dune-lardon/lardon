data_path = "/eos/experiment/neutplatform/protodune/rawdata/"

""" User's specific """
lardon_path = "/eos/user/l/lzambell/analysis/coldbox/lardon/"
store_path = lardon_path+"/reco"
plot_path  = lardon_path+"/results"

""" to add in the json file ?? """

LAr_temperature = 87. #K - to check in the slow control 
e_drift = 0.4  #kV/cm - default but should be overwritten in settings

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
e_per_ADCtick=0
channel_calib = ''
ADC_per_fC = 0
elec = "none"
channel_map = ""
broken_channels = []
view_offset = []
view_z_offset = []

view_chan_repet = []
signal_is_inverted = False
strips_length = ''

drift_length = 0.
anode_z = 0.
view_length = []
x_boundaries = []
y_boundaries = []
