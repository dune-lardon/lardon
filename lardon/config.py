import os
""" User's specific """

""" Where is LARDON """
lardon_path = os.environ.get("LARDON_PATH")
#"/exp/dune/app/users/lzambell/analysis/lardon/lardon"#"."

""" Where to store the reconstructed output file """
store_path = os.environ.get("LARDON_RECO")

""" Where to store the control plots """
plot_path  = lardon_path+"/results"



""" General variable overwritten by the detector settings """
data_path = ""
domain = ""

LAr_temperature = 89. #K - to check in the slow control 
e_drift = []  #in kV/cm

""" default values overwritten by the json file """
tpc_orientation = ""
n_view = -1
n_module = 1
module_used = []
n_module_used = 0
module_name = []
module_daqch_start = []
view_name = []
view_type = [[]]
view_angle = [[]]
view_pitch = []
view_nchan = []
view_capa = []
n_tot_channels = -1
module_nchan = []
sampling = []
n_sample = [1000]
e_per_ADCtick = []
channel_calib = ''
elec = []
channel_map = ""
broken_channels = []
view_boundaries_min = [[]]
view_boundaries_max = [[]]
view_z_offset = [[]]

view_chan_repet = [[]]
view_offset_repet = [[[]]]
unwrappers = None
signal_is_inverted = []
strips_length = ''

drift_length = []
anode_z = []
view_length = []
x_boundaries = [[]]
y_boundaries = [[]]
drift_direction = []
elec = []
daq = ""
daq_nlinks = []
daq_links_offset = []
daq_TRBuilder_number = 0
daq_link_name=[]
inner_coll_plane = []
imod = 0
daq_shift = 0
y_cru = -1

""" pds variables """
n_pds_channels = 1
pds_sampling = 0
n_pds_sample = 0
pds_channel_map = ""
pds_n_modules=0
pds_modules_type = []
pds_x_centers = []
pds_y_centers = []
pds_z_centers = []
pds_length = -1
