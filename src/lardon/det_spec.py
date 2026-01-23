import jsonc as json
import lardon.config as cf
import os
import lardon.utils.filenames as fname


def is_concerned(key, run):
    to_be_used = {}
    for k in key:
        if(k == 'default'): 
            to_be_used[k]=False
        else:
            r = k.split('-')
            if(run >= int(r[0]) and run <= int(r[1])):
                to_be_used[k]=True
            else:
                to_be_used[k]=False
                
    return to_be_used
        
        

def configure(detector, run, do_pds, hash_path):
    """ configures LARDON to the detector """

    
    """ access to files """    
    with open(cf.lardon_path+'/settings/'+detector+'/path.json','r') as f:
        locations = json.load(f)
        key = fname.get_data_path(locations, run, hash_path)

        if(key == "none"):
            print("the data does not seem to exists in the paths provided in path.json file")
            print("if you have not provided the file full path (option -file), lardon will crash")
            #exit()
        else:
            cf.domain = key
            cf.data_path = locations[key]

        

    """ detector parameters """
    with open(cf.lardon_path+'/settings/'+detector+'/geo.json','r') as f:
           
        param = json.load(f)
        """ some parameters can change with time """
        to_be_used = is_concerned(param.keys(), int(run))


    
        """ Load the default geometry settings """
        data = param['default']
                
        for k, v in to_be_used.items():
            if(v == True):
                custom = param[k]
                for key, val in custom.items():
                    data[key] = val

                    
        cf.tpc_orientation = data['tpc_orientation']
        cf.n_view = int(data['n_view'])
        cf.n_module = int(data['n_module'])

        cf.LAr_temperature = [float(x) for x in data['LAr_temperature']]
        
        cf.view_name = data['view_name']
        cf.view_type = data['view_type']
        cf.module_name = data["module_name"]
        cf.view_angle = [[float(x) for x in tmp] for tmp in data['view_angle']]
        cf.view_pitch = [float(x) for x in data['view_pitch']]
        cf.view_nchan = [int(x) for x in  data['view_nchan']]
        cf.view_capa  = [float(x) for x in data['view_capa']]
        cf.view_length = [float(x) for x in data["view_length"]]
        cf.view_boundaries_min = [[float(x) for x in xv ] for xv in data['view_boundaries_min']]
        cf.view_boundaries_max = [[float(x) for x in xv ] for xv in data['view_boundaries_max']]
        
        cf.n_tot_channels = int(data['n_tot_channels'])
        cf.module_nchan = [int(x) for x in data['module_nchan']]#int(cf.n_tot_channels/cf.n_module)

        cf.n_sample = [int(x) for x in data['n_sample']]
        cf.sampling = [float(x) for x in data['sampling']]
        cf.e_per_ADCtick = [float(x) for x in data['e_per_ADCtick']]

        cf.view_chan_repet  = [int(x) for x in data['view_chan_repet']]
        cf.view_offset_repet  = [[[float(x) for x in xv] for xv in xm] for xm in data['view_offset_repet']]

        cf.view_z_offset = [[float(x) for x in tmp] for tmp in data['view_z_offset']]
        
        cf.drift_length = [float(x) for x in data['drift_length']]

        cf.anode_z = [float(x) for x in data['anode_z']]
        cf.n_drift_volumes = int(data['n_drift_volumes'])

        try:
            cf.unwrappers = [[float(x) for x in xv] for xv in data['unwrappers']]
        except KeyError:
            cf.unwrappers = None

        try:
            cf.inner_coll_plane = [[bool(x) for x in xm] for xm in data['inner_coll_plane']]
        except KeyError:
            cf.inner_coll_plane = None

        try:
            cf.y_cru = float(data['y_cru'])
        except KeyError:
            cf.y_cru = None

            
        cf.x_boundaries = [[float(x) for x in xv] for xv in data['x_boundaries']]
        cf.y_boundaries = [[float(x) for x in xv] for xv in data['y_boundaries']]

        try : 
            cf.strips_length = cf.lardon_path+"/settings/chmap/"+data['strips_length']
        except KeyError:
            print('No strip length available')
            
        


        cf.e_drift = [float(x) for x in data["e_drift"]]
        
        cf.channel_map = cf.lardon_path+"/settings/chmap/"+data["chmap"]

        cf.signal_is_inverted = [bool(int(x)) for x in data['signal_is_inverted']]
        
        cf.module_used = [int(x)  for x in data['module_used']]
        cf.n_module_used = len(cf.module_used)
        cf.module_daqch_start = [int(x) for x in data['module_daqch_start']]

        try:
            cf.broken_channels = data["broken_channels"]
        except KeyError:
            print("No broken channels !")
        try :            
            if(len(data['channel_calib'])>0):
                cf.channel_calib = cf.lardon_path+"/settings/calib/"+data['channel_calib']
        except KeyError:
            print('No Channel Calibration available - Constant value will be used')

        cf.drift_direction = [float(x) for x in data["drift_direction"]]
        cf.elec = [x for x in data["elec"]]
        cf.daq = data["daq"]
        cf.daq_nlinks = [int(x) for x in data['daq_nlinks']]
        cf.daq_links_offset = [int(x) for x in data['daq_links_offset']]

        cf.daq_TRBuilder_number = data['daq_TRBuilder_number']
        cf.daq_link_name = [x for x in data['daq_link_name']]
        
        if(do_pds == True):
            
            cf.n_pds_stream_channels = data['n_pds_stream_channels']
            cf.n_pds_trig_channels = data['n_pds_trig_channels']
            cf.n_pds_tot_channels = cf.n_pds_stream_channels + cf.n_pds_trig_channels
            cf.pds_sampling = data['pds_sampling']
            cf.n_pds_stream_sample = data['n_pds_stream_sample']
            cf.n_pds_trig_sample = data['n_pds_trig_sample']
            cf.pds_channel_map = cf.lardon_path+"/settings/chmap/"+data['pds_channel_map']
            cf.pds_modules_type = [x for x in data['pds_modules_type']]
            cf.pds_n_modules = data['pds_n_modules']
            cf.pds_x_center = [float(x) for x in data['pds_x_center']]
            cf.pds_y_center = [float(y) for y in data['pds_y_center']]
            cf.pds_z_center = [float(z) for z in data['pds_z_center']]
            cf.pds_drift_vol = [int(x) for x in data['pds_drift_vol']]
            cf.pds_x_length = [float(x) for x in data['pds_x_length']]
            cf.pds_y_length = [float(y) for y in data['pds_y_length']]
            cf.pds_z_length = [float(z) for z in data['pds_z_length']]
            cf.pds_daq       = [ x for x in data['pds_daq']]
            cf.pds_daq_link_name   = [ x for x in data['pds_daq_link_name']]
            cf.pds_daq_link_offset = [int(x) for x in data['pds_daq_link_offset']]
            cf.pds_daq_link_nstream = [int(x) for x in data['pds_daq_link_nstream']]
            cf.pds_daqch_stream_start = int(data['pds_daqch_stream_start'])
            cf.pds_daqch_trig_start = int(data['pds_daqch_trig_start'])
            

            
                  
