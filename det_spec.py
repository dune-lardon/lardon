import json as json
import config as cf

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
        
        

def configure(detector, elec, run):
    """ long term parameters """
    with open(cf.lardon_path+'/settings/'+detector+'_'+elec+'/geo.json','r') as f:
           
        param = json.load(f)
        to_be_used = is_concerned(param.keys(), int(run))


    
        """ Load the default geometry settings """
        data = param['default']
                
        for k, v in to_be_used.items():
            if(v == True):
                custom = param[k]
                for key, val in custom.items():
                    data[key] = val

                    

        cf.n_view = int(data['n_view'])
        cf.n_module = int(data['n_module'])

        cf.view_name = data['view_name']
        cf.view_type = data['view_type']
        cf.view_angle = [float(x) for x in data['view_angle']]
        cf.view_pitch = [float(x) for x in data['view_pitch']]
        cf.view_nchan = [int(x) for x in data['view_nchan']]
        cf.view_capa  = [float(x) for x in data['view_capa']]
        
        cf.n_tot_channels = int(data['n_tot_channels'])
        cf.n_sample = int(data['n_sample'])
        cf.sampling = float(data['sampling'])
        cf.e_per_ADCtick = float(data['e_per_ADCtick'])
        cf.ADC_per_fC = float(data['ADC_per_fC'])
        cf.data_path += "/" + data['sub_path']

        cf.view_chan_repet  = [int(x) for x in data['view_chan_repet']]
        cf.view_offset_repet  = [[[float(x) for x in xv] for xv in xm] for xm in data['view_offset_repet']]
        
        cf.view_z_offset = [float(x) for x in data['view_z_offset']]
        
        cf.drift_length = float(data['drift_length'])

        cf.anode_z = float(data['anode_z'])
        cf.view_length = [float(x) for x in data["view_length"]]

        cf.view_offset = [[float(x) for x in xv ] for xv in data['view_offset']]
        cf.x_boundaries = [[float(x) for x in xv] for xv in data['x_boundaries']]
        cf.y_boundaries = [[float(x) for x in xv] for xv in data['y_boundaries']]

        cf.strips_length = cf.lardon_path+"/settings/chmap/"+data['strips_length']


        cf.e_drift = float(data["e_drift"])
        cf.channel_map = cf.lardon_path+"/settings/chmap/"+data["chmap"]

        cf.signal_is_inverted = [bool(int(x)) for x in data['signal_is_inverted']]
        if(cf.n_module > 1):
            cf.module_used = [bool(int(x))  for x in data['module_used']]
        else:
            cf.module_used = [True]

        try:
            cf.broken_channels = data["broken_channels"]
        except KeyError:
            print("No broken channels !")
        try : 
            cf.channel_calib = cf.lardon_path+"/settings/calib/"+data['channel_calib']
        except KeyError:
            print('No Channel Calibration available - Constant value will be used')
