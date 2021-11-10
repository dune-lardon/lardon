import json as json
import config as cf

def configure(detector, period, elec, run):
    """ long term parameters """
    with open('settings/geo_'+detector+'.json','r') as f:
        data = json.load(f)['period_'+period]

        cf.n_view = int(data['n_view'])
        cf.view_name = data['view_name']
        cf.view_type = data['view_type']
        cf.view_angle = [int(x) for x in data['view_angle']]
        cf.view_pitch = [float(x) for x in data['view_pitch']]
        cf.n_chan_per_view = [int(x) for x in data['n_chan_per_view']]
        
        cf.n_tot_channels = int(data[elec]['n_tot_channels'])
        cf.n_sample = int(data[elec]['n_sample'])
        cf.sampling = float(data[elec]['sampling'])
        cf.ADC_to_fC = float(data[elec]['ADC_to_fC'])
        cf.data_path += "/" + data[elec]['sub_path']


    """ shorter term parameters """
    with open('settings/run_'+detector+'.json','r') as f:
        data = json.load(f)[elec]
        run_keys = sorted(list(data.keys()))
        for r in run_keys:
            if(int(r) >= int(run)):
                run_key_set = r
                break


        cf.channel_map = "settings/chmap/"+data[run_key_set]["chmap"]
        
