import json as json
import config as cf

def configure(detector, period, elec, run):
    """ long term parameters """
    with open('settings/geo_'+detector+'.json','r') as f:
        data = json.load(f)['period_'+period]

        cf.n_view = int(data['n_view'])
        cf.view_name = data['view_name']
        cf.view_type = data['view_type']
        cf.view_angle = [float(x) for x in data['view_angle']]
        cf.view_pitch = [float(x) for x in data['view_pitch']]
        cf.view_nchan = [int(x) for x in data['view_nchan']]
        cf.view_capa  = [float(x) for x in data['view_capa']]
        
        cf.n_tot_channels = int(data[elec]['n_tot_channels'])
        cf.n_sample = int(data[elec]['n_sample'])
        cf.sampling = float(data[elec]['sampling'])
        cf.ADC_to_fC = float(data[elec]['ADC_to_fC'])
        cf.data_path += "/" + data[elec]['sub_path']
        cf.view_offset = [float(x) for x in data[elec]['view_offset']]


    """ shorter term parameters """
    with open('settings/run_'+detector+'.json','r') as f:
        data = json.load(f)[elec]
        run_keys = sorted(list(data.keys()))
        for r in run_keys:
            if(int(r) >= int(run)):
                run_key_set = r
                break


        cf.channel_map = "settings/chmap/"+data[run_key_set]["chmap"]
        try:
            cf.broken_channels = data[run_key_set]["broken_channels"]
        except KeyError:
            print("WARNING: No information available on broken channels.")
