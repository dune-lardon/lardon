import config as cf
import data_containers as dc

import numpy as np

def compute_pedestal(wf_noise,noise_type='None'):
    mean = np.mean(wf_noise, axis=-1)
    std = np.std(wf_noise, axis=-1)
    ped = dc.noise( mean, std )

    if(noise_type=='raw'):
      dc.evt_list[-1].set_noise_raw(ped)
      dc.data_daq -= mean[:,None]
    elif(noise_type=='filt'): dc.evt_list[-1].set_noise_filt(ped)
    else:
      print('ERROR: You must set a noise type for pedestal setting')
      sys.exit()

