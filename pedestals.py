import config as cf
import data_containers as dc

import numpy as np


def compute_pedestal_mean():
    return np.mean(dc.data_daq, axis=-1)


def compute_pedestal_rms():
    return np.std(dc.data_daq, axis=-1)


def compute_pedestal_raw():    
    mean, std = compute_pedestal_mean(), compute_pedestal_rms()
    ped = dc.noise( mean, std )
    dc.evt_list[-1].set_noise_raw(ped)

    dc.data_daq -= mean[:,None]


def compute_pedestal():
    ped = dc.noise(compute_pedestal_mean(), compute_pedestal_rms() )
    dc.evt_list[-1].set_noise_filt(ped)


