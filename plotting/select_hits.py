import config as cf
import data_containers as dc


def get_hits_pos(view, selection="True"):
    return [x.X for x in dc.hits_list if x.view == view and eval(selection)]


def get_hits_ch(view, selection="True"):
    return [x.channel for x in dc.hits_list if x.view == view and eval(selection)]

def get_hits_tdc(view, selection="True"):
    return [x.t for x in dc.hits_list if x.view == view and eval(selection)]

def get_hits_z(view, selection="True"):
    return [x.Z for x in dc.hits_list if x.view == view and eval(selection)]

def get_hits_charge(view, selection="True"):
    return [x.charge for x in dc.hits_list if x.view == view and eval(selection)]

def get_hits_adc(view, selection="True"):
    return [x.adc for x in dc.hits_list if x.view == view and eval(selection)]
