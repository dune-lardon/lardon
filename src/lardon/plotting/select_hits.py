import lardon.config as cf
import lardon.data_containers as dc
import numpy as np

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
    return [x.max_adc for x in dc.hits_list if x.view == view and eval(selection)]


def get_2dtracks_pos(view, selection="True"):
    return [[p[0] for p in t.path] for t in dc.tracks2D_list if t.view==view and eval(selection)]

def get_2dtracks_z(view, selection="True"):
    return [[p[1] for p in t.path] for t in dc.tracks2D_list if t.view==view and eval(selection)]

def get_2dtracks_ID(view, selection="True"):
    return [t.trackID for t in dc.tracks2D_list if t.view==view and eval(selection)]


def get_3dtracks(view, axis, selection="True"):
    return [[p[axis] for p in t.path[view]] for t in dc.tracks3D_list if eval(selection)] 
    

def get_3dtracks_x(view, selection="True"):
    return get_3dtracks(view, 0, selection)

def get_3dtracks_y(view, selection="True"):
    return get_3dtracks(view, 1, selection)

def get_3dtracks_z(view, selection="True"):
    return get_3dtracks(view, 2, selection)

def get_3dtracks_z_corr(view, selection="True"):
    axis=2
    return  [[p[axis]+t.z0_corr if t.z0_corr < 9999 else p[axis] for p in t.path[view]] for t in dc.tracks3D_list if eval(selection)] 


def get_3dtracks_corr(view, selection="True"):
    axis=2
    return  [[True if t.z0_corr < 9999 else False for p in t.path[view]] for t in dc.tracks3D_list if eval(selection)] 

def get_3dsingle_hits():
    return [(h.X, h.Y, h.Z) for h in dc.single_hits_list]

def get_3dghost():
    return [p for t in dc.ghost_list for p in t.path] 
