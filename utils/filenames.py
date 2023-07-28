import os
import config as cf



def get_data_path(paths):
    for key, val in paths.items():
        if(os.path.exists(val)==True):
            return key
    return "none"



def run_directory_divided(run):
    r = int(run)
    long_run = f'{r:08d}'
    run_path = ""
    for i in range(0,8,2):
        run_path += long_run[i:i+2]+"/"
    return run_path

def run_directory_simple(run):
    return str(run)


def get_run_directory(run):
    
    for r in [run_directory_divided(run), run_directory_simple(run)]:
        if(os.path.exists(cf.data_path+'/'+r)==True):
            return r
    return ""
