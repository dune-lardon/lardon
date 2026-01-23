import os
import lardon.config as cf
import glob


def get_data_path(paths, run, hash_path):
    for key, val in paths.items():
        for r in [run_directory_divided(run), run_directory_simple(run), run_directory_simple(hash_path)]:
            if(os.path.exists(val+'/'+r)==True):
                if(len(glob.glob(val+'/'+r+'/*'+run+'*'))>0):
                    
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


def get_run_directory(run, hash_path):
    for r in [run_directory_divided(run), run_directory_simple(run), run_directory_simple(hash_path)]:

        if(os.path.exists(cf.data_path+'/'+r)==True):
            return r
    return ""

def extract_info_form_file(file_path):
    slash = file_path.rfind('/')+1
    filename = file_path[slash:]
        
    run, sub, flow, writer, server, detector = -1, "-1","-1", "-1", "-1", "none"
    if(filename[:2] == "np"):
        first_under = filename.find('_')
        det = filename[4:first_under]
        if(det == "vd"):
            detector = "pdvd"
        elif(det=="hd"):
            detector = "pdhd"
        elif(det == "vdcoldbox"):
            detector = "cbbot"
        else:
            print("Cannot recognize the detector name, please provide the info with -det option")
            exit()
            
        run_pos = filename.find('_run')+4
        run = int(filename[run_pos:run_pos+6])
        sub_pos = run_pos + 7
        sub    = str(int(filename[sub_pos:sub_pos+4]))
        
        flow_pos = filename.find('_dataflow')
        if(flow_pos>0):
            flow_pos += 9
            flow_len = 9
        else:
            flow_pos = filename.find('-d')
            if(flow_pos>0):
                flow_pos += 2
                flow_len = 2
                
        writer_pos = filename.find('_datawriter_')
        if(writer_pos>0):
            writer_pos += 12
            writer_len = 12
        else:
            writer_pos = filename.find('_dw_')
            if(writer_pos>0):
                writer_pos += 4
                writer_len = 4

        server_pos = filename.find('-s')+2
        ts_pos = len(filename) - 20

        if(flow_pos>0):
            flow = filename[flow_pos:writer_pos-writer_len]
            if(server_pos>0):
                server = filename[server_pos:flow_pos-flow_len]
            
        if(writer_pos>0):            
            writer = filename[writer_pos:ts_pos-1]


    elif("pvdt" in filename):
        detector = "pdvd"
        first_under = filename.find("_")
        pvdt_pos = filename.find('_pvdt')
        run = int(filename[:first_under])
        sub = filename[first_under+1:pvdt_pos]

    else:
        print('Case not recognized ... update the extract_info_form_file function please!')
        

    return run, sub, flow, writer, server, detector

