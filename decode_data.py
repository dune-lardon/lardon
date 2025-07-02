import config as cf
import data_containers as dc
import numpy as np
import glob as glob

import utils.wib_daq as wib
import utils.lyon_daq as lyon
import utils.cern_daq as cern
import utils.filenames as fname

import channel_mapping as cmap



class decoder:
    def __init__(self, det, run, sub, flow_writer="0-0", hash_path='', filename=None):
        self.run = run
        self.sub = sub
        self.filename = filename
        self.det = det
        self.daq = cf.daq
        self.flow = flow_writer[:flow_writer.find("-")]
        self.writer = flow_writer[flow_writer.find("-")+1:]
        self.hash_path = hash_path



    def get_file_path(self):
        run_path = fname.get_run_directory(self.run, self.hash_path)


        if("cern"  in self.daq):
            path = f"{cf.data_path}/{run_path}/"
            fl = glob.glob(path+"*.bin")
            s = int(self.sub)
            if(len(fl) == 0 or len(fl) < s):
                print('file numbering do not match ...')#
                print('--> ', path, " contains ", len(fl), " files")
                exit()
            f = fl[s]
            return f


        if("lyon" in self.daq):
            path = f"{cf.data_path}/{run_path}/{self.run}_{self.sub}"
            fl = glob.glob(path+"_*")
            

        else:
            path = cf.data_path + "/" + run_path
            r = int(self.run)
            s = int(self.sub)
            long_sub = f'{s:04d}'
            sub_name = 'run'+str(f'{r:06d}')+'_*'+long_sub+'_'

            if(self.daq == "wib_2" or self.daq == "wib_2_eth"):
                app_name = "dataflow"+self.flow+"_datawriter_"+self.writer
                if(self.det == "cbbot" and r >= 37004):
                    app_name = "df-s02-d"+self.flow+"_dw_"+self.writer+"_"
            else:
                app_name = ""
            fl = glob.glob(path+"/*"+sub_name+"*"+app_name+"*hdf5")
            print(path+"/*"+sub_name+"*"+app_name+"*hdf5")
        if(len(fl) != 1):
            print('None or more than one file matches ... : ', fl)
            exit()

        return fl[0]

    def open_file(self):
        f = self.filename if self.filename else self.get_file_path()
        self.filename = f

        self.daq_decoder = (wib.wib if 'wib' in self.daq else lyon.lyon if 'lyon' in self.daq else cern.cern)(f, self.daq, self.det, self.run)

        if('cern' in self.daq):
            self.daq_decoder.set_run(self.run)
        
    def read_run_header(self):
        nb_event = self.daq_decoder.read_run_header()
        return nb_event

    def read_evt_header(self, ievt):
        self.daq_decoder.read_evt_header(self.sub, ievt, self.flow)

    def read_evt(self, ievt):
        self.daq_decoder.read_evt(ievt)

    def read_pds_evt(self, ievt):
        self.daq_decoder.read_pds_evt(ievt)

    def close_file(self):
        self.daq_decoder.close_file()
        print('file closed!')

