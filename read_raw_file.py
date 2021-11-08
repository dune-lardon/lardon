import config as cf
import numpy as np
import numba as nb
import os
import glob as glob
import sys
from abc import ABC, abstractmethod



@nb.jit
def read_evt_uint12_nb(data):
    #tt = np.frombuffer(data, dtype=np.uint8)
    #assert np.mod(tt.shape[0],3)==0

    out=np.empty(len(data)//3*2,dtype=np.uint16)

    for i in nb.prange(len(data)//3):
        fst_uint8=np.uint16(data[i*3])
        mid_uint8=np.uint16(data[i*3+1])
        lst_uint8=np.uint16(data[i*3+2])

        out[i*2]   = (fst_uint8 << 4) + (mid_uint8 >> 4)
        out[i*2+1] = ((mid_uint8 % 16) << 8) + lst_uint8

    return out




class decoder(ABC):
     
    @abstractmethod
    def open_file(self):
        pass

    @abstractmethod
    def read_run_header(self):
        pass

    @abstractmethod
    def read_evt_header(self, i):
        pass

    @abstractmethod
    def read_evt(self, i):
        pass

    @abstractmethod
    def close_file(self):
        pass


class top_decoder(decoder):
    def __init__(self, run, sub):
        self.run = run
        self.sub = sub
        print(' -- reading a top drift electronics file')
        
        """ some TDE specific parameters """
        self.evskey = 0xFF
        self.endkey = 0xF0
        self.evdcard0 = 0x5 #number of cards disconnected -- maybe too CB oriented ?

        self.header_type = np.dtype([
            ('k0','B'),
            ('k1','B'),
            ('run_num', '<u4'), 
            ('run_flag', 'c'),
            ('trig_type', '<B'),
            ('padding', '<3c'),
            ('trig_num', '<u4'),
            ('time_s', '<i8'),
            ('time_ns', '<i8'), 
            ('evt_flag', '<B'), 
            ('evt_num', '<u4'),
            ('lro', '<u4'),
            ('cro', '<u4')
        ])

        self.header_size = self.header_type.itemsize

    def open_file(self):
        path = cf.data_path + "/" + self.run + "/" + self.run + "_" + self.sub
        fl = glob.glob(path+"*")
        if(len(fl) != 1):
            print('none or more than one file matches ... : ', fl)
            sys.exit()
        f = fl[0]
        print('Reconstructing ', f)

        f_type = f[f.rfind('.')+1:]
        print(' NB : file type is ', f_type)
        self.f_in = open(f,'rb')
        


    def read_run_header(self):
        run_nb, nb_evt = np.fromfile(self.f_in, dtype='<u4', count=2)
        print('run: ', run_nb, ' nb of events ', nb_evt)

        """ Read the run header of the binary data file """
        self.sequence = []
        for i in range(nb_evt):
            seq  = np.fromfile( self.f_in, dtype='<u4', count=4)
            """4 uint of [event number - event total size with header- event data size - 0]"""
            self.sequence.append(seq[1])
            

        self.event_pos = []
        self.event_pos.append( self.f_in.tell() )
        for i in range(nb_evt-1):
            self.f_in.seek(self.sequence[i], 1)
            """ get the byte position of each event """
            self.event_pos.append( self.f_in.tell() ) 
            """ End of run header reading part """

        return nb_evt


    def read_evt_header(self, i):
        self.f_in.seek(self.event_pos[i],0)

        head = np.frombuffer(self.f_in.read(self.header_size), dtype=self.header_type)
        print(head)

        if( not((head['k0'][0] & 0xFF)==self.evskey and (head['k1'][0] & 0xFF)==self.evskey)):
            print(" problem in the event header ")
        good_evt = (head['evt_flag'][0] & 0x3F) == self.evdcard0

        if(not good_evt):
            print(" problem, the event has ", head['evt_flag'][0] & 0x3F, ' cards disconnected instead of ', self.evdcard0)

        self.lro = head['lro'][0]
        self.cro = head['cro'][0]

    def read_evt(self, i):
        print('hello', i, self.lro, self.cro)
        if(self.lro < 0 or self.cro < 0):
            print(' please read the event header first! ')
            sys.exit()


        idx = self.event_pos[i] + self.header_size
        self.f_in.seek(idx,0)
        out = read_evt_uint12_nb( self.f_in.read(self.cro) )


        if(len(out)/cf.n_sample != self.n_tot_channels):
            print(' The event is incomplete ... ')
            sys.exit()

        dc.data_daq = np.reshape(out, (self.n_tot_channels, self.n_sample))
        
        self.lro = -1
        self.cro = -1


    def close_file(self):
        self.f_in.close()
        print('file closed!')




class bot_decoder(decoder):
    def __init__(self, run, sub):
        self.run = run
        self.sub = sub
        print(' -- reading a bottom drift electronics file')

        """ bde specific parameters """

        self.trigger_header_type = np.dtype([
            ('header_marker','<u4'),    #4
            ('header_version','<u4'),   #8
            ('trig_num', '<u8'),        #16    
            ('timestamp', '<u8'),       #24
            ('n_component', '<u8'),     #32
            ('run_nb', '<u4'),          #36
            ('error_bit', '<u4'),       #40
            ('trigger_type', '<u2'),    #42
            ('sequence_nb', '<u2'),     #44
            ('max_sequence_nb', '<u2'), #46
            ('unused', '<u2')           #48
        ])

        self.trigger_header_size = self.trigger_header_type.itemsize
        
        self.component_header_type = np.dtype([
            ('version','<u4'),      #4
            ('unused','<u4'),       #8
            ('geo_version','<u4'),  #12
            ('geo_sys','<u2'),      #14
            ('geo_region','<u2'),   #16
            ('geo_ID','<u4'),       #20
            ('geo_unused','<u4'),   #24
            ('window_begin','<u8'), #32
            ('window_end','<u8')    #40
        ])
        self.component_header_size = self.component_header_type.itemsize

        self.fragment_header_type = np.dtype([
            ('frag_marker','<u4'),  #4
            ('frag_version','<u4'), #8
            ('frag_size', '<u8'),   #16
            ('trig_num', '<u8'),    #24
            ('timestamp', '<u8'),   #32
            ('tbegin', '<u8'),      #40
            ('tend','<u8'),         #48
            ('run_nb','<u4'),       #52
            ('error_bit','<u4'),    #56
            ('frag_type','<u4'),    #60
            ('sequence_nb', '<u2'), #62
            ('unused','<u2'),       #64
            ('geo_version','<u4'),  #68
            ('geo_sys','<u2'),      #70
            ('geo_region','<u2'),   #72
            ('geo_ID','<u4'),       #76    
            ('geo_unused','<u4')    #80
        ]) 
        
        self.fragment_header_size = self.fragment_header_type.itemsize

        self.wib_header_type = np.dtype([
            ('sof','<u1'),       #1
            ('ver_fib','<u1'),   #2
            ('crate_slot','<u1'),#3
            ('res','<u1'),       #4
            ('mm_oos_res','<u2'),#6
            ('wib_err','<u2'),   #8
            ('ts1','<u4'),       #12
            ('ts2','<u2'),       #14
            ('count_z','<u2')    #16
        ])
        self.wib_header_size = self.wib_header_type.itemsize

        """ Not sure what's written in it """
        self.cb_header_type = np.dtype([
            ('word1','<u4'),  #4
            ('word2','<u4'),  #8
            ('word3','<u4'),  #12
            ('word4','<u4'),  #16
        ])
        
        self.cb_header_size = self.cb_header_type.itemsize


        
    def open_file(self):
        print('hello')

    def read_run_header(self):
        print('hello')


    def read_evt_header(self, i):
        print('hello ', i)



    def read_evt(self, i):
        print('hello')


    def close_file(self):

        print('good bye!')
