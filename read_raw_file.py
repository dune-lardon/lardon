import config as cf
import data_containers as dc
import numpy as np
import numba as nb
import os
import glob as glob
import sys
import tables as tab
from abc import ABC, abstractmethod



@nb.jit
def read_evt_uint12_nb(data):
    """ reads the top electronics event """
    l = len(data)
    assert np.mod(l,3)==0

    out=np.empty(l//3*2,dtype=np.uint16)

    for i in nb.prange(l//3):
        fst_uint8=np.uint16(data[i*3])
        mid_uint8=np.uint16(data[i*3+1])
        lst_uint8=np.uint16(data[i*3+2])

        out[i*2]   = (fst_uint8 << 4) + (mid_uint8 >> 4)
        out[i*2+1] = ((mid_uint8 % 16) << 8) + lst_uint8

    return out

def decode_8_to_5_3(x):
    read5 =  x & 0x1f
    read3 = (x & 0xe0) >> 5
    return read5, read3

def get_unix_time(t):
    return t*20/1e9


@nb.jit
def read_8evt_uint12_nb(data):
    """ reads the bottom electronics event """

    l = len(data)
    assert np.mod(l,12)==0
            

    out = np.empty(l//3*2,dtype=np.uint16)
    
    for i in nb.prange(l//12):
        b0  = np.uint16(data[i*12])&0xff
        b1  = np.uint16(data[i*12+1])&0xff
        b2  = np.uint16(data[i*12+2])&0xff
        b3  = np.uint16(data[i*12+3])&0xff
        b4  = np.uint16(data[i*12+4])&0xff
        b5  = np.uint16(data[i*12+5])&0xff
        b6  = np.uint16(data[i*12+6])&0xff
        b7  = np.uint16(data[i*12+7])&0xff
        b8  = np.uint16(data[i*12+8])&0xff
        b9  = np.uint16(data[i*12+9])&0xff
        b10 = np.uint16(data[i*12+10])&0xff
        b11 = np.uint16(data[i*12+11])&0xff

        out[i*8+0] = b0 | ((b2&0xf) << 8)
        out[i*8+1] = ((b2&0xf0)>>4) | (b4 << 4)
        out[i*8+2] = b6 | ((b8&0xf) << 8)
        out[i*8+3] = ((b8&0xf0)>>4) | (b10 << 4)
        out[i*8+4] = b1 | ((b3&0xf) << 8)
        out[i*8+5] = ((b3&0xf0)>>4) | (b5 << 4)
        out[i*8+6] = b7 | ((b9&0xf) << 8)
        out[i*8+7] = ((b9&0xf0)>>4) | (b11 << 4)
    return out



class decoder(ABC):
     
    @abstractmethod
    def open_file(self):
        pass

    @abstractmethod
    def read_run_header(self):
        pass

    @abstractmethod
    def read_evt_header(self, ievt):
        pass

    @abstractmethod
    def read_evt(self, ievt):
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
        fl = glob.glob(path+"_*")
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


    def read_evt_header(self, ievt):
        self.f_in.seek(self.event_pos[ievt],0)

        head = np.frombuffer(self.f_in.read(self.header_size), dtype=self.header_type)


        if( not((head['k0'][0] & 0xFF)==self.evskey and (head['k1'][0] & 0xFF)==self.evskey)):
            print(" problem in the event header ")
        good_evt = (head['evt_flag'][0] & 0x3F) == self.evdcard0

        if(not good_evt):
            print(" problem, the event has ", head['evt_flag'][0] & 0x3F, ' cards disconnected instead of ', self.evdcard0)

        self.lro = head['lro'][0]
        self.cro = head['cro'][0]
        
        dc.evt_list.append( dc.event("top", head['run_num'][0], self.sub, ievt, head['trig_num'][0], head['time_s'][0], head['time_ns'][0]) )

    def read_evt(self, ievt):
        if(self.lro < 0 or self.cro < 0):
            print(' please read the event header first! ')
            sys.exit()


        idx = self.event_pos[ievt] + self.header_size
        self.f_in.seek(idx,0)
        out = read_evt_uint12_nb( self.f_in.read(self.cro) )


        if(len(out)/cf.n_sample != cf.n_tot_channels):
            print(' The event is incomplete ... ')
            sys.exit()

        out = out.astype(np.float32)
        dc.data_daq = np.reshape(out, (cf.n_tot_channels, cf.n_sample))

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
        self.n_chan_per_link = 256
        self.n_chan_per_wib = 128
        self.n_chan_per_block  = 64
        self.n_block_per_wib = 4 #128/64

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
            ('ts1','<u4'),       #12 #the time written sounds weird (maybe different bit ordering)
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

        self.wib_frame_size = self.wib_header_size + 4*(self.cb_header_size + int(64*3/2))



        
    def open_file(self):
        r = int(self.run)
        long_run = f'{r:08d}'
        run_path = ""
        for i in range(0,8,2):
            run_path += long_run[i:i+2]+"/"
        path = cf.data_path + "/" + run_path
        
        s = int(self.sub)
        long_sub = f'{s:04d}'
        sub_name = 'run'+str(f'{r:06d}')+'_'+long_sub
        
        fl = glob.glob(path+"*"+sub_name+"*hdf5")

        if(len(fl) != 1):
            print('none or more than one file matches ... : ', fl)
            sys.exit()

        f = fl[0]
        print('Reconstructing ', f)
        self.f_in = tab.open_file(f,"r")
        

    def read_run_header(self):

        self.events_list = []

        for group in self.f_in.walk_groups():
            if(group._v_depth != 1):
                continue
            self.events_list.append(group._v_name)

        self.events_list.sort()
        nb_evt = len(self.events_list)

        return nb_evt


    def read_evt_header(self, ievt):
        trig_rec = self.f_in.get_node("/"+self.events_list[ievt], name='TriggerRecordHeader',classname='Array').read()

        header_magic = 0x33334444
        header_version = 0x00000002


        head = np.frombuffer(trig_rec[:self.trigger_header_size], dtype=self.trigger_header_type)
        if(head['header_marker'][0] != header_magic or head['header_version'][0] != header_version):
            print(' there is a problem with the header magic / version ')
            
        self.nlinks = head['n_component'][0]
        self.links = []        
        for i in range(self.nlinks):
            s = self.trigger_header_size + i*self.component_header_size
            t = s + self.component_header_size
            comp = np.frombuffer(trig_rec[s:t], dtype=self.component_header_type)
            self.links.append(comp['geo_ID'][0])

        t_unix = get_unix_time(head['timestamp'][0])

        t_s = int(t_unix)
        t_ns = (t_unix - t_s) * 1e9
        dc.evt_list.append( dc.event("bot", head['run_nb'][0], self.sub, ievt, head['trig_num'][0], t_s, t_ns) )


    def read_evt(self, ievt):
        
        for ilink in range(self.nlinks):
            name = f'{ilink:02d}'

            link_data = self.f_in.get_node("/"+self.events_list[ievt]+"/TPC/CRP004", name='Link'+name,classname='Array').read()
            frag_head = np.frombuffer(link_data[:self.fragment_header_size], dtype = self.fragment_header_type)

            n_frames = int((len(link_data)-self.fragment_header_size)/self.wib_frame_size)
            if(n_frames != cf.n_sample):
                print(" the link has ", n_frames, " frames ... but ", cf.n_sample, ' are expected !')
     
            wib_head = np.frombuffer(link_data[self.fragment_header_size:self.fragment_header_size+self.wib_header_size], dtype = self.wib_header_type)
            
            _, fiber = decode_8_to_5_3(wib_head['ver_fib'][0])
            crate, slot = decode_8_to_5_3(wib_head['crate_slot'][0])
            #print(ilink, ' ', name, ' fiber ', fiber, ' crate ', crate, ' slot ', slot)


            # remove the fragment header
            link_data = link_data[self.fragment_header_size:]
    
            #remove the wib headers (size16, once per wib frame of size 464)
            link_data = link_data.reshape(-1,self.wib_frame_size)[:,self.wib_header_size:].flatten()
    
            #remove the CB headers (size16, once per cold data block of size 112 (4 per wib frame))
            link_data = link_data.reshape(-1,112)[:,self.cb_header_size:].flatten()    

            """ decode data """
            out = read_8evt_uint12_nb(link_data)


            ''' array structure is all channel at time=0 then at time=1 etc '''
            ''' change it to all times of channel 1, then channel 2 etc '''            
            out = np.reshape(out, (-1,self.n_chan_per_link)).T

            
            ''' some groups of channels have to swapped to read them in the correct order '''
            ''' should be changed to non-hardcoded values '''
            
            ''' I'm not sure I can explain how I did that '''
            ''' LZ : I'll make it non-hardcoded '''
            out = np.reshape(out, (16,4,32768)) #=8192*4
            out[:,[1,2],:] = out[:,[2,1],:]
            out = np.reshape(out, (self.n_chan_per_link,cf.n_sample))

            out = out.astype(np.float32)


            dc.data_daq[ilink*self.n_chan_per_link:(ilink+1)*self.n_chan_per_link] = out
        self.nlinks = 0
        self.links = []



    def close_file(self):
        self.f_in.close()
        print('good bye!')
