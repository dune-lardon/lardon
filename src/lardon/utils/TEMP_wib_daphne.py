import lardon.channel_mapping as cmap
import lardon.config as cf
import lardon.data_containers as dc

import numpy as  np
import numba as nb
import re
import time

#import tables as tab
import h5py as hp

def get_unix_time_wib_2(t):
    #return t*16/1e9
    ts = t*16 # in units of nanoseconds
    ts_s = int(ts/1e9)
    ts_ns = ts-ts_s*1e9
    return ts_s, ts_ns


def get_bits(val, offset, width):
    return (val >> offset) & ((1 << width) - 1)

def get_fragment_header():
    fragment_header_type = np.dtype([
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
        ('source_version','<u4'), #68
        ('source_elemID','<u4')  #72
    ]) 
        return fragment_header_type


########
def decode_daphne_daq(x):
    x = int(x)
    infos = {}

    infos['version'] = x & 0x3f
    infos['det_id']  = (x >> 6 ) & 0x3f
    infos['crate']   = (x >> 12) & 0x3ff
    infos['slot']    = (x >> 22) & 0xf
    infos['link']    = (x >> 26) & 0x3f
    return infos

def decode_daphne_channels(x):
    chan1 = x & 0x3f
    chan2 = (x>>6) & 0x3f
    chan3 = (x>>12) & 0x3f
    chan4 = (x>>18) & 0x3f    
    return [chan1[0], chan2[0], chan3[0], chan4[0]]

    
def get_daphne_header():
    daphne_header_type = np.dtype([
        ('infos',    '<u4'), #4
        ('timestamp','<u8'), #12
        ('channels', '<u4'), #16   
        ('tbd',      '<u4')  #20
    ])
    return daphne_header_type

def get_daphne_trailer():
    daphne_trailer_type = np.dtype([
        ('tbd',      '<u4')  #4
    ])
    return daphne_trailer_type
#######


def get_daq_header(words):
    #in case of felix
    w0 = words[0]
    ts1 = words[1]
    ts2 = words[2]

    header = {
        "version":  get_bits(w0, 0, 6),
        "det_id":   get_bits(w0, 6, 6),
        "crate_id": get_bits(w0, 12, 10),
        "slot_id":  get_bits(w0, 22, 4),
        "link_id":  get_bits(w0, 26, 6),
        "timestamp": np.uint64(ts1) | (np.uint64(ts2) << 32),
    }

    return header



def get_daq_eth_header(words):
    #in case of ETH
    w0 = words[0]
    w1 = words[1]   # timestamp

    header = {
        "version":      get_bits(w0, 0, 6),
        "det_id":       get_bits(w0, 6, 6),
        "crate_id":     get_bits(w0, 12, 10),
        "slot_id":      get_bits(w0, 22, 4),
        "stream_id":    get_bits(w0, 26, 8),
        "reserved":     get_bits(w0, 34, 6),
        "seq_id":       get_bits(w0, 40, 12),
        "block_length": get_bits(w0, 52, 12),
        "timestamp":    w1
    }
    return header


def get_daphne_stream_header(words):
    w0, w1 = words[0], words[1]
    return {
        "channel_0":    get_bits(w0, 0, 6),
        "channel_1":    get_bits(w0, 6, 6),
        "channel_2":    get_bits(w0, 12, 6),
        "channel_3":    get_bits(w0, 18, 6),
        "tbd1"     :    get_bits(w0, 24, 8),
        "tbd2"     :    w1
        }

def get_daphne_felix_header(words):
    w0, w1 = words[0], words[1]

    return {
        "channel":               get_bits(w0, 0, 6),
        "algorithm_id":          get_bits(w0, 6, 4),
        "trigger_sample_value":  get_bits(w0, 16, 16),
        "threshold":             get_bits(w1, 0, 16),
        "baseline":              get_bits(w1, 16, 16),
    }


def get_daphne_eth_header(words):
    w0 = words[0]

    header = {
        "trig_sample": get_bits(w0, 0, 14),
        "threshold":   get_bits(w0, 16, 14),
        "baseline":    get_bits(w0, 32, 14),
        "version":     get_bits(w0, 52, 4),
        "channel":     get_bits(w0, 56, 8),
        "w1": words[1],
        "w2": words[2],
        "w3": words[3],
        "w4": words[4],
        "w5": words[5],
        "w6": words[6],
    }
    return header




def get_daq_header(daq):

    if('felix' in daq):
        daq_header_type = np.dtype([
            ("word0", 'u4'),        # version, det_id, crate_id, slot_id, link_id
            ("timestamp_1", 'u4'),
            ("timestamp_2", 'u4'),
        ])
    else:
        daq_header_type = np.dtype([
            ("w0", 'u4'),        # trig_samp, thresh, baseline, v, channel
            ("w1", 'u4'),
            ("w2", 'u4'),
            ("w3", 'u4'),
            ("w4", 'u4'),
            ("w5", 'u4'),
            ("w6", 'u4'),
        ])        
    return daq_header_type

def get_daphner_header(daq):

    if('felix' in daq):
        daphne_header_type = np.dtype([
            ("word0", 'u4'),        # # channel, algorithm_id, flags
            ("word1", 'u4'),
        ])
    else:
        daphne_header_type = np.dtype([
            ("w0", 'u4'),        # trig_samp, thresh, baseline, v, channel
            ("w1", 'u4'),
            ("w2", 'u4'),
            ("w3", 'u4'),
            ("w4", 'u4'),
            ("w5", 'u4'),
            ("w6", 'u4'),
        ])        
    return daphne_header_type
    
    


class daphne:
    def __init__(self, f_in, events_list, daq_pds):
        self.f_in = f_in
        self.events_list = events_list
        self.daq_pds = list(set(daq_pds))
            
        self.fragment_header_type = get_fragment_header(self.daq)  
        self.fragment_header_size = self.fragment_header_type.itemsize

        self.n_bits_per_adc = 14
        self.n_bytes_u32 = 4
        self.n_bytes_u64 = 8

        self.daq_header_size     = 3*self.n_bytes_u32
        self.daq_eth_header_size = 2*self.n_bytes_u64

        self.daphne_felix_header_size  = 2*self.n_bytes_u32
        self.daphne_felix_trailer_size = 1*self.n_bytes_u32
        self.daphne_eth_header_size    = 7*self.n_bytes_u64

        self.daphne_peak_size = 13*self.n_bytes_u32
        
        if('daphne_felix_stream' in self.daq_pds):
            self.n_samples_per_frame   = 64
            self.n_channels_per_stream = 4
            self.num_adc_words_felix_stream = self.n_channels_per_stream * self.n_samples_per_frame * self.n_bits_per_adc // self.n_bytes_u32 #=112

            #self.daphne_felix_stream_size = self.daq_header_size + self.daphne_header_size + self.daphne_trailer_size + self.num_adc_words*self.bytes_per_word #=472


        if('daphne_felix_trigger' in self.daq_pds):
            self.num_adc = 1024
            self.num_adc_words_felix_trigger = self.num_adc * self.n_bits_per_adc // self.n_bytes_u32 #=448
            #self.peak_size = 13*self.bytes_per_word
            #self.daphne_felix_trigger_size = self.daq_header_size + self.daphne_header_size + self.peak_size + self.num_adc_words*self.bytes_per_word #=1864

            
        if('daphne_eth_trigger' in self.daq_pds):
            #self.bits_per_adc = 14
            #self.bytes_per_word  = 8 #data written as uint64
            self.num_adc = 1024
            self.num_adc_words_eth_trigger = self.num_adc * self.bits_per_adc // self.bits_per_word #=224
            #self.daphne_eth_trigger_size = self.daq_header_size + self.daphne_header_size + self.num_adc_words*self.bytes_per_word#=1864


    def read_pds_evt(self, ievt):
        
        if('daphne_felix_stream' in self.daq_pds):
            read_pds_felix_stream(self, ievt)
            
        if('daphne_felix_trigger' in self.daq_pds):
            read_pds_felix_trigger(self, ievt)
            
    def read_pds_felix_stream(self, ievt):
        
        cf.n_pds_sample = -1

        frame_size = self.daq_header_size + self.daphne_felix_header_size + self.daphne_felix_trailer_size + self.num_adc_words_felix_stream*self.n_bytes_u32 #=472
            
            
        self.n_stream = cf.pds_daq_link_nstream
        offset = cf.pds_daq_link_offset
        
        print("number of stream ", self.n_stream, " offset ", offset)
        
        names = ["0x"+format(istream+offset, '08x') for istream in range(self.n_stream)]
        print(names)
        
        pds_tstart = []
        for istream in range(self.n_stream):
            name = names[istream]
            try:
                path = f"/{self.events_list[ievt]}/RawData/Detector_Readout_{name}_{cf.pds_daq_link_name}"
                """ don't read the fragment header """
                stream_data = self.f_in[path][self.fragment_header_size:]
                
            except KeyError:
                print("no ", path, " data ")
                continue
                
            
            if(len(stream_data) == 0):
                """ the event is empty, just skip it """
                continue

            daq_header = np.frombuffer(stream_data[:self.daq_header_size:], dtype='<u4')
            daphne_header = np.frombuffer(stream_data[self.daq_header_size:self.daq_header_size+self.daphne_felix_header_size], dtype='<u4')

        
            pds_tstart.append(daq_header['timestamp'][0])
            
            if(debug):
                print('Reading ', name)
                print(f"DAQ version {daq_header['version']}, Detector : {daq_header['version']}, Stream in crate {daq_header['crate_id']} Slot {daq_header['slot_id']} Link {daq_header['link_id']} -- TimeStamp : {get_unix_time_wib_2({daq_header['timestamp']})}")
                print(f"Channels : {daphne_header['channel_0']}, {daphne_header['channel_1']}, {daphne_header['channel_2']}, {daphne_header['channel_3']}")

            
            """ remove the headers and trailers """
            #stream_data = stream_data[self.fragment_header_size:]
            stream_data = stream_data.reshape(-1,frame_size)[:,self.daq_header_size+self.daphne_felix_header_size:self.daphne_frame_size-self.daphne_felix_trailer_size].flatten()


            cf.n_pds_sample = int(len(stream_data)/self.num_adc_words_felix_stream)*64

            if(cf.n_pds_sample != dc.data_pds.shape[-1]):
                dc.data_pds = np.zeros((cf.n_pds_channels, cf.n_pds_sample), dtype=np.float32)
                
            out = read_eth_evt_uint14_nb(stream_data)
            #print('out shape=== ', out.shape)
            out = np.reshape(out, (-1,self.n_channels_per_stream)).T
            #print('reshaped ', out.shape)
            out = out.astype(np.float32)


            for ichan in range(self.n_channels_per_stream):
                daq = self.n_channels_per_stream*istream + ichan                
                glob = dc.chmap_daq_pds[daq].globch
                #print(ichan, " : daq ",daq, 'glob ',glob)
                if(glob < 0):
                    continue
                else:
                    #print(ichan, " :: ", dc.chmap_daq_pds[daq].chan, channels[ichan], '-->', daq, glob)
                    #assert dc.chmap_daq_pds[daq].chan == channels[ichan]
                    if('M' in dc.chmap_pds[glob].det):                        
                        dc.data_pds[glob] = -1*out[ichan]
                    else:
                        dc.data_pds[glob] = out[ichan]
                        
        if(cf.n_pds_sample > 0):
            t_s, t_ns = get_unix_time_wib_2(min(pds_tstart))
            dc.evt_list[-1].set_pds_timestamp(t_s, t_ns)

        
    def read_pds_felix_trigger(self, ievt)

        frame_size = self.daq_header_size + self.daphne_felix_header_size + self.num_adc_words_felix_trigger*self.n_bytes_u32 #=472

        
            
        self.n_stream = cf.pds_daq_link_nstream_trigger
        offset = cf.pds_daq_link_offset_trigger
        
        print("number of stream ", self.n_stream, " offset ", offset)
        
        names = ["0x"+format(istream*10+offset, '08x') for istream in range(self.n_stream)]
        print(names)
        
        pds_tstart = []
        for istream in range(self.n_stream):
            name = names[istream]
            try:
                path = f"/{self.events_list[ievt]}/RawData/Detector_Readout_{name}_{cf.pds_daq_link_name_trigger}"
                """ don't read the fragment header """
                stream_data = self.f_in[path][self.fragment_header_size:]
                
            except KeyError:
                print("no ", path, " data ")
                continue
                
            
            if(len(stream_data) == 0):
                """ the event is empty, just skip it """
                continue
    
            nframes = int(len(stream_data)/frame_size)
            
