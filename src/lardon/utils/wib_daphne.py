import lardon.channel_mapping as cmap
import lardon.config as cf
import lardon.data_containers as dc

import numpy as  np
import numba as nb
import re
import time


import h5py as hp


def get_unix_timestamp_wib_2(t):
    return t*16e-9

def get_unix_time_wib_2(t):
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


def get_daq_header(daq):
    if('felix' in daq):
        daq_header_type = np.dtype([
            ("w0", 'u4'),     
            ("timestamp_1", '<u4'),
            ("timestamp_2", '<u4'),
        ])
    else:
        daq_header_type = np.dtype([
            ("w0", '<u4'),      
            ("w1", '<u4'),
        ])        
    return daq_header_type

def decode_daq_header(x, daq):
    if('felix' in daq):    
        w0  = x['w0']
        ts1 = x['timestamp_1']
        ts2 = x['timestamp_2']

        return {
            "version":  get_bits(w0, 0, 6),
            "det_id":   get_bits(w0, 6, 6),
            "crate_id": get_bits(w0, 12, 10),
            "slot_id":  get_bits(w0, 22, 4),
            "link_id":  get_bits(w0, 26, 6),
            "timestamp": np.uint64(ts1) | (np.uint64(ts2) << 32),
        }

    else:
        w0 = x['w0']
        w1 = x['w1']

        return  {
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




def get_daphne_header(daq):
    if('felix' in daq):
        daphne_header_type = np.dtype([
            ("w0", '<u4'),
            ("w1", '<u4'),
        ])
    else:
        daphne_header_type = np.dtype([
            ("w0", '<u4'),
            ("w1", '<u4'),
            ("w2", '<u4'),
            ("w3", '<u4'),
            ("w4", '<u4'),
            ("w5", '<u4'),
            ("w6", '<u4'),
            
        ])        
    return daphne_header_type


def decode_daphne_header(x, daq):
    if(daq == 'daphne_felix_stream'):
        w0, w1 = x['w0'], x['w1']
        return {
            "channel_0":    get_bits(w0, 0, 6),
            "channel_1":    get_bits(w0, 6, 6),
            "channel_2":    get_bits(w0, 12, 6),
            "channel_3":    get_bits(w0, 18, 6),
            "tbd1"     :    get_bits(w0, 24, 8),
            "tbd2"     :    w1
        }
    elif(daq == 'daphne_felix_trigger'):
        w0, w1 = x['w0'], x['w1']

        return {
            "channel":               get_bits(w0, 0, 6),
            "algorithm_id":          get_bits(w0, 6, 4),
            "trigger_sample_value":  get_bits(w0, 16, 16),
            "threshold":             get_bits(w1, 0, 16),
            "baseline":              get_bits(w1, 16, 16),
        }
    elif(daq == 'daphne_eth_trigger'):
        w0 = x['w0']

        return {
            "trig_sample": get_bits(w0, 0, 14),
            "threshold":   get_bits(w0, 16, 14),
            "baseline":    get_bits(w0, 32, 14),
            "version":     get_bits(w0, 52, 4),
            "channel":     get_bits(w0, 56, 8),
            "w1": x['w1'],
            "w2": x['w2'],
            "w3": x['w3'],
            "w4": x['w4'],
            "w5": x['w5'],
            "w6": x['w6']
        }

def get_daphne_trailer(daq):
    if('felix' in daq):
        daphne_trailer_type = np.dtype([
            ("w0", '<u4'),     
        ])
        return daphne_trailer_type
    else:
        print(daq, ' has no trailer')
""" nb: no trailer decoder has the trailer is not filled with any informations """


def get_daphne_peak(daq):
    if('felix' in daq):
        daphne_peak_type = np.dtype([
            ("w0", '<u4'),
            ("w1", '<u4'),
            ("w2", '<u4'),     
            ("w3", '<u4'),
            ("w4", '<u4'),
            ("w5", '<u4'),     
            ("w6", '<u4'),
            ("w7", '<u4'),
            ("w8", '<u4'),     
            ("w9", '<u4'),
            ("w10", '<u4'),
            ("w11", '<u4'),
            ("w12", '<u4'),     
        ])
        return daphne_peak_type
    else:
        print(daq, ' has no peak structure')

def decode_daphne_peak(x):
    words = np.frombuffer(x,'<u4')
    peaks = []

    # 5 peaks: each has an odd word and an even word
    for i in range(5):
        w_odd  = words[2*i]     # word1,3,5,7,9
        w_even = words[2*i + 1] # word2,4,6,8,10

        peak = {
            "num_subpeaks":         get_bits(w_odd, 0, 4),
            "adc_integral":         get_bits(w_odd, 8, 23),
            "found":                get_bits(w_odd, 31, 1),

            "adc_max":              get_bits(w_even, 0, 14),
            "sample_max":           get_bits(w_even, 14, 9),
            "samples_over_baseline": get_bits(w_even, 23, 9),
        }
        peaks.append(peak)

    # time-start words
    w11 = words[10]
    w12 = words[11]

    samples_start = [
        get_bits(w11, 22, 10),
        get_bits(w11, 12, 10),
        get_bits(w11, 2,  10),
        get_bits(w12, 22, 10),
        get_bits(w12, 12, 10),
    ]

    # trailer = words[12]

    return peaks, samples_start



@nb.jit(nopython = True)
def read_eth_evt_uint14_nb(data):

    tt = np.frombuffer(data, dtype=np.uint64)
    assert np.mod(tt.shape[0],14)==0

    out=np.zeros(tt.shape[0]//14*64,dtype=np.uint16)


    n_words_per_frame = 64*14//64
    n_words_per_fragment = 64*n_words_per_frame

    for i in nb.prange(tt.shape[0]//n_words_per_fragment): 
        frag_off = i*n_words_per_fragment

        for j in range(64): #loop on frame
            adc_words = tt[frag_off+j*n_words_per_frame:frag_off+(j+1)*n_words_per_frame]
            for k in range(64): #loop on channels
                word = int(14*k/64)
                first_bit = int((14*k)%64)
                nbits_first_word = min(14, 64-first_bit)
                adc = adc_words[word] >> first_bit
                if(nbits_first_word < 14):
                    adc +=  (adc_words[word+1] << nbits_first_word)
                final = adc & 0x3FFF

                out[(i*64*64) + (64*j) + k] = final

    return out


@nb.njit(parallel=True)
def read_felix_adc_trigger(adc_words):
    BITS_PER_ADC = 14
    BITS_PER_WORD = 32        # sizeof(word_t) * 8
    NUM_ADCS = 1024
    NUM_ADC_WORDS = NUM_ADCS * BITS_PER_ADC // BITS_PER_WORD  # 448

    n_frames = adc_words.shape[0]
    out = np.empty((n_frames, NUM_ADCS), dtype=np.uint16)


    for f in nb.prange(n_frames):
        words = adc_words[f]

        for i in range(NUM_ADCS):
            bitpos = BITS_PER_ADC * i
            word_index = bitpos // BITS_PER_WORD
            first_bit_position = bitpos % BITS_PER_WORD

            if first_bit_position <= BITS_PER_WORD - BITS_PER_ADC:
                out[f, i] = (words[word_index] >> first_bit_position) & 0x3FFF
            else:
                low = words[word_index] >> first_bit_position
                high = words[word_index + 1] << (BITS_PER_WORD - first_bit_position)
                out[f, i] = (low | high) & 0x3FFF

    return out


@nb.njit(parallel=True)
def assemble_waveforms(final, channels, t_start, waveforms):
    n = waveforms.shape[0]
    L = waveforms.shape[1]

    for i in nb.prange(n):
        ch = channels[i]
        if(ch < 0):
            continue
        t0 = t_start[i]
        for j in range(L):
            final[ch, t0 + j] = waveforms[i, j]



class daphne:
    def __init__(self, f_in):
        self.f_in = f_in
        
        self.fragment_header_type = get_fragment_header()  
        self.fragment_header_size = self.fragment_header_type.itemsize
               
        self.n_bits_per_adc = 14
        self.n_bytes_u32 = 4
        self.n_bytes_u64 = 8
        self.num_adc = 1024
        self.n_samples_per_frame   = 64
        self.n_channels_per_stream = 4


        if(len(dc.chmap_pds) == 0):
            return
        
        max_chan = max(ch.chan for ch in dc.chmap_pds)
        """ to remap the channel number written in the stream to an internal daq channel"""
        self.chan2daq = np.full(max_chan+1, -1, dtype=np.int32)
        for ch in dc.chmap_pds:
            self.chan2daq[ch.chan] = ch.daqch

        """ list of channel number kept for the analysis """
        """ e.g. lardon is not interested in PDVD PMTs facing the cryostat """
        self.ok_chans = [x.chan for x in dc.chmap_pds]

        
    def read_pds_evt(self, evt):

        for daq, link_name, offset, nstream in zip(cf.pds_daq, cf.pds_daq_link_name, cf.pds_daq_link_offset, cf.pds_daq_link_nstream):

            if(daq == 'daphne_felix_stream'):
                self.read_pds_felix_stream(evt, link_name, offset, nstream)            
            elif(daq == 'daphne_felix_trigger'):
                self.read_pds_felix_trigger(evt, link_name, offset, nstream)
            
    def read_pds_felix_stream(self, evt, link_name, offset, nstream):        
        cf.n_pds_stream_sample = -1
        
        daq_header_type = get_daq_header('daphne_felix_stream')
        daq_header_size = daq_header_type.itemsize
        daphne_header_type = get_daphne_header('daphne_felix_stream')
        daphne_header_size = daphne_header_type.itemsize
        daphne_trailer_type = get_daphne_trailer('daphne_felix_stream')
        daphne_trailer_size = daphne_trailer_type.itemsize

        adc_size = self.n_channels_per_stream * self.n_samples_per_frame * self.n_bits_per_adc // 8 #=448

        frame_size = daq_header_size + daphne_header_size + daphne_trailer_size + adc_size #=472
        
        names = ["0x"+format(istream+offset, '08x') for istream in range(nstream)]
        
        
        pds_tstart = []
        for istream in range(nstream):
            name = names[istream]
            try:
                path = f"/{evt}/RawData/Detector_Readout_{name}_{link_name}"
                """ don't read the fragment header """
                stream_data = self.f_in[path][:]#self.fragment_header_size:]
                
            except KeyError:
                print("no ", path, " data ")
                continue

            fragment = np.frombuffer(stream_data[:self.fragment_header_size], dtype=self.fragment_header_type)
            
            print('---> FULL STREAM FRAGMENT TIMESTAMP ', fragment['timestamp'])
            stream_data = stream_data[self.fragment_header_size:]
            
            if(len(stream_data) == 0):
                """ the event is empty, just skip it """
                continue

            daq_header = np.frombuffer(stream_data[:daq_header_size], dtype=daq_header_type)
            
            daphne_header = np.frombuffer(stream_data[daq_header_size:daq_header_size+daphne_header_size], dtype=daphne_header_type)
            
            daq_infos = decode_daq_header(daq_header, 'daphne_felix_stream')
            daphne_infos = decode_daphne_header(daphne_header, 'daphne_felix_stream')

            pds_tstart.append(daq_infos['timestamp'][0])


            debug = False
            if(debug):
                print('Reading ', name)
                print(f"DAQ version {daq_infos['version']}, Detector : {daq_infos['version']}, Stream in crate {daq_infos['crate_id']} Slot {daq_infos['slot_id']} Link {daq_infos['link_id']} -- TimeStamp : {get_unix_time_wib_2(daq_infos['timestamp'][0])}")
                print(f"Channels : {daphne_infos['channel_0']}, {daphne_infos['channel_1']}, {daphne_infos['channel_2']}, {daphne_infos['channel_3']}")

            
            """ remove the headers and trailers """
            stream_data = stream_data.reshape(-1,frame_size)[:,daq_header_size+daphne_header_size:frame_size-daphne_trailer_size].flatten()


            cf.n_pds_stream_sample = int(len(stream_data)/adc_size)*64

            
            if(cf.n_pds_stream_sample != dc.data_stream_pds.shape[-1]):
                dc.data_stream_pds = np.zeros((cf.n_pds_stream_channels, cf.n_pds_stream_sample), dtype=np.float32)
                
            out = read_eth_evt_uint14_nb(stream_data)
            out = np.reshape(out, (-1,self.n_channels_per_stream)).T
            out = out.astype(np.float32)


            for ichan in range(self.n_channels_per_stream):
                daq = self.n_channels_per_stream*istream + ichan                
                glob = dc.chmap_daq_pds[daq].globch
                if(glob < 0):
                    continue
                else:

                    if('M' in dc.chmap_pds[glob].det):                        
                        dc.data_stream_pds[glob] = -1*out[ichan]
                    else:
                        dc.data_stream_pds[glob] = out[ichan]
                        
        if(cf.n_pds_stream_sample > 0):
            
            ts = get_unix_timestamp_wib_2(min(pds_tstart))
            print('PDS STREAM timestamp ', get_unix_time_wib_2(min(pds_tstart)))
            print('TS = ', ts)
            dc.evt_list[-1].set_pds_stream_timestamp(ts)
            
        
    def read_pds_felix_trigger(self, evt, link_name, offset, nstream):

        
        daq_header_type = get_daq_header('daphne_felix_trigger')
        daq_header_size = daq_header_type.itemsize
        daphne_header_type = get_daphne_header('daphne_felix_trigger')
        daphne_header_size = daphne_header_type.itemsize
        daphne_peak_type = get_daphne_peak('daphne_felix_trigger')
        daphne_peak_size = daphne_peak_type.itemsize

        num_adc_size = self.num_adc * self.n_bits_per_adc // 8

        
        frame_dtype = np.dtype([
            ("daq", daq_header_type),
            ("hdr", daphne_header_type),
            ("adc", '<u4', int(num_adc_size/4)),
            ("peaks", daphne_peak_type),   # PeakDescriptorData = 13 words
        ])


        """ pds chmap """
        
        """ extract triggered time, slot*100+channel, waveforms """
        """ do not care about the peak found by DAQ - some are missing """
        
        
        frame_size = daq_header_size + daphne_header_size + num_adc_size + daphne_peak_size
        
        print("number of streams in data ", nstream, " offset ", offset)


        
        names = ["0x"+format(istream*10+offset, '08x') for istream in range(nstream)]

        
        times_chunk, daq_chans_chunk, adcs_chunk = [], [], []
        for istream in range(nstream):
            name = names[istream]
            try:
                path = f"/{evt}/RawData/Detector_Readout_{name}_{link_name}"

                """ don't read the fragment header """
                stream_data = self.f_in[path][:]#self.fragment_header_size:].reshape(-1)
                
            except KeyError:
                print("no ", path, " data ")
                continue

            
            fragment = np.frombuffer(stream_data[:self.fragment_header_size], dtype=self.fragment_header_type)
            print('---> TRIGGER FRAGMENT TIMESTAMP ', fragment['timestamp'])

            stream_data = stream_data[self.fragment_header_size:].reshape(-1)

            
            if(len(stream_data) == 0):
                """ the event is empty, just skip it """
                continue


            
            nframes = int(len(stream_data)/frame_size)

            
            frames = stream_data.view(frame_dtype)
            """ extract all timestamps, channel nb and adc at once """
            times = (
                frames["daq"]["timestamp_1"].astype(np.uint64)
                | (frames["daq"]["timestamp_2"].astype(np.uint64) << 32)
            )

            times_chunk.append(times)

            slots = (frames["daq"]["w0"] >>22) & 0xF
            channels = frames["hdr"]["w0"] & 0x3F

            daq_trigger_offset = cf.pds_daqch_trig_start
            new_chan = slots*100 + channels
            """ mask to ignore peaks from PDS we are not interested in """
            mask_ch = np.in1d(new_chan, self.ok_chans, assume_unique=False)

            """ convert channel nb into daq number """
            daq_chans = np.asarray([self.chan2daq[x]-daq_trigger_offset if m==True else -1 for x,m in zip(new_chan, mask_ch)], dtype=np.int32)
            daq_chans_chunk.append(daq_chans)
            
            adc_words = frames["adc"]
            adcs = read_felix_adc_trigger(adc_words)

            adcs_chunk.append(adcs)
            
        
        times = np.concatenate(times_chunk)
        daq_chans = np.concatenate(daq_chans_chunk)
        adcs = np.concatenate(adcs_chunk, axis=0)

        min_t, max_t = min(times), max(times)
        cf.n_pds_trig_sample = int(max_t - min_t)+1024

        ts = get_unix_timestamp_wib_2(min_t)
        dc.evt_list[-1].set_pds_trig_timestamp(ts)

        print('PDS TRIGGER timestamp ', get_unix_time_wib_2(min_t))
        print('TS = ', ts)


        times = times-min_t
        if(cf.n_pds_trig_sample != dc.data_trig_pds.shape[-1]):
            dc.data_trig_pds = np.zeros((cf.n_pds_trig_channels, cf.n_pds_trig_sample), dtype=np.float32)
            dc.data_trig_pds[:] = np.nan

        assemble_waveforms(dc.data_trig_pds, daq_chans, times, adcs)
        
