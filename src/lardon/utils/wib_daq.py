import lardon.channel_mapping as cmap
import lardon.config as cf
import lardon.data_containers as dc
import lardon.utils.enum_type as et

import lardon.utils.wib_daphne as daphne
import numpy as  np
import numba as nb
import re

#import tables as tab
import h5py as hp

def extract_server_number(filename):
    match = re.search(r'df-s(\d+)', filename)
    if match:
        return int(match.group(1))
    else:
        return -1


def get_unix_timestamp_wib_1(t):
    return t*20e-9

def get_unix_time_wib_1(t):
    ts = t*20 # in units of nanoseconds
    ts_s = int(ts/1e9)
    ts_ns = ts-ts_s*1e9
    return ts_s, ts_ns

def get_unix_timestamp_wib_2(t):
    return t*16e-9

def get_unix_time_wib_2(t):
    #return t*16/1e9
    ts = t*16 # in units of nanoseconds
    ts_s = int(ts/1e9)
    ts_ns = ts-ts_s*1e9
    return ts_s, ts_ns


def get_daq_eth_infos(x):

    x = int(x)
    infos = {}
    infos['version']   = (x      ) & 0x3f
    infos['det_id']    = (x >>  6) & 0x3f
    infos['crate']     = (x >> 12) & 0x3ff
    infos['slot']      = (x >> 22) & 0xf
    infos['stream']    = (x >> 26) & 0xff
    infos['reserved']  = (x >> 34) & 0x3f
    infos['seq_id']    = (x >> 40) & 0xfff
    infos['block_len'] = (x >> 52) & 0xfff
    

    return infos



def get_wib_2_eth_infos(x):

    x = int(x)
    infos = {}
    infos['channel'] = (x >> 56) & 0xff
    infos['version'] = (x >> 52) & 0xf
    infos['context'] = (x >> 44) & 0xff
    infos['ready']   = (x >> 43) & 0x1
    infos['calib']   = (x >> 42) & 0x1
    infos['pulser']  = (x >> 41) & 0x1
    infos['femb_s']  = (x >> 39) & 0x3
    infos['wib_s']   = (x >> 37) & 0x1
    infos['lol']     = (x >> 36) & 0x1
    infos['link_v']  = (x >> 35) & 0x3
    infos['crc_err'] = (x >> 33) & 0x3
    infos['cd']      = (x >> 32) & 0x1
    infos['ts1']     = (x >> 16) & 0x7fff
    infos['ts2']     = (x      ) & 0x7fff

    return infos
    

def get_wib_2_infos(x):

    version = x & 0x3F
    det_id  = (x & 0xFC0)>>6
    crate   = (x & 0x3FF000) >> 12
    slot    = (x & 0x1C00000) >> 22
    link    = (x & 0xFC000000) >> 26

    return crate, slot, link

def decode_8_to_5_3(x):
    read5 =  x & 0x1f
    read3 = (x & 0xe0) >> 5
    return read5, read3




@nb.jit(nopython = True)
def read_uint12_bit_nb(data):

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


@nb.jit(nopython = True)
def read_uint14_bit_nb(data, n_chan):

    tt = np.frombuffer(data, dtype=np.uint32)
    assert np.mod(tt.shape[0],14)==0

    out=np.empty(tt.shape[0]//14*32,dtype=np.uint16)

    for i in nb.prange(tt.shape[0]//112): 
        adc_words = tt[i*112:(i+1)*112]
        for k in range(n_chan): 
            word = int(14*k/32)
            first_bit = int((14*k)%32)
            nbits_first_word = min(14, 32-first_bit)
            adc = adc_words[word] >> first_bit
            if(nbits_first_word < 14):
                adc +=  (adc_words[word+1] << nbits_first_word)
            final = adc & 0x3FFF
            out[i*n_chan+k] = final
    return out


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


def get_trigger_header(daq):

    if(daq == 'wib_1'):
        trigger_header_type = np.dtype([
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
        return trigger_header_type

    elif( daq == 'wib_2' or daq == "wib_2_eth"):        
        trigger_header_type = np.dtype([
            ('header_marker','<u4'),    #4
            ('header_version','<u4'),   #8
            ('trig_num', '<u8'),        #16   
            ('timestamp', '<u8'),       #24
            ('n_component', '<u8'),     #32
            ('run_nb', '<u4'),          #36
            ('error_bit', '<u4'),       #40
            ('trigger_type', '<u8'),    #48
            ('sequence_nb', '<u2'),     #50
            ('max_sequence_nb', '<u2'), #52
            ('unused', '<u4'),          #56
            ('source_version','<u4'),   #60
            ('source_elemID','<u4'),    #64
        ])
        return trigger_header_type        
    else:
        print('ERROR unknown daq', daq)
        return -1

def get_trigger_type(x):
    """
    Convert a number that is a single-bit mask (power of two)
    into the bit index.
    """
    if x == 0 or (x & (x - 1)) != 0:
        raise ValueError("Trigger Type is not a power of two (single bit).")
    return x.bit_length() - 1


def get_component_header(daq):

    if(daq == 'wib_1'):
        component_header_type = np.dtype([
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
        return component_header_type

    elif(daq == 'wib_2' or daq == "wib_2_eth"):
        component_header_type = np.dtype([
            ('version','<u4'),        #4
            ('unused','<u4'),         #8
            #('source_version','<u4'), #12
            #('source_elemID','<u4'),  #16
            ('source', '<u8'),        #16
            ('window_begin','<u8'),   #24
            ('window_end','<u8')      #32                
        ])
        return component_header_type

    else:
        print('ERROR unknown daq', daq)
        return -1



def get_fragment_header(daq):

    if(daq == "wib_1"):        
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
            ('geo_version','<u4'),  #68
            ('geo_sys','<u2'),      #70
            ('geo_region','<u2'),   #72
            ('geo_ID','<u4'),       #76    
            ('geo_unused','<u4')    #80
        ]) 
        return fragment_header_type

    elif(daq == "wib_2" or daq == "wib_2_eth"):
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

    else:
        print('ERROR unknown daq', daq)
        return -1


def get_colddata_header_type(daq):
    if(daq == 'wib_1'):
        cb_header_type = np.dtype([
            ('word1','<u4'),  #4
            ('word2','<u4'),  #8
            ('word3','<u4'),  #12
            ('word4','<u4'),  #16
        ])
        return cb_header_type
    else : 
        return None

def get_wib_trailer_type(daq):
    if(daq == 'wib_2'):
        wib_trailer_type = np.dtype([
            ('word1','<u4')  #4
        ])
        return wib_trailer_type
        
    else:
        return None

        

def get_wib_header(daq):

    if(daq == "wib_1"):
        wib_header_type = np.dtype([
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
        return wib_header_type
        
    elif(daq == "wib_2"):
        wib_header_type = np.dtype([
            ('wibinfos', '<u4'), #4
            ('ts',    '<u8'), #12
            ('word2', '<u4'), #16
            ('word3', '<u4')  #20
        ])
        return wib_header_type

    elif(daq == "wib_2_eth"):
        wib_header_type = np.dtype([
            ('daqethinfos', '<u8'),  #8
            ('ts',    '<u8'),    #16
            ('wibinfos', '<u8'), #24
            ('reserved', '<u8')  #32
        ])
        return wib_header_type
        

    else:
        print('ERROR unknown daq', daq)
        return -1


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

    
def get_daphne_header(daq):
    if(daq != 'wib_2_eth'):
        print('ERROR, not pds streamed')
        return -1
    else:
        daphne_header_type = np.dtype([
            ('infos',    '<u4'), #4
            ('timestamp','<u8'), #12
            ('channels', '<u4'), #16   
            ('tbd',      '<u4')  #20
        ])
        return daphne_header_type

def get_daphne_trailer(daq):
    if(daq != 'wib_2_eth'):
        print('ERROR, not pds streamed')
        return -1
    else:
        daphne_trailer_type = np.dtype([
            ('tbd',      '<u4')  #4
        ])
        return daphne_trailer_type


#def get_daphne_eth_frame_header(daq):
    
    
class wib:
    def __init__(self, f, daq, det, run):

        self.filename = f
        
        try:
            self.f_in = hp.File(f, "r")
        except IOError:
            print('File ', f, ' does not exist...')
            exit()
        
        self.daq = daq
        self.det = det
        self.run = run

        self.server = extract_server_number(f)
        
        self.n_chan_per_link  = 256
        self.n_chan_per_wib   = 128
        self.n_chan_per_block = 64
        #self.n_block_per_wib  = 4 #128/64
        
        '''
        self.n_chan_per_frame = 64
        self.n_samp_per_frame = 64
        self.n_chan_per_wib = 128
        self.n_chan_per_block  = 64
        self.n_block_per_wib = 4 #128/64
        '''


        self.trigger_header_type = get_trigger_header(self.daq)
        self.trigger_header_size = self.trigger_header_type.itemsize
        
        self.component_header_type = get_component_header(self.daq)
        self.component_header_size = self.component_header_type.itemsize
    
        self.fragment_header_type = get_fragment_header(self.daq)  
        self.fragment_header_size = self.fragment_header_type.itemsize
        
        self.wib_header_type = get_wib_header(self.daq)
        self.wib_header_size = self.wib_header_type.itemsize


        if(self.daq == 'wib_1'):
            self.cb_header_size = get_colddata_header_type(self.daq).itemsize            
            self.wib_frame_size = self.wib_header_size + 4*(self.cb_header_size + int(64*3/2))



        elif(self.daq == 'wib_2_eth'):

            self.pds_decode = daphne.daphne(self.f_in)

            self.daphne_header_type = get_daphne_header(self.daq)
            self.daphne_header_size = self.daphne_header_type.itemsize        

            self.daphne_trailer_type = get_daphne_trailer(self.daq)
            self.daphne_trailer_size = self.daphne_trailer_type.itemsize        

            self.n_samp_per_frame = 64
            self.n_chan_per_link  = 64

            self.wib_trailer_size = 0            
            self.wib_frame_size = self.wib_header_size + self.wib_trailer_size + int(self.n_chan_per_link*self.n_samp_per_frame*14/8)




        elif(self.daq == 'wib_2'):

            self.wib_trailer_size = get_wib_trailer_type(daq).itemsize
            self.wib_frame_size = self.wib_header_size + self.wib_trailer_size +int(self.n_chan_per_link*14/8) 

        else:
            print('the DAQ ', self.daq, ' decoder is not yet implemented')
            return

    def read_run_header(self):
        self.events_list = []

        for name, group in self.f_in.items():  # Top-level groups
            if isinstance(group, hp.Group):
                self.events_list.append(name)

        self.events_list.sort()

        nb_evt = len(self.events_list)

        return nb_evt



    def read_evt_header(self, sub, ievt, flow):
        if(self.daq == "wib_1"):
            return self.read_evt_header_wib_1(sub, ievt, flow)
        else:
            return self.read_evt_header_wib_2(sub, ievt, flow)


    def read_evt_header_wib_1(self, sub, ievt, flow):
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

        t_s, t_ns = get_unix_time_wib_1(head['timestamp'][0])
        ts = get_unix_timestamp_wib_1(head['timestamp'][0])

        dc.evt_list.append( dc.event(self.det, "bot", head['run_nb'][0], sub, ievt, head['trig_num'][0], ts, t_s, t_ns, 0) )


    def read_evt_header_wib_2(self, sub, ievt, flow):
        fl = int(flow)
        name =  f'{fl:08d}'

        if(self.server >= 0):            
            name = f'{self.server*10+fl:08x}'
        else:
            name = f'{cf.daq_TRBuilder_number+fl:08x}'
        
        path = f"/{self.events_list[ievt]}/RawData/TR_Builder_0x{name}_TriggerRecordHeader"
        trig_rec = self.f_in[path][:]
        

        header_magic =  0x33334444
        header_version_1 = 0x00000003
        header_version_2 = 0x00000004


        head = np.frombuffer(trig_rec[:self.trigger_header_size], dtype=self.trigger_header_type)

        if(head['header_marker'][0] != header_magic or (head['header_version'][0] != header_version_1 and head['header_version'][0] != header_version_2)):
            print(' there is a problem with the header magic / version ')
            print("marker : ", head['header_marker'][0], " vs ", header_magic)
            print("version : ", head['header_version'][0], " vs ", header_version)


        trigger = get_trigger_type(int(head['trigger_type'][0]))
        
        self.nlinks = head['n_component'][0]


        """
        self.links = []        
        
        for i in range(self.nlinks):
            s = self.trigger_header_size + i*self.component_header_size
            t = s + self.component_header_size
            comp = np.frombuffer(trig_rec[s:t], dtype=self.component_header_type)
            #self.links.append(comp['source_elemID'][0])
            print('Link ', i)
            print('version' , comp['version'], 'source ', comp['source'], ' wbeg ', comp['window_begin'], ' wend', comp['window_end'])
            slot = (comp['source'] >> 32) & 0xffff
            print(slot)
        """


        """ nb of links have to be hard-coded as the number of components is wrong """
        self.nlinks = cf.daq_nlinks


        t_s, t_ns = get_unix_time_wib_2(head['timestamp'][0])
        #print('---> TRIGGER RECORD TIMESTAMP ', head['timestamp'])
        ts = get_unix_timestamp_wib_2(head['timestamp'][0])
        #print('<->', ts)
        dc.evt_list.append( dc.event(self.det, "bot", head['run_nb'][0], sub, ievt, head['trig_num'][0], ts, t_s, t_ns, trigger) )
        




    def read_evt(self, ievt):
        
        if(self.daq == "wib_1"):
            return self.read_evt_wib_1(ievt)
        elif(self.daq == "wib_2"):
            return self.read_evt_wib_2(ievt)
        elif(self.daq == "wib_2_eth"):
            return self.read_evt_wib_2_eth(ievt)
        

    def read_evt_wib_1(self, ievt):

        for ilink in range(self.nlinks):
            name = f'{ilink:02d}'

            """ to be improved !"""
            try :
                link_data = self.f_in.get_node("/"+self.events_list[ievt]+"/TPC/CRP004", name='Link'+name,classname='Array').read()
            except tab.NoSuchNodeError:

                try :
                    link_data = self.f_in.get_node("/"+self.events_list[ievt]+"/TPC/APA004", name='Link'+name,classname='Array').read()
                except tab.NoSuchNodeError:
                    print('no link number ', ilink, 'with name APA004 nor CRP004')
                    continue

            frag_head = np.frombuffer(link_data[:self.fragment_header_size], dtype = self.fragment_header_type)

            n_frames = int((len(link_data)-self.fragment_header_size)/self.wib_frame_size)

            if(n_frames != cf.n_sample):
                print(" the link ", name, " has ", n_frames, " frames ... but ", cf.n_sample, ' are expected !')
                    
                cf.n_sample = n_frames
                if(n_frames == 0):
                    return

                """ reshape the dc arrays accordingly """
                dc.data = np.zeros((cf.n_module, cf.n_view, max(cf.view_nchan), cf.n_sample[cf.imod]), dtype=np.float32)
                dc.data_daq = np.zeros((cf.module_nchan[cf.imod], cf.n_sample[cf.imod]), dtype=np.float32) #view, vchan
                #dc.alive_chan = np.ones((cf.n_tot_channels, cf.n_sample), dtype=bool)
                dc.alive_chan = np.ones(cf.module_nchan[cf.imod], dtype=bool)

                cmap.set_unused_channels()
                dc.mask_daq  = np.ones((cf.module_nchan[cf.imod], cf.n_sample[cf.imod]), dtype=bool)
                    

            wib_head = np.frombuffer(link_data[self.fragment_header_size:self.fragment_header_size+self.wib_header_size], dtype = self.wib_header_type)
            
            _, fiber = decode_8_to_5_3(wib_head['ver_fib'][0])
            crate, slot = decode_8_to_5_3(wib_head['crate_slot'][0])



            # remove the fragment header
            link_data = link_data[self.fragment_header_size:]
    
            #remove the wib headers (size16, once per wib frame of size 464)
            link_data = link_data.reshape(-1,self.wib_frame_size)[:,self.wib_header_size:].flatten()
    
            #remove the CB headers (size16, once per cold data block of size 112 (4 per wib frame))
            link_data = link_data.reshape(-1,112)[:,self.cb_header_size:].flatten()    

            """ decode data """
            out = read_uint12_bit_nb(link_data)


            ''' array structure is all channel at time=0 then at time=1 etc '''
            ''' change it to all times of channel 1, then channel 2 etc '''            
            out = np.reshape(out, (-1,self.n_chan_per_link)).T

            
            ''' some groups of channels have to swapped to read them in the correct order '''
            ''' should be changed to non-hardcoded values '''
            
            ''' NB : I'm not sure I can explain how I did that '''
            out = np.reshape(out, (16,4,4*cf.n_sample)) #=8192*4
            out[:,[1,2],:] = out[:,[2,1],:]
            out = np.reshape(out, (self.n_chan_per_link,cf.n_sample))
            
            out = out.astype(np.float32)


            dc.data_daq[ilink*self.n_chan_per_link:(ilink+1)*self.n_chan_per_link] = out

        self.nlinks = 0
        self.links = []


    def read_evt_wib_2(self, ievt):

        if(self.det == 'cbbot'):
            names = ["0x%08d"%ilink for ilink in range(self.nlinks)]
            names[10] = "0x0000000a"
            names[11] = "0x0000000b"


        for ilink in range(self.nlinks):   
            name = names[ilink]
            try :
                path = f"/{self.events_list[ievt]}/RawData/Detector_Readout_{name}_{cf.daq_link_name[cf.imod]}"
                link_data = self.f_in[path][:]
                #link_data = self.f_in.get_node("/"+self.events_list[ievt]+"/RawData", name='Detector_Readout_'+name+'_WIB',classname='Array').read()                
            except KeyError:
                continue


            frag_head = np.frombuffer(link_data[:self.fragment_header_size], dtype = self.fragment_header_type)

            n_frames = int((len(link_data)-self.fragment_header_size)/self.wib_frame_size)

            if(n_frames != cf.n_sample):                    
                cf.n_sample = n_frames
                if(n_frames == 0):
                    return
                    

                """ reshape the dc arrays accordingly """
                dc.data = np.zeros((cf.n_module, cf.n_view, max(cf.view_nchan), cf.n_sample[cf.imod]), dtype=np.float32)
                dc.data_daq = np.zeros((cf.module_nchan[cf.imod], cf.n_sample[cf.imod]), dtype=np.float32) #view, vchan
                dc.alive_chan = np.ones(cf.module_nchan[cf.imod], dtype=bool)
                cmap.set_unused_channels()
                dc.mask_daq  = np.ones((cf.module_nchan[cf.imod], cf.n_sample[cf.imod]), dtype=bool)
                    

            wib_head = np.frombuffer(link_data[self.fragment_header_size:self.fragment_header_size+self.wib_header_size], dtype = self.wib_header_type)

            
            crate, link, slot = get_wib_2_infos(wib_head['wibinfos'][0])

            # remove the fragment header
            link_data = link_data[self.fragment_header_size:]


            #remove the wib header and trailer (size20+4, once per wib frame of size 464)

            link_data = link_data.reshape(-1,self.wib_frame_size)[:,self.wib_header_size:self.wib_frame_size-self.wib_trailer_size].flatten()

            """ decode data """            
            out = read_uint14_bit_nb(link_data, self.n_chan_per_link)


            ''' array structure is all channel at time=0 then at time=1 etc '''
            ''' change it to all times of channel 1, then channel 2 etc '''

            out = np.reshape(out, (-1,self.n_chan_per_link)).T
            out = out.astype(np.float32)

            dc.data_daq[ilink*self.n_chan_per_link:(ilink+1)*self.n_chan_per_link] = out

        self.nlinks = 0
        self.links = []




    def read_evt_wib_2_eth(self, ievt):
              
        names = ["0x"+format(ilink+cf.daq_links_offset[cf.imod], '08x') for ilink in range(cf.daq_nlinks[cf.imod])]

        
        cf.n_sample[cf.imod] = -1
        tstart_link = [np.nan for x in range(self.nlinks[cf.imod])]
        tstop_link  = [0. for x in range(self.nlinks[cf.imod])]


        delta_start, delta_stop = 0, 0
        delta_samp = 0

        # first pass where the 1st and last timestamps of each wib frames are extracted
        for ilink in range(self.nlinks[cf.imod]):   
            name = names[ilink]
            
            try :
                path = f"/{self.events_list[ievt]}/RawData/Detector_Readout_{name}_{cf.daq_link_name[cf.imod]}"
                link_data = self.f_in[path]                
            except KeyError:
                continue

            data_len = link_data.size - self.fragment_header_size
            N = data_len // self.wib_frame_size #nb of frames
            if(N==0):
                """ there is no wib frame ... stop here """
                continue

            offset = self.fragment_header_size
            ts_offset = 8
            
            first_ts_pos = offset + ts_offset
            last_frame_pos = offset + (N-1)*self.wib_frame_size
            last_ts_pos = last_frame_pos + ts_offset

            first_ts = np.frombuffer(link_data[first_ts_pos:first_ts_pos+8], dtype='<u8')[0]/32
            last_ts = np.frombuffer(link_data[last_ts_pos:last_ts_pos+8], dtype='<u8')[0]/32

            tstart_link[ilink] = int(first_ts)
            tstop_link[ilink]  = int(last_ts)

        if(len(tstart_link)==0):
            return


        tstart = min(tstart_link) if len(tstart_link)>0 else 0
        if(np.isnan(tstart)):
           return
        tstop  = max(tstop_link) if len(tstop_link)>0 else 0
        
        n_tot_frames = (tstop-tstart)        
        n_tot_frames *= cf.sampling[cf.imod]/1.953125
        n_tot_frames += 64
        
        n_tot_frames = int(((n_tot_frames+1) // 64) * 64)
        
        cf.n_sample[cf.imod] = n_tot_frames
        dc.data_daq = np.zeros((cf.module_nchan[cf.imod], cf.n_sample[cf.imod]), dtype=np.float32) #view, vchan
        event_begin = tstart
        event_end   = tstop
        

        for ilink in range(self.nlinks[cf.imod]):   
            name = names[ilink]
            try :
                path = f"/{self.events_list[ievt]}/RawData/Detector_Readout_{name}_{cf.daq_link_name[cf.imod]}"
                link_data = self.f_in[path][:]                
            except KeyError:
                continue

            link_data = link_data[self.fragment_header_size:]
            
            if(len(link_data) == 0):
                continue

            wib_ts = link_data.reshape(-1, self.wib_frame_size)[:,8:16].flatten()
            wib_ts = np.frombuffer(wib_ts, dtype='<u8')/32
            wib_ts = wib_ts.astype(np.int64)
            #
            
            n_link_frames = int((len(link_data)-self.fragment_header_size)/self.wib_frame_size)*self.n_samp_per_frame
            n_event_frames = n_link_frames#int(wib_ts[-1]-wib_ts[0])+64


            if(n_event_frames != n_link_frames):
                if(n_link_frames > n_event_frames):
                    print(f'there is a DAQ problem ... {n_link_frames} in the wib, but the timestamps says {n_event_frames}!')
                else:
                    print(f'... the wib have skipped {n_event_frames-n_link_frames} frames')
            
            n_frames = n_link_frames#n_event_frames
            link_start = tstart_link[ilink]
            link_stop = tstop_link[ilink]
            delta_start = int(link_start-event_begin)
            delta_stop = int(event_end-link_stop)

            
            t_bef = delta_start if delta_start > 0 else 0
            t_end = delta_stop if delta_stop > 0 else 0

            t_bef *= cf.sampling[cf.imod]/1.953125
            t_end *= cf.sampling[cf.imod]/1.953125
            l_bef = int(t_bef)
            l_end = int(t_end)


            
            """
            wib_head = np.frombuffer(link_data[:self.wib_header_size], dtype = self.wib_header_type)
            print('\n',ilink, name,'\n')
            #print('DAQ and WIB INFOS') 
            # to get the detector, crate, slot, (etc) informations
            # detdataformats/include/detdataformats/DAQEthHeader.hpp
            daq_infos = get_daq_eth_infos(wib_head['daqethinfos'][0])
            print(daq_infos)
            wib = daq_infos['slot'] +1
            stream = daq_infos['stream']
            loc_stream = stream & 0x3
            link = (stream >> 6) &1
            crate = daq_infos['crate']
            #print('--> wib', wib, ' stream ', stream, ' loc ', loc_stream, 'link', link, 'crate', crate)

                        
            #To get the WIB infos like pulser/calibration/?
            #fddetdataformats/include/fddetdataformats/WIBEthFrame.hpp
            # cf https://edms.cern.ch/document/2088713/9 'deimos'
            wib_infos = get_wib_2_eth_infos(wib_head['wibinfos'][0])
            print(wib_infos)
            print('---')
            """

            #remove the wib header  (size32, once per wib frame of size 7200)
            link_data = link_data.reshape(-1,self.wib_frame_size)[:,self.wib_header_size:self.wib_frame_size].flatten()

            """ decode data """
            out = read_eth_evt_uint14_nb(link_data)#

            ''' array structure is all channel at time=0 then at time=1 etc '''
            ''' change it to all times of channel 1, then channel 2 etc '''

            out = np.reshape(out, (-1,self.n_chan_per_link)).T
            out = out.astype(np.float32)


            

            if(l_bef > 0 or l_end > 0):
                out = np.pad(out, ((0,0), (l_bef, l_end)), 'constant', constant_values=np.nan)


            """ one wib frame is 64 ticks """
            wib_delta_ts = np.diff(wib_ts)
            
            if(np.all(wib_delta_ts==64) == False):
                
                wib_delta_ts -= 64

                idx = np.where(wib_delta_ts>0)
                skipped = wib_delta_ts[idx]
                skipped = skipped.astype(np.int32)

                idx = idx[0]*64

                #print("skipping ", idx)
                for i, p in enumerate(idx):
                    out = np.insert(out, [p-1 for x in range(skipped[i])], np.nan, axis=1)



            if(out.shape[1] != dc.data_daq.shape[1]):
                
                continue
            dc.data_daq[ilink*self.n_chan_per_link:(ilink+1)*self.n_chan_per_link] = out
        
        #self.nlinks = 0
        self.links = []

        

        if(cf.n_sample[cf.imod] > 0):

            dc.data = np.zeros(( cf.n_view, max(cf.view_nchan), cf.n_sample[cf.imod]), dtype=np.float32)
            new_shape = (cf.module_nchan[cf.imod], cf.n_sample[cf.imod])
            dc.alive_chan = np.ones(cf.module_nchan[cf.imod], dtype=bool)
            cmap.set_unused_channels()
            dc.mask_daq  = np.ones(new_shape, dtype=bool)

                    
            charge_tstart = min(tstart_link)*32
            ts = get_unix_timestamp_wib_2(charge_tstart)
            #t_s, t_ns = get_unix_time_wib_2(charge_tstart)

            #print('min of start link ', min(tstart_link), ' --> ', min(tstart_link)*32)
            #print('<-> timestamp ', ts)
            
            dc.evt_list[-1].set_charge_timestamp(cf.imod, ts)

    def read_pds_evt(self, ievt):
        self.pds_decode.read_pds_evt(self.events_list[ievt])
        return 

        
        self.n_samples_per_frame   = 64
        self.n_channels_per_stream = 4
        self.daphne_data_size = int(self.n_samples_per_frame * self.n_channels_per_stream * 14/8)
        self.daphne_frame_size = self.daphne_header_size + self.daphne_data_size + self.daphne_trailer_size


        cf.n_pds_sample = -1
        
        """ Hard coded at the moment for CB BOT data """

        self.n_stream = cf.pds_daq_link_nstream
        offset = cf.pds_daq_link_offset
        print("number of stream ", self.n_stream, " offset ", offset)
        """
        self.n_stream = 4
        offset = 1
        if(self.det == 'pdhd'):
            self.n_stream = 10

        if(self.det == 'pdvd'):
            self.n_stream = 4
            offset = 720
        """           
        names = ["0x"+format(istream+offset, '08x') for istream in range(self.n_stream)]
        print(names)
        
        pds_tstart = []
        for istream in range(self.n_stream):
            name = names[istream]
            try:
                path = f"/{self.events_list[ievt]}/RawData/Detector_Readout_{name}_{cf.pds_daq_link_name}"
                stream_data = self.f_in[path][:]
                
            except KeyError:
                print(path, ":: nope")
                continue
            
            """ don't read the fragment header """
            daphne = np.frombuffer(stream_data[self.fragment_header_size:self.fragment_header_size+self.daphne_header_size], dtype=self.daphne_header_type)

            if(len(daphne) == 0):
                """ the event is empty, just skip it """
                continue
            
            channels = decode_daphne_channels(daphne['channels'])
            print('stream', name, ' channels: ', channels)
            print(daphne)
            pds_tstart.append(daphne['timestamp'][0])
            
            """ remove the headers and trailers """
            stream_data = stream_data[self.fragment_header_size:]
            stream_data = stream_data.reshape(-1,self.daphne_frame_size)[:,self.daphne_header_size:self.daphne_frame_size-self.daphne_trailer_size].flatten()


            cf.n_pds_sample = int(len(stream_data)/self.daphne_data_size)*64

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

        
    def close_file(self):
        self.f_in.close()

