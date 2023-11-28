import numpy as  np
import numba as nb

import channel_mapping as cmap
import config as cf
import data_containers as dc
import tables as tab

def get_unix_time_wib_1(t):
    return t*20/1e9

def get_unix_time_wib_2(t):
    return t*16/1e9
    

def get_wib_2_eth_infos(x):

    x = int(x)
    version = (x      ) & 0x3f
    det_id  = (x >> 6 ) & 0x3f
    crate   = (x >> 12) & 0x3ff
    slot    = (x >> 22) & 0xf
    strm    = (x >> 26) & 0xff
    rsvd    = (x >> 34) & 0x3f
    seqid   = (x >> 40) & 0xfff
    b_len   = (x >> 52) & 0xfff

    return crate, slot, strm

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
            ('trig_num', '<u8'),        #24    
            ('timestamp', '<u8'),       #24
            ('n_component', '<u8'),     #32
            ('run_nb', '<u4'),          #36
            ('error_bit', '<u4'),       #40
            ('trigger_type', '<u2'),    #42
            ('sequence_nb', '<u2'),     #44
            ('max_sequence_nb', '<u2'), #46
            ('unused', '<u2'),          #48
            ('source_version','<u4'),   #52
            ('source_elemID','<u4'),    #56
        ])
        return trigger_header_type
        
    else:
        print('ERROR unknown daq', daq)
        return -1



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
            ('source_version','<u4'), #12
            ('source_elemID','<u4'),  #16
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
            ('unknown', '<u8'),  #8
            ('ts',    '<u8'),    #16
            ('wibinfos', '<u8'), #24
            ('reserved', '<u8')  #32
        ])
        return wib_header_type
        

    else:
        print('ERROR unknown daq', daq)
        return -1




class wib:
    def __init__(self, f, daq, det):

        self.filename = f

        try:
            self.f_in = tab.open_file(f, 'r')
        except IOError:
            print('File ', f, ' does not exist...')
            exit()
        
        self.daq = daq
        self.det = det
            
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

        for group in self.f_in.walk_groups():
            if(group._v_depth != 1):
                continue
            self.events_list.append(group._v_name)

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

        t_unix = get_unix_time_wib_1(head['timestamp'][0])

        t_s = int(t_unix)
        t_ns = (t_unix - t_s) * 1e9
        dc.evt_list.append( dc.event(self.det, "bot", head['run_nb'][0], sub, ievt, head['trig_num'][0], t_s, t_ns) )


    def read_evt_header_wib_2(self, sub, ievt, flow):
        fl = int(flow)
        name =  f'{fl:08d}'

        trig_rec = self.f_in.get_node("/"+self.events_list[ievt], name='RawData/TR_Builder_0x'+name+'_TriggerRecordHeader',classname='Array').read()

        header_magic =  0x33334444
        header_version = 0x00000003


        head = np.frombuffer(trig_rec[:self.trigger_header_size], dtype=self.trigger_header_type)
        if(head['header_marker'][0] != header_magic or head['header_version'][0] != header_version):
            print(' there is a problem with the header magic / version ')
            print("marker : ", head['header_marker'][0], " vs ", header_magic)
            print("version : ", head['header_version'][0], " vs ", header_version)


        self.nlinks = head['n_component'][0]

        if(self.det == '50l'):
            self.nlinks = 2
        

        self.links = []        

        for i in range(self.nlinks):
            s = self.trigger_header_size + i*self.component_header_size
            t = s + self.component_header_size
            comp = np.frombuffer(trig_rec[s:t], dtype=self.component_header_type)
            self.links.append(comp['source_elemID'][0])

        t_unix = get_unix_time_wib_2(head['timestamp'][0]) ##lllll
        t_s = int(t_unix)
        t_ns = (t_unix - t_s) * 1e9

        dc.evt_list.append( dc.event(self.det, "bot", head['run_nb'][0], sub, ievt, head['trig_num'][0], t_s, t_ns) )





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
                dc.data = np.zeros((cf.n_module, cf.n_view, max(cf.view_nchan), cf.n_sample), dtype=np.float32)
                dc.data_daq = np.zeros((cf.n_tot_channels, cf.n_sample), dtype=np.float32) #view, vchan
                dc.alive_chan = np.ones((cf.n_tot_channels, cf.n_sample), dtype=bool)

                cmap.set_unused_channels()
                dc.mask_daq  = np.ones((cf.n_tot_channels, cf.n_sample), dtype=bool)
                    

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
                link_data = self.f_in.get_node("/"+self.events_list[ievt]+"/RawData", name='Detector_Readout_'+name+'_WIB',classname='Array').read()
            except tab.NoSuchNodeError:
                #print('no link number ', ilink, 'with name RawData/Detector_Readout_'+name+'_WIB') 
                continue


            frag_head = np.frombuffer(link_data[:self.fragment_header_size], dtype = self.fragment_header_type)

            n_frames = int((len(link_data)-self.fragment_header_size)/self.wib_frame_size)

            if(n_frames != cf.n_sample):                    
                cf.n_sample = n_frames
                if(n_frames == 0):
                    return
                    

                """ reshape the dc arrays accordingly """
                dc.data = np.zeros((cf.n_module, cf.n_view, max(cf.view_nchan), cf.n_sample), dtype=np.float32)
                dc.data_daq = np.zeros((cf.n_tot_channels, cf.n_sample), dtype=np.float32) #view, vchan
                dc.alive_chan = np.ones((cf.n_tot_channels, cf.n_sample), dtype=bool)
                cmap.set_unused_channels()
                dc.mask_daq  = np.ones((cf.n_tot_channels, cf.n_sample), dtype=bool)
                    

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

        if(self.det == '50l'):
           names = ["0x%08d"%(64+ilink) for ilink in range(self.nlinks)]
        
        cf.n_sample = -1
        tstart_link = []
        tstop_link  = []


        delta_start, delta_stop = 0, 0
        delta_samp = 0

        for ilink in range(self.nlinks):   
            name = names[ilink]

            try :
                link_data = self.f_in.get_node("/"+self.events_list[ievt]+"/RawData", name='Detector_Readout_'+name+'_WIBEth',classname='Array').read()
            except tab.NoSuchNodeError:
                #print('no link number ', ilink, 'with name RawData/Detector_Readout_'+name+'_WIBEth') 
                continue

            frag_head = np.frombuffer(link_data[:self.fragment_header_size], dtype = self.fragment_header_type)

            n_link_frames = int((len(link_data)-self.fragment_header_size)/self.wib_frame_size)*self.n_samp_per_frame

            

            # remove the fragment header
            link_data = link_data[self.fragment_header_size:]

            if(len(link_data) == 0):
                continue

            #extract the timestamps of all wib frames
            wib_ts = link_data.reshape(-1, self.wib_frame_size)[:,8:16].flatten()
            wib_ts = np.frombuffer(wib_ts, dtype='<u8')/32
            wib_ts = wib_ts.astype(np.int64)

            
            tstart_link.append(wib_ts[0])
            tstop_link.append(wib_ts[-1])

            n_event_frames = int(wib_ts[-1]-wib_ts[0])+64


            if(n_event_frames != n_link_frames):
                if(n_link_frames > n_event_frames):
                    print(f'there is a DAQ problem ... {n_link_frames} in the wib, but the timestamps says {n_event_frames}!')
                else:
                    print(f'... the wib have skipped {n_event_frames-n_link_frames} frames')
            
            n_frames = n_event_frames

            if(cf.n_sample < 0): #first pass
                if(n_event_frames == 0):
                    continue
                else:
                    cf.n_sample = n_event_frames
                    dc.data_daq = np.zeros((cf.n_tot_channels, cf.n_sample), dtype=np.float32) #view, vchan
                    event_begin = tstart_link[0]
                    event_end   = tstop_link[0]
            else:
                delta_start = int(event_begin-tstart_link[-1])
                delta_stop = int(tstop_link[-1]-event_end)

                t_bef = delta_start if delta_start > 0 else 0
                t_end = delta_stop if delta_stop > 0 else 0

                
                if(delta_start > 0): event_begin = tstart_link[-1]
                if(delta_stop  > 0): event_end   = tstop_link[-1]

                if(n_event_frames > cf.n_sample):
                    if(t_bef > 0 or t_end > 0):
                        dc.data_daq = np.pad(dc.data_daq, ((0,0), (t_bef, t_end)), 'constant', constant_values=np.nan)

                elif(n_event_frames < cf.n_sample):
                    if(n_event_frames == 0):
                        continue
                    else:
                        if(t_bef > 0 or t_end > 0):   
                            dc.data_daq = np.pad(dc.data_daq, ((0,0), (t_bef, t_end)), 'constant', constant_values=np.nan)

                        delta_samp = cf.n_sample-n_frames


                else:
                    if(t_bef > 0 or t_end > 0):   
                        dc.data_daq = np.pad(dc.data_daq, ((0,0), (t_bef, t_end)), 'constant', constant_values=np.nan)


                cf.n_sample = int(event_end-event_begin)+64

            """
            wib_head = np.frombuffer(link_data[self.fragment_header_size:self.fragment_header_size+self.wib_header_size], dtype = self.wib_header_type)

            crate, link, slot = get_wib_2_eth_infos(wib_head['wibinfos'][0])
            print('crate: ', crate, 'link ', link, ' slot ', slot)
            """


            #remove the wib header  (size32, once per wib frame of size 7200)
            link_data = link_data.reshape(-1,self.wib_frame_size)[:,self.wib_header_size:self.wib_frame_size].flatten()

            """ decode data """
            out = read_eth_evt_uint14_nb(link_data)#

            ''' array structure is all channel at time=0 then at time=1 etc '''
            ''' change it to all times of channel 1, then channel 2 etc '''

            out = np.reshape(out, (-1,self.n_chan_per_link)).T
            out = out.astype(np.float32)

            l_bef = -1*delta_start if delta_start < 0 else 0
            l_end = -1*delta_stop if delta_stop < 0 else 0
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


                for i, p in enumerate(idx):
                    out = np.insert(out, [p-1 for x in range(skipped[i])], np.nan, axis=1)

            dc.data_daq[ilink*self.n_chan_per_link:(ilink+1)*self.n_chan_per_link] = out

        self.nlinks = 0
        self.links = []

        if(cf.n_sample > 0):
            dc.data = np.zeros((cf.n_module, cf.n_view, max(cf.view_nchan), cf.n_sample), dtype=np.float32)
            dc.alive_chan = np.ones((cf.n_tot_channels, cf.n_sample), dtype=bool)
            cmap.set_unused_channels()
            dc.mask_daq  = np.ones((cf.n_tot_channels, cf.n_sample), dtype=bool)
        


    def close_file(self):
        self.f_in.close()
        #print('file closed!')
