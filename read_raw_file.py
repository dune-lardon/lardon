import config as cf
import data_containers as dc
import numpy as np
import numba as nb
import os
import glob as glob
#import sys
import tables as tab
from abc import ABC, abstractmethod
import importlib
import utils.bde_headers as head_bde
import utils.filenames as fname

import channel_mapping as cmap


@nb.jit(nopython = True)
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



@nb.jit(nopython = True)
def read_evt_uint12_nb_RD(data):
    """ reads the bot electronics event of 50l """
    l = len(data)

    assert np.mod(l,6)==0

    out=np.empty(l//3*4,dtype=np.uint16)


    for i in nb.prange(l//6):

        b0  = np.uint16(data[i*6+0])
        b1  = np.uint16(data[i*6+1])
        b2  = np.uint16(data[i*6+2])
        b3  = np.uint16(data[i*6+3])
        b4  = np.uint16(data[i*6+4])
        b5  = np.uint16(data[i*6+5])

        out[i*8+7] = (b0 & 0X0FFF)<<0 
        out[i*8+6] = ((b1 & 0X00FF)<<4) + ((b0 & 0XF000) >> 12)
        out[i*8+5] = ((b2& 0X000F)<<8) + ((b1& 0XFF00) >> 8 )
        out[i*8+4] = (b2& 0XFFF0)>>4
        out[i*8+3] = (b3& 0X0FFF)<<0
        out[i*8+2] = ((b4& 0X00FF)<<4) + ((b3& 0XF000) >> 12)
        out[i*8+1] = ((b5& 0X000F)<<8) + ((b4& 0XFF00) >> 8)
        out[i*8+0] = (b5& 0XFFF0)>>4


    return out




def decode_8_to_5_3(x):
    read5 =  x & 0x1f
    read3 = (x & 0xe0) >> 5
    return read5, read3

def get_unix_time_cb1(t):
    return t*20/1e9

def get_unix_time(t):
    return t*16/1e9


def get_wib2_infos(x):

    version = x & 0x3F
    det_id  = (x & 0xFC0)>>6
    crate   = (x & 0x3FF000) >> 12
    slot    = (x & 0x1C00000) >> 22
    link    = (x & 0xFC000000) >> 26

    return crate, slot, link


@nb.jit(nopython = True)
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



@nb.jit(nopython = True)
def read_evt_uint14_nb(data):

    tt = np.frombuffer(data, dtype=np.uint32)
    assert np.mod(tt.shape[0],14)==0

    out=np.empty(tt.shape[0]//14*32,dtype=np.uint16)

    for i in nb.prange(tt.shape[0]//112): 
        adc_words = tt[i*112:(i+1)*112]
        for k in range(256):
            word = int(14*k/32)
            first_bit = int((14*k)%32)
            nbits_first_word = min(14, 32-first_bit)
            adc = adc_words[word] >> first_bit
            if(nbits_first_word < 14):
                adc +=  (adc_words[word+1] << nbits_first_word)
            final = adc & 0x3FFF
            out[i*256+k] = final
    return out


class decoder(ABC):
     
    @abstractmethod
    def open_file(self):
        pass

    @abstractmethod
    def get_filename(self):
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
    def __init__(self, run, sub, filename=None, det='cb', dummy=""):
        self.run = run
        self.sub = sub
        self.filename = filename
        self.detector = det
        print(' -- reading a top drift electronics file')
        
        """ some TDE specific parameters """
        self.evskey = 0xFF
        self.endkey = 0xF0
        self.evdcard0 = 0x0 #number of cards disconnected (CB-60deg)
        if(det == 'cb1'):
            self.evdcard0 = 0x5 #number of cards disconnected in CB1

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


    def get_filename(self):
        path = f"{cf.data_path}/{self.run}/{self.run}_{self.sub}"
        fl = glob.glob(path+"_*")
        if(len(fl) != 1):
            print('none or more than one file matches ... : ', fl)
            exit()
        f = fl[0]
        return f


    def open_file(self):
        f = self.filename if self.filename else self.get_filename()
        print('Reconstructing ', f)

        f_type = f[f.rfind('.')+1:]
        print(' NB : file type is ', f_type)
        try:
            self.f_in = open(f,'rb')
        except IOError:
            print('File ', f, ' does not exist...')
            exit()


    def read_run_header(self):
        run_nb, nb_evt = np.fromfile(self.f_in, dtype='<u4', count=2)


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
            print("  the event has ", head['evt_flag'][0] & 0x3F, ' cards disconnected instead of ', self.evdcard0)

        self.lro = head['lro'][0]
        self.cro = head['cro'][0]


        dc.evt_list.append( dc.event(self.detector, "top", head['run_num'][0], self.sub, ievt, head['trig_num'][0], head['time_s'][0], head['time_ns'][0]) )

    def read_evt(self, ievt):
        if(self.detector == "cb1"):
            return self.read_evt_one_block(ievt)
        else : 
            return self.read_evt_two_blocks(ievt)


    def read_evt_one_block(self, ievt):
        if(self.lro < 0 or self.cro < 0):
            print(' please read the event header first! ')
            exit()


        idx = self.event_pos[ievt] + self.header_size
        self.f_in.seek(idx,0)
        out = read_evt_uint12_nb( self.f_in.read(self.cro) )


        if(len(out)/cf.n_sample != cf.n_tot_channels):
            print(' The event is incomplete ... ')
            print(len(out))
            exit()

        out = out.astype(np.float32)
        dc.data_daq = np.reshape(out, (cf.n_tot_channels, cf.n_sample))

        self.lro = -1
        self.cro = -1


    def read_evt_two_blocks(self, ievt):
        if(self.lro < 0 or self.cro < 0):
            print(' please read the event header first! ')
            exit()


        idx = self.event_pos[ievt] + self.header_size
        self.f_in.seek(idx,0)
        out = read_evt_uint12_nb( self.f_in.read(self.cro) )

        self.f_in.read(1) #The bruno byte :-)

        """ read second header"""
        head = np.frombuffer(self.f_in.read(self.header_size), dtype=self.header_type)
        self.cro = head['cro'][0]
        out = np.append(out,read_evt_uint12_nb( self.f_in.read(self.cro) ))



        if(len(out)/cf.n_sample != cf.n_tot_channels):
            print(' The event is incomplete ... ')
            exit()

        out = out.astype(np.float32)
        dc.data_daq = np.reshape(out, (cf.n_tot_channels, cf.n_sample))

        self.lro = -1
        self.cro = -1



    def close_file(self):
        self.f_in.close()
        print('file closed!')




class bot_decoder(decoder):
    def __init__(self, run, sub, filename=None, det='cb', flow_writer="0-0"):
        self.run = run
        self.sub = sub
        self.filename = filename
        self.detector = det
        self.flow = flow_writer[:flow_writer.find("-")]
        self.writer = flow_writer[flow_writer.find("-")+1:]

        print(' -- reading a bottom drift electronics file')


        """ bde specific parameters """
        self.n_chan_per_link = 256
        self.n_chan_per_wib = 128
        self.n_chan_per_block  = 64
        self.n_block_per_wib = 4 #128/64


        self.trigger_header_type = head_bde.get_trigger_header(det)
        self.trigger_header_size = self.trigger_header_type.itemsize

        self.component_header_type = head_bde.get_component_header(det)
        self.component_header_size = self.component_header_type.itemsize
        
        self.fragment_header_type = head_bde.get_fragment_header(det)     
        self.fragment_header_size = self.fragment_header_type.itemsize

        self.wib_header_type = head_bde.get_wib_header(det)     
        self.wib_header_size = self.wib_header_type.itemsize


        if(det == "cb1"):
            
            """ Not sure what's written in it """
            self.cb_header_type = np.dtype([
                ('word1','<u4'),  #4
                ('word2','<u4'),  #8
                ('word3','<u4'),  #12
                ('word4','<u4'),  #16
            ])
            
            self.cb_header_size = self.cb_header_type.itemsize            
            self.wib_frame_size = self.wib_header_size + 4*(self.cb_header_size + int(64*3/2))

        if(det == "cb"):
            """ Not sure what's written in it """
            self.wib_trailer_type = np.dtype([
                ('word1','<u4')  #4
            ])
            self.wib_trailer_size = self.wib_trailer_type.itemsize            
            """ data now encoded in 14-bit """
            self.wib_frame_size = self.wib_header_size + self.wib_trailer_size +int(self.n_chan_per_link*14/8) 


    def get_filename(self):
        
        run_path = fname.get_run_directory(self.run)
        path = cf.data_path + "/" + run_path


        r = int(self.run)
        s = int(self.sub)
        long_sub = f'{s:04d}'
        sub_name = 'run'+str(f'{r:06d}')+'_*'+long_sub+'_'

        

        if(self.detector == "cb"):
            app_name = "dataflow"+self.flow+"_datawriter_"+self.writer
        elif(self.detector == "cb1"):
            app_name = "dataflow"+self.flow
        else:
            app_name = ""



        fl = glob.glob(path+"*"+sub_name+"*"+app_name+"*hdf5")

        if(len(fl) != 1):
            print('none or more than one file matches ... : ', fl)
            exit()

        return fl[0]



    def open_file(self):
        f = self.filename if self.filename else self.get_filename()
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
        if(self.detector == "cb1"):
            return self.read_evt_header_cb1(ievt)
        elif(self.detector == "cb"):
            return self.read_evt_header_cb(ievt)


    def read_evt_header_cb1(self, ievt):
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

        t_unix = get_unix_time_cb1(head['timestamp'][0])

        t_s = int(t_unix)
        t_ns = (t_unix - t_s) * 1e9
        dc.evt_list.append( dc.event(self.detector,"bot", head['run_nb'][0], self.sub, ievt, head['trig_num'][0], t_s, t_ns) )


    def read_evt_header_cb(self, ievt):
        fl = int(self.flow)
        name =  f'{fl:08d}'

        trig_rec = self.f_in.get_node("/"+self.events_list[ievt], name='RawData/TR_Builder_0x'+name+'_TriggerRecordHeader',classname='Array').read()

        header_magic =  0x33334444
        header_version = 0x00000003


        head = np.frombuffer(trig_rec[:self.trigger_header_size], dtype=self.trigger_header_type)
        if(head['header_marker'][0] != header_magic or head['header_version'][0] != header_version):
            print(' there is a problem with the header magic / version ')
            print("marker : ", head['header_marker'][0], " vs ", header_magic)
            print("version : ", head['header_version'][0], " vs ", header_version)
        self.nlinks = 12#head['n_component'][0]
        self.links = []        
        for i in range(self.nlinks):
            s = self.trigger_header_size + i*self.component_header_size
            t = s + self.component_header_size
            comp = np.frombuffer(trig_rec[s:t], dtype=self.component_header_type)
            self.links.append(comp['source_elemID'][0])

        t_unix = get_unix_time(head['timestamp'][0])

        t_s = int(t_unix)
        t_ns = (t_unix - t_s) * 1e9
        dc.evt_list.append( dc.event(self.detector,"bot", head['run_nb'][0], self.sub, ievt, head['trig_num'][0], t_s, t_ns) )





    def read_evt(self, ievt):
        if(self.detector == "cb1"):
            return self.read_evt_cb1(ievt)
        elif(self.detector == "cb"):
            return self.read_evt_cb(ievt)


    def read_evt_cb1(self, ievt):

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
            out = read_8evt_uint12_nb(link_data)


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




    def read_evt_cb(self, ievt):

        for ilink in range(self.nlinks):
            name = "0x%08d"%ilink
            if(ilink == 10):
                name = "0x0000000a"
            elif(ilink == 11):
                name = "0x0000000b"
        

            """ to be improved !"""
            try :
                link_data = self.f_in.get_node("/"+self.events_list[ievt]+"/RawData", name='Detector_Readout_'+name+'_WIB',classname='Array').read()
            except tab.NoSuchNodeError:
                print('no link number ', ilink, 'with name RawData/Detector_Readout_'+name+'_WIB')
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
            
            crate, link, slot = get_wib2_infos(wib_head['word1'][0])


            # remove the fragment header
            link_data = link_data[self.fragment_header_size:]
    
            #remove the wib header and trailer (size20+4, once per wib frame of size 464)
            link_data = link_data.reshape(-1,self.wib_frame_size)[:,self.wib_header_size:-self.wib_trailer_size].flatten()
    
            """ decode data """
            out = read_evt_uint14_nb(link_data)


            ''' array structure is all channel at time=0 then at time=1 etc '''
            ''' change it to all times of channel 1, then channel 2 etc '''            
            out = np.reshape(out, (-1,self.n_chan_per_link)).T
            out = out.astype(np.float32)
            dc.data_daq[ilink*self.n_chan_per_link:(ilink+1)*self.n_chan_per_link] = out

        self.nlinks = 0
        self.links = []



    def close_file(self):
        self.f_in.close()
        print('good bye!')




class dp_decoder(decoder):
    def __init__(self, run, sub, filename=None):
        self.run = run
        self.sub = sub
        self.filename = filename
        print(' -- reading a top drift electronics file')
        
        """ some TDE specific parameters """
        self.evskey = 0xFF
        self.endkey = 0xF0
        self.evdcard0 = 0x19 #default number of cards disconnected

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


    def get_filename(self):
        run_path = fname.get_run_directory(self.run)
        path = f"{cf.data_path}/{run_path}/{self.run}_{self.sub}"
        fl = glob.glob(path+".*")
        if(len(fl) != 1):
            print('none or more than one file matches ... : ', fl)
            exit()
        f = fl[0]
        return f


    def open_file(self):
        f = self.filename if self.filename else self.get_filename()
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

        if(len(dc.evt_list)==0):
            dc.evt_list.append( dc.event("dp","top", head['run_num'][0], self.sub, ievt, head['trig_num'][0], head['time_s'][0], head['time_ns'][0]) )
        else:
            print(head['run_num'][0], self.sub, ievt, head['trig_num'][0], head['time_s'][0], head['time_ns'][0])

    def read_evt(self, ievt):
        if(self.lro < 0 or self.cro < 0):
            print(' please read the event header first! ')
            exit()


        idx = self.event_pos[ievt] + self.header_size
        self.f_in.seek(idx,0)
        out = read_evt_uint12_nb( self.f_in.read(self.cro) )


        self.f_in.read(1) #The bruno byte :-)

        """ read second header"""
        head = np.frombuffer(self.f_in.read(self.header_size), dtype=self.header_type)
        self.cro = head['cro'][0]
        out = np.append(out,read_evt_uint12_nb( self.f_in.read(self.cro) ))



        if(len(out)/cf.n_sample != cf.n_tot_channels):
            print(' The event is incomplete ... ')
            exit()

        out = out.astype(np.float32)
        dc.data_daq = np.reshape(out, (cf.n_tot_channels, cf.n_sample))

        self.lro = -1
        self.cro = -1


    def close_file(self):
        self.f_in.close()
        print('file closed!')




class _50l_decoder(decoder):
    def __init__(self, run, sub, filename=None):
        self.run = run
        self.sub = sub
        self.filename = filename
        self.detector = "50l"

        self.evt_byte_size = 129528
        self.n_femb = 4       
        
    def get_filename(self):
        run_path = fname.get_run_directory(self.run)
        path = f"{cf.data_path}/{run_path}/"
        fl = glob.glob(path+"*.bin")
        s = int(self.sub)
        if(len(fl) == 0 or len(fl) < s):
            print('file numbering do not match ...')#
            print('--> ', path, " contains ", len(fl), " files")
            exit()
        f = fl[s]
        return f


    def open_file(self):
        f = self.filename if self.filename else self.get_filename()
        print('Reconstructing ', f)
        self.filename = f
        self.f_in = open(f,'rb')
        
        


    def read_run_header(self):
        import os
        self.fsize = os.path.getsize(self.filename)

        n_evt = int(self.fsize/self.evt_byte_size)
        return n_evt


    def read_evt_header(self, ievt):
        name = self.filename.replace('.bin','')
        fsplit = name.split('_')        
        timestamp = int(fsplit[-1])

        t_s = int(timestamp/100)
        t_ns = (timestamp - t_s*100) * 1e7

        dc.evt_list.append( dc.event(self.detector,"bot", self.run, self.sub, ievt, ievt, t_s, t_ns) )

    
    def read_evt(self, ievt):
        idx = ievt * self.evt_byte_size
        self.f_in.seek(idx,0)

        t = []
        for i in range(self.n_femb):

            self.f_in.read(9*2)
            t.extend(np.fromfile(self.f_in, dtype='>u2', count=3824))

            for k in range(3):
                self.f_in.read(8*2)
                t.extend(np.fromfile(self.f_in, dtype='>u2', count=3825))
            self.f_in.read(8*2)
            t.extend(np.fromfile(self.f_in, dtype='>u2', count=851))
        dt = np.dtype(np.uint16)
        tt = np.asarray(t, dtype=dt)

        """ remove the extra byte """
        tt = tt.reshape(-1,25)[:,1:].flatten()

        out = read_evt_uint12_nb_RD( tt)
        
        out = np.reshape(out, (-1, 32)).T # split into 32 channel chunks
        out = np.reshape(out, (32, 4, 646)) # Reshape into 
        out = out.swapaxes(0,1)
        out = np.reshape(out, (128, 646))
        dc.data_daq = out
        dc.data_daq[:64,:] *= 2


    def close_file(self):
        self.f_in.close()
        print('file closed!')
