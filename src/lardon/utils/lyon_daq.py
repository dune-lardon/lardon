import lardon.channel_mapping as cmap
import lardon.config as cf
import lardon.data_containers as dc

import numpy as  np
import numba as nb




@nb.jit(nopython = True)
def read_uint12_bit_nb(data):
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



def get_header():
    header_type = np.dtype([
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
    return header_type

def open_file(f):
    f_type = f[f.rfind('.')+1:]
    
    return 


class lyon:
    def __init__(self, f, daq, det, run):
        self.filename = f
        f_type = f[f.rfind('.')+1:]
        print(' NB : file type is ', f_type)
        try: 
            self.f_in = open(f, 'rb')
        except IOError:
            print('File ', f, ' does not exist...')
            exit()
        
        self.daq = daq
        self.det = det
        self.run = run
        """Lyon DAQ control keys"""
        self.evskey = 0xFF
        self.endkey = 0xF0

        """number of cards disconnected"""
        if(self.det == 'cb1top'):
            self.evdcard0 = 0x5
        elif(self.det == 'dp'):
            self.evdcard0 = 0x19
        elif(self.det == 'cbtop'):
            self.evdcard0 = 0x0
        elif(self.det == 'pdvd'):
            self.evdcard0 = 0x2

            
        self.header = get_header()
        self.header_size = self.header.itemsize

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


    def read_evt_header(self, sub, ievt, flow):
        ''' flow is a useless parameter '''
        self.f_in.seek(self.event_pos[ievt],0)

        head = np.frombuffer(self.f_in.read(self.header_size), dtype=self.header)


        if( not((head['k0'][0] & 0xFF)==self.evskey and (head['k1'][0] & 0xFF)==self.evskey)):
            print(" problem in the event header ")
        good_evt = (head['evt_flag'][0] & 0x3F) == self.evdcard0

        if(not good_evt):
            print("  the event has ", head['evt_flag'][0] & 0x3F, ' cards disconnected instead of ', self.evdcard0)

        self.lro = head['lro'][0]
        self.cro = head['cro'][0]

        print('EVENT TIMESTAMP', head['time_s'][0])
        timestamp = head['time_s'][0] + head['time_ns'][0]*1e-9
        print('-->', timestamp)
        dc.evt_list.append( dc.event(self.det, "top", head['run_num'][0], sub, ievt, head['trig_num'][0], timestamp, head['time_s'][0], head['time_ns'][0], 0) )
        dc.evt_list[-1].set_charge_timestamp(cf.imod, timestamp)#t_s, t_ns)

    def read_evt(self, ievt):


        if(self.daq != 'lyon_pdvd'):
            if(self.lro < 0 or self.cro < 0):
                print(' please read the event header first! ')
                exit()
            idx = self.event_pos[ievt] + self.header_size
            self.f_in.seek(idx,0)
            out = read_uint12_bit_nb( self.f_in.read(self.cro) )


            
        if(self.daq == 'lyon_pdvd'):

                idx = self.event_pos[ievt] #+ self.header_size
                self.f_in.seek(idx,0)
                head = np.frombuffer(self.f_in.read(self.header_size), dtype=self.header)
                self.cro = head['cro'][0]

                out = read_uint12_bit_nb( self.f_in.read(self.cro) )

                self.f_in.read(1) #The bruno byte :-)

                """ read second header"""
                head = np.frombuffer(self.f_in.read(self.header_size), dtype=self.header)
                self.cro = head['cro'][0]
                out = np.append(out,read_uint12_bit_nb( self.f_in.read(self.cro) ))

                
        if(self.daq == 'lyon_2_blocks'):
            self.f_in.read(1) #The bruno byte :-)

            """ read second header"""
            head = np.frombuffer(self.f_in.read(self.header_size), dtype=self.header)
            self.cro = head['cro'][0]
            out = np.append(out,read_uint12_bit_nb( self.f_in.read(self.cro) ))


            
        #if(len(out)/cf.n_sample[cf.imod] != cf.n_tot_channels):
        if(len(out)/cf.n_sample[cf.imod] != cf.module_nchan[cf.imod]):
            print(' The event is incomplete ... ', len(out), ', ',cf.n_sample[cf.imod], '=', len(out)/cf.n_sample[cf.imod], ' vs ', cf.module_nchan[cf.imod])
            exit()

        out = out.astype(np.float32)
        #dc.data_daq = np.reshape(out, (cf.n_tot_channels, cf.n_sample[cf.imod]))
        dc.data_daq = np.reshape(out, (cf.module_nchan[cf.imod], cf.n_sample[cf.imod]))

        self.lro = -1
        self.cro = -1


    def read_pds_evt(self, ievt):
        print('no such feature exists')
        
    def close_file(self):
        self.f_in.close()
        #print('file closed!')
