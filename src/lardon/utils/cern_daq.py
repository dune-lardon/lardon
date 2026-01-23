""" LZ: To decode 50L data (before summer 2023) taken at CERN. The DAQ may have come from an other experiment, but I don't know which one """ 
import numpy as  np
import numba as nb

import lardon.channel_mapping as cmap
import lardon.config as cf
import lardon.data_containers as dc


@nb.jit(nopython = True)
def read_uint12_bit_sbnd_nb(data):
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


class cern:
    def __init__(self, f, daq, det):

        self.filename = f

        try:
            self.f_in = open(f, 'rb')
        except IOError:
            print('File ', f, ' does not exist...')
            exit()



        self.daq = daq
        self.det = det
            

        self.evt_byte_size = 129528
        self.n_femb = 4       

    def set_run(self, run):
        self.run = run

    def read_evt_header(self, sub, ievt, flow):
        name = self.filename.replace('.bin','')
        fsplit = name.split('_')        
        timestamp = int(fsplit[-1])

        t_s = int(timestamp/100)
        t_ns = (timestamp - t_s*100) * 1e7

        dc.evt_list.append( dc.event(self.det, "bot", self.run, sub, ievt, ievt, timestamp, t_s, t_ns, 0) )


    def read_run_header(self):
        import os
        self.fsize = os.path.getsize(self.filename)

        n_evt = int(self.fsize/self.evt_byte_size)
        return n_evt

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

        out = read_uint12_bit_sbnd_nb( tt)
        
        out = np.reshape(out, (-1, 32)).T # split into 32 channel chunks
        out = np.reshape(out, (32, 4, 646)) # Reshape into 
        out = out.swapaxes(0,1)
        out = np.reshape(out, (128, 646))
        dc.data_daq = out
        dc.data_daq[:64,:] *= 2

    def read_pds_evt(self, ievt):
        print('no such feature exists')

    def close_file(self):
        self.f_in.close()
        #print('file closed!')

