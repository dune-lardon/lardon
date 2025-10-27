import numpy as  np


def get_trigger_header(det):

    if(det == 'cb1'):
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

    elif( det == 'cb'):        
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
        print('ERROR unknown detector', det)
        return -1



def get_component_header(det):
    if(det == 'cb1'):
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
    elif(det == 'cb'):
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
        print('ERROR unknown detector', det)
        return -1



def get_fragment_header(det):

    if(det == "cb1"):        
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

    elif(det == "cb"):
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
        print('ERROR unknown detector', det)
        return -1


def get_wib_header(det):
    if(det == "cb1"):
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
        
    elif(det == "cb"):
        wib_header_type = np.dtype([
            ('word1', '<u4'), #4
            ('ts', '<u8'),    #12
            ('word2', '<u4'), #16
            ('word3', '<u4')  #20
        ])

        return wib_header_type

    else:
        print('ERROR unknown detector', det)
        return -1
