import lardon.config as cf
import lardon.data_containers as dc
import lardon.lar_param as lar

from rtree import index
import numpy as np

def min_distance(xx_a, zz_a, mm_a, xx_b, zz_b, mm_b):
    
    min_dist = 99999
    pts = [(-1, -1, -1, -1)]
    for xa, za, ma, ii in zip(xx_a, zz_a, mm_a, [0,1]):
        for xb, zb, mb, jj in zip(xx_b, zz_b, mm_b, [0,1]):
            d = np.sqrt(pow(xb-xa,2)+pow(zb-za,2))

            if(d < min_dist):
                min_dist = d
                mod = (ma, mb)
                idx = (ii,jj)
                pts[0] = (xa,za,xb,zb)
    return min_dist, mod, idx, pts


def ghost_finder():

    if(dc.reco['ghost']['search'] == False):
        return

    dist_thresh = dc.reco['ghost']['dmin']
    
    """ to be called before 3D finder """

    """ 2D tracks in collection view """
    tracks = [t for t in dc.tracks2D_list if t.view == 2]
    n_trk = len(tracks)


    if(n_trk < 2):
        #print("cannot search for ghosts, bye")
        return 

    best_match = [-1 for x in range(n_trk)]
    best_mindist  = [-1 for x in range(n_trk)]

    for i,ti in enumerate(tracks):
        i_direction = cf.drift_direction[ti.module_ini]
        
        yi_start, zi_start = ti.path[0][0], ti.path[0][1]
        yi_stop, zi_stop   = ti.path[-1][0], ti.path[-1][1]
        qi = ti.tot_charge
        mi_ini, mi_end = ti.module_ini, ti.module_end
        
        best_dist = 9999
        best_idx  = -1
        
        for j in range(i+1, n_trk):

            tj = tracks[j]
            if(i==j):
                continue
            yj_start, zj_start = tj.path[0][0], tj.path[0][1]
            yj_stop, zj_stop   = tj.path[-1][0], tj.path[-1][1]
            qj = tj.tot_charge
            mj_ini, mj_end = tj.module_ini, tj.module_end

            min_2Ddist, mod, idx, _ = min_distance([yi_start, yi_stop], [zi_start, zi_stop], [mi_ini, mi_end], [yj_start, yj_stop], [zj_start, zj_stop], [mj_ini, mj_end])

            if(mod[0] == mod[1]):
                if(min_2Ddist < dist_thresh):                
                    if(min_2Ddist < best_dist):
                        best_dist = min_2Ddist
                        best_idx = j
                        print(idx)

        best_match[i] = best_idx
        best_mindist[i] = best_dist

    for i,ti in enumerate(tracks):
        if(ti.ghost == True):
            continue

        idx = best_match[i]
        if(idx < 0):
            continue
        nmatch = best_match.count(idx)

        if(nmatch > 1):
            continue
        dmin = best_mindist[i]
        tj = tracks[idx]
        if(tj.ghost == True):
            continue

        qi = ti.tot_charge
        qj = tj.tot_charge

        if(ti.ini_slope*tj.ini_slope>=0):
            continue

        if(qi < qj):
            ti.ghost = True
            ghost = dc.ghost(ti.trackID, tj.trackID, dmin, qi, qj, ti.n_hits, [ti.module_ini, ti.module_end])
            dc.ghost_list.append(ghost)
            
            print('-----')
            print('THE GHOST: ')
            ti.mini_dump()
            print('THE  TRACK ')
            tj.mini_dump()
            print('\n')
        else:
            tj.ghost = True
            ghost = dc.ghost(tj.trackID, ti.trackID, dmin, qj, qi, tj.n_hits, [tj.module_ini, tj.module_end])
            dc.ghost_list.append(ghost)
            
            print('-----')
            print('THE GHOST: ')
            tj.mini_dump()
            print('THE  TRACK ')
            ti.mini_dump()
            print('\n')

def find_2d_track(ID):
    for t in dc.tracks2D_list:
        if(t.trackID == ID):
            return t
    return None

def find_3d_track(ID):
    for t in dc.tracks3D_list:
        if(t.ID_3D == ID):
            return t
    return None


def ghost_trajectory():
    if(dc.reco['ghost']['search'] == False):   
        return

    debug = False

    ang_track = np.radians(cf.view_angle[0][2])
    pitch = cf.view_pitch[2]

    prev_ghost_list = dc.ghost_list.copy()
    dc.ghost_list.clear()
    for g in prev_ghost_list:
        ghost_track = find_2d_track(g.ghost_ID)
        t2d = find_2d_track(g.trk2D_ID)
        if(ghost_track == None or t2d == None):        
            continue

        if(t2d.match_3D < 0):
            continue

        t3d = find_3d_track(t2d.match_3D)
        t3d.dump()
        print('with')
        t2d.mini_dump()
        if(np.all([m[2] for m in t3d.match_ID]) < 0):
            continue

        ghost = dc.ghost(ghost_track.trackID, t2d.trackID, g.min_dist, ghost_track.tot_charge, t2d.tot_charge, ghost_track.n_hits, [ghost_track.module_ini,ghost_track.module_end] )

        ghost_track.set_match_hits_ghost(t3d.ID_3D)
        
        dc.ghost_list.append(ghost)
        dc.evt_list[-1].n_ghosts += 1

        yc_start, zc_start = ghost_track.path[0][0], ghost_track.path[0][1]
        yc_stop, zc_stop   = ghost_track.path[-1][0], ghost_track.path[-1][1]
        g_modules = [ghost_track.module_ini,ghost_track.module_end]
        
        x_start, y_start, z_start = t3d.path[2][0][0], t3d.path[2][0][1], t3d.path[2][0][2]
        x_stop, y_stop, z_stop = t3d.path[2][-1][0], t3d.path[2][-1][1], t3d.path[2][-1][2]
        t_modules = [t3d.module_ini,t3d.module_end]        

        min_2Ddist, mod, idx, pts = min_distance([yc_start, yc_stop], [zc_start, zc_stop], g_modules, [y_start, y_stop], [z_start, z_stop], t_modules)
        print('--> ', min_2Ddist, mod, idx)
        
        if(pts[0][3] == z_start):
            x_3dcontact, y_3dcontact, z_3dcontact = x_start, y_start, z_start
            theta = t3d.ini_theta
            phi   = t3d.ini_phi
        else:
            x_3dcontact, y_3dcontact, z_3dcontact = x_stop, y_stop, z_stop
            theta = t3d.end_theta
            phi   = t3d.end_phi

        """ this is actually useless """
        if(pts[0][1] == zc_start):
            y_2dcontact, z_2dcontact = yc_start, zc_start
        else:
            y_2dcontact, z_2dcontact = yc_stop, zc_stop

        theta = np.radians(theta)
        phi   = np.pi + np.radians(phi) #add pi because back interpolation

        ux = np.tan(theta)*np.cos(phi)        

        angle = np.sin(theta)*np.sin(phi)
        
        dr = pitch/angle if angle!=0 else pitch

            
        ghost_path = ghost_track.path
        ghost_charge = ghost_track.dQ
            
        n_pts = len(ghost_path)
            
        trajectory = []
        dQ = []
        ds = []
        hits = ghost_track.hits_ID

        for p in range(n_pts):
            y = ghost_path[p][0]
            dz = z_3dcontact - ghost_path[p][1]
            z = z_3dcontact + dz
            x = x_3dcontact+-1.*dz*ux
            trajectory.append( (x,y,z) )

            dQ.append(ghost_charge[p])
            ds.append(dr)

            
        ghost.set_3D_ghost(t2d.match_3D, trajectory, dQ, ds, hits, x_3dcontact, y_3dcontact, z_3dcontact, np.degrees(theta), np.degrees(phi-np.pi), t3d.t0_corr, t3d.z0_corr)

        if(debug):
            import matplotlib.pyplot as plt
            """ debugging plots """
            fig = plt.figure()
            ax_xz = fig.add_subplot(121)
            ax_yz = fig.add_subplot(122, sharey=ax_xz)
            trk3d_path = t3d.path[2]

            
            xz_3d = [(x[0],x[2]) for x in trk3d_path]
            ax_xz.scatter(*zip(*xz_3d), marker='o', color='k')
            xz_ghost = [(x[0],x[2]) for x in trajectory]
            ax_xz.scatter(*zip(*xz_ghost), marker='o', color='r')
            
            
            yz_3d = [(x[1],x[2]) for x in trk3d_path]
            ax_yz.scatter(*zip(*yz_3d), marker='o', color='k', label='3D track')
            yz_ghost = [(x[1],x[2]) for x in trajectory]
            ax_yz.scatter(*zip(*yz_ghost), marker='o', color='r', label='3D ghost')
            ax_yz.scatter(*zip(*ghost_path), marker='o', color='tab:cyan', label='2D Ghost' )


            ax_xz.set_xlabel('X [cm]')
            ax_xz.set_ylabel('Z [cm]')

            ax_yz.set_xlabel('Y [cm]')
            ax_yz.set_ylabel('Z [cm]')

            ax_yz.legend(frameon=False)

            plt.show()
            plt.close()

        

