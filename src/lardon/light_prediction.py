import lardon.config as cf
import lardon.data_containers as dc
import lardon.lar_param as lar


import lardon.plotting as plot

import numpy as np


class light_prediction:
    def __init__(self, labs):

        self.param_file=cf.pds_pred_fit_param
        self.get_correction_parameters()
        self.labs = labs
        self.alpha = 0.21 #Nex/Ni
        self.beta = 1./(1.+self.alpha)
        self.Wion = 23.6 #excitation energy in eV
        self.Wq = self.beta * self.Wion
        self.dedx_mip = 2.1 #MeV
        
        self.n_photons_per_cm = [(1-self.beta*lar.recombination(imod=i))*self.dedx_mip*1e6/self.Wq for i in range(cf.n_module)]
        #self.n_photons_per_cm = [(1-lar.recombination(imod=i))*self.dedx_mip*1e6/self.Wq for i in range(cf.n_module)]
        
        self.tpc_bounds = (min([cf.x_boundaries[i][0] for i in range(cf.n_module)]), max([cf.x_boundaries[i][1] for i in range(cf.n_module)]),
                           min([cf.y_boundaries[i][0] for i in range(cf.n_module)]), max([cf.y_boundaries[i][1] for i in range(cf.n_module)]),
                           min(cf.anode_z), max(cf.anode_z))

        
        
    def get_correction_parameters(self):
        if(self.param_file is None):
            print('No corrections ')
            self.correction_function = self.correct_detector_effect_dummy
            return


        with open(self.param_file, 'r') as fparams:
            infos = fparams.readline().split('\t')
            function = infos[0]
            nparam = int(infos[1])
            print(function, ' with ', nparam, ' parameters')
            
            if(function == "modified_gaisser_hillas"):
                #print('modified gaisser hillas!')
                self.correction_function = self.correct_detector_effect_modified_gaisser_hillas
            if(function == "poly_4"):
                #print('modified gaisser hillas!')
                self.correction_function = self.correct_detector_effect_polynomial4

            else:
                self.correction_function = self.correct_detector_effect_dummy
                print('no ',function,' correction function defined')
                


                #self.Nmax, self.dmax, self.d0, self.lambda_low, self.lambda_high = [], [], [], [], []
            self.correction_params = []
            for lines in fparams.readlines()[1:]:
                li = lines.split('\t')
                p = [float(x) for x in li[1:]]
                #print(p)
                self.correction_params.append(p)
            #print(len(self.correction_params))



    def set_track(self, P0, P1, tdir, light_channels):
        self.P0 = P0
        self.tdir = tdir / np.linalg.norm(tdir)

        """ t0, t1 gives the full line extrapolation in the TPC AV """
        self.t0, self.t1 = self.clip_line_to_box()

        """ for the muon decay test: track goes from ta (highest z extrapolation) to tb (lowest track z point) """
        """ ATTENTION FOR PDHD """
        if(cf.tpc_orientation == 'Vertical'):
            idx = 2
        else:
            idx = 1
            
        self.ta = self.t0 if self.P0[idx]+self.t0*self.tdir[idx] > self.P0[idx]+self.t1*self.tdir[idx] else self.t1
        low_trk = self.P0 if self.P0[idx] < P1[idx] else P1
        self.tb = np.dot(np.asarray(low_trk) - np.asarray(self.P0), self.tdir)
        #(low_trk[idx]-self.P0[idx])/self.tdir[idx]#np.dot(np.asarray(low_trk) - np.asarray(self.P0), self.tdir)
        
        if(self.ta > self.tb):
            self.ta, self.tb = self.tb, self.ta

        
        self.ta = np.clip(self.ta, self.t0, self.t1)
        self.tb = np.clip(self.tb, self.t0, self.t1)

        """
        print('track extrapolation: ', self.t0, self.t1)
        print('extrapolated track goes from ',(self.P0 + self.t0*self.tdir),'to', (self.P0 + self.t1*self.tdir))

        print('(clipped) from muon decay test: ', self.ta, self.tb)
        print('extrapolated dk track goes from ',(self.P0 + self.ta*self.tdir),'to', (self.P0 + self.tb*self.tdir))
        """
        L = np.linalg.norm((self.P0 + self.t1*self.tdir) - (self.P0 + self.t0*self.tdir))
        n_points = int(L)
        self.ds = L / n_points
        self.ts = np.linspace(self.t0, self.t1, n_points)

        L_dk = np.linalg.norm((self.P0 + self.ta*self.tdir) - (self.P0 + self.tb*self.tdir))
        n_points_dk = int(L_dk)
        self.ds_dk = L_dk / n_points_dk
        self.ts_dk = np.linspace(self.ta, self.tb, n_points_dk)

        #print('L full AV ', L, " L decay ", L_dk, " --> ratio ", L_dk/L)
        light_modules = list(set([dc.chmap_pds[ch].module for ch in light_channels]))

        """ per module """
        geo_prediction = [self.predict_light_geometric(m)  if m in light_modules else (0,[],0,0, 0, 0) for m in range(cf.pds_n_modules)]
        
        self.geo_prediction_per_module        = [x[0] for x in geo_prediction]
        self.geo_dk_prediction_per_module     = [x[1] for x in geo_prediction]
        
        self.track_maxval_point_per_module    = [x[2] for x in geo_prediction]
        self.track_maxval_distance_per_module = [x[3] for x in geo_prediction]
        self.track_dk_maxval_point_per_module    = [x[4] for x in geo_prediction]
        self.track_dk_maxval_distance_per_module = [x[5] for x in geo_prediction]
        """
        self.track_maxval_costheta_per_module = [x[3] for x in geo_prediction]
        """
        
        close = [self.closest_distance(m) if m in light_modules else (0,[],0, 0, [], 0,[]) for m in range(cf.pds_n_modules)]
        self.track_closest_point_per_module    = [x[0] for x in close]
        self.track_closest_distance_per_module = [x[1] for x in close]
        self.track_closest_costheta_per_module = [x[2] for x in close]

        self.track_dk_closest_point_per_module    = [x[3] for x in close]
        self.track_dk_closest_distance_per_module = [x[4] for x in close]
        self.track_dk_closest_costheta_per_module = [x[5] for x in close]
        self.pds_impact_point_per_module          = [x[6] for x in close]

        
        """ per channels """
        self.pds_impact_point_per_channels = [self.pds_impact_point_per_module[dc.chmap_pds[ch].module]  for ch in light_channels]
        
        self.track_closest_distance_per_channels = [self.track_closest_distance_per_module[dc.chmap_pds[ch].module]  for ch in light_channels]
        self.track_closest_point_per_channels=[self.track_closest_point_per_module[dc.chmap_pds[ch].module]  for ch in light_channels]
        self.track_closest_costheta_per_channels=[self.track_closest_costheta_per_module[dc.chmap_pds[ch].module]  for ch in light_channels]
        self.track_maxval_distance_per_channels = [self.track_maxval_distance_per_module[dc.chmap_pds[ch].module]  for ch in light_channels]
        self.track_maxval_point_per_channels=[self.track_maxval_point_per_module[dc.chmap_pds[ch].module]  for ch in light_channels]

        
        self.track_dk_closest_distance_per_channels=[self.track_dk_closest_distance_per_module[dc.chmap_pds[ch].module]  for ch in light_channels]
        self.track_dk_closest_point_per_channels=[self.track_dk_closest_point_per_module[dc.chmap_pds[ch].module]  for ch in light_channels]
        self.track_dk_closest_costheta_per_channels=[self.track_dk_closest_costheta_per_module[dc.chmap_pds[ch].module]  for ch in light_channels]
        self.track_dk_maxval_distance_per_channels=[self.track_dk_maxval_distance_per_module[dc.chmap_pds[ch].module]  for ch in light_channels]
        self.track_dk_maxval_point_per_channels=[self.track_dk_maxval_point_per_module[dc.chmap_pds[ch].module]  for ch in light_channels]



        """
        self.track_maxval_distance_per_channels=[self.track_maxval_distance_per_module[dc.chmap_pds[ch].module]  for ch in light_channels]
        self.track_maxval_point_per_channels=[self.track_maxval_point_per_module[dc.chmap_pds[ch].module]  for ch in light_channels]
        self.track_maxval_costheta_per_channels=[self.track_maxval_costheta_per_module[dc.chmap_pds[ch].module]  for ch in light_channels]
        """

        
        self.geo_prediction_per_channels = [self.geo_prediction_per_module[dc.chmap_pds[ch].module] for ch in light_channels]
        self.prediction_per_channels = [self.correction_function(ch, self.track_closest_distance_per_module[dc.chmap_pds[ch].module])*self.geo_prediction_per_module[dc.chmap_pds[ch].module] for ch in light_channels]


        
        self.geo_dk_prediction_per_channels = [self.geo_dk_prediction_per_module[dc.chmap_pds[ch].module] for ch in light_channels]
        self.dk_prediction_per_channels = [self.correction_function(ch, self.track_dk_closest_distance_per_module[dc.chmap_pds[ch].module])*self.geo_dk_prediction_per_module[dc.chmap_pds[ch].module] for ch in light_channels]



            
    def clip_line_to_box(self):
        bxmin, bxmax, bymin, bymax, bzmin, bzmax = self.tpc_bounds
        t0, t1 = -np.inf, np.inf

        for i, (p, di, mn, mx) in enumerate([
                (self.P0[0], self.tdir[0], bxmin, bxmax),
                (self.P0[1], self.tdir[1], bymin, bymax),
                (self.P0[2], self.tdir[2], bzmin, bzmax)]):
            if abs(di) < 1e-12:
                # Line parallel: must lie inside slab
                if p < mn or p > mx:
                    #print('CASE A', di)
                    t0,t1= None, None
            else:
                tmin = (mn - p) / di
                tmax = (mx - p) / di
                if tmin > tmax:
                    tmin, tmax = tmax, tmin
                t0 = max(t0, tmin)
                t1 = min(t1, tmax)
        #print('-->', t0, t1)
        if t0 > t1:
            return t1,t0#0,1#None, None

        return t0, t1

    def closest_distance(self, pds_mod):
        x_center = cf.pds_x_center[pds_mod]
        y_center = cf.pds_y_center[pds_mod]
        z_center = cf.pds_z_center[pds_mod]
        
        x_length = cf.pds_x_length[pds_mod]/2.
        y_length = cf.pds_y_length[pds_mod]/2.
        z_length = cf.pds_z_length[pds_mod]/2.

        pds_bounds = (x_center-x_length, x_center+x_length,
                      y_center-y_length, y_center+y_length,
                      z_center-z_length, z_center+z_length)



        xmin, xmax, ymin, ymax, zmin, zmax = pds_bounds
        txmin, txmax, tymin, tymax, tzmin, tzmax = self.tpc_bounds
        
        faces = []
        center = []
        if xmin == xmax:  # YZ plane
            faces.append(('x', xmin, ymin, ymax, zmin, zmax))
            pds_normal = np.array([1, 0, 0])
            
        if ymin == ymax:  # XZ plane
            faces.append(('y', ymin, xmin, xmax, zmin, zmax))
            pds_normal = np.array([0, 1, 0])
            
        if zmin == zmax:  # XY plane
            faces.append(('z', zmin, xmin, xmax, ymin, ymax))
            pds_normal = np.array([0, 0, 1])

        # clamp line to rectangle
        def clamp_to_rect(P, xmin, xmax, ymin, ymax, zmin, zmax):
            return np.array([
                np.clip(P[0], xmin, xmax),
                np.clip(P[1], ymin, ymax),
                np.clip(P[2], zmin, zmax)
            ])

        #clamp line to finite segement (decay case)
        def clamp_line_to_segment(t):
            return np.clip(t, self.ta, self.tb)
        
        # clamp line point to TPC volume
        def clamp_line_to_tpc_point(t_hit):            
            if t_hit < self.t0:
                t_hit = self.t0
            if t_hit > self.t1:
                t_hit = self.t1
            return t_hit, self.P0 + t_hit * self.tdir


        # closest point between line and segment
        def closest_point_line_segment(P0, d, A, B):
            AB = B - A
            AP = P0 - A
            dAB = np.dot(d, AB)
            ABAB = np.dot(AB, AB)
            dd = np.dot(d, d)
            denom = dd * ABAB - dAB * dAB

            # Line parallel to segment
            if abs(denom) < 1e-12:
                t = np.dot(d, A - P0) / dd
                C = P0 + t * d
                s = np.dot(AB, C - A) / ABAB
                s = np.clip(s, 0, 1)
                Dp = A + s * AB
                return C, Dp

            t = (np.dot(d, AP) * ABAB - np.dot(AB, AP) * dAB) / denom
            C = P0 + t * d

            # Clamp s to segment
            s = (np.dot(d, AP) + t * dAB) / ABAB
            s = np.clip(s, 0, 1)
            Dp = A + s * AB

            return C, Dp

        
        # Compute closest distance over all faces + edges
        best_dist, best_dist_dk = np.inf, np.inf
        best_cl, best_cl_dk = None, None
        best_cp, best_cp_dk = None, None


    
        for axis, c, a1min, a1max, a2min, a2max in faces:
            t_hit = 0
            # ----- Line-plane projection -----
            if axis == 'x':
                if abs(self.tdir[0]) < 1e-12:
                    Pproj = self.P0.copy()
                    Pproj[0] = c
                else:
                    t_hit = (c - self.P0[0]) / self.tdir[0]
                    Pproj = self.P0 + t_hit * self.tdir
                Cp = clamp_to_rect(Pproj, c, c, a1min, a1max, a2min, a2max)

            elif axis == 'y':
                if abs(self.tdir[1]) < 1e-12:
                    Pproj = self.P0.copy()
                    Pproj[1] = c
                else:
                    t_hit = (c - self.P0[1]) / self.tdir[1]
                    Pproj = self.P0 + t_hit * self.tdir
                Cp = clamp_to_rect(Pproj, a1min, a1max, c, c, a2min, a2max)

            else:  # axis == 'z'
                if abs(self.tdir[2]) < 1e-12:
                    Pproj = self.P0.copy()
                    Pproj[2] = c
                else:
                    t_hit = (c - self.P0[2]) / self.tdir[2]
                    Pproj = self.P0 + t_hit * self.tdir
                Cp = clamp_to_rect(Pproj, a1min, a1max, a2min, a2max, c, c)

            # Closest point on line
            t_line = np.dot(self.tdir, Cp - self.P0)
            t_line, Cl = clamp_line_to_tpc_point( t_line)

            dist = np.linalg.norm(Cl - Cp)
            if dist < best_dist:
                best_dist = dist
                best_cl = Cl
                best_cp = Cp

            #segment case
            #t_line = np.dot(self.tdir, Cp - self.P0)
            t_line = clamp_line_to_segment(t_line)
            Cl_dk = self.P0 + t_line * self.tdir

            dist_dk = np.linalg.norm(Cl_dk - Cp)
            if dist_dk < best_dist_dk:
                best_dist_dk = dist_dk
                best_cl_dk = Cl_dk
                best_cp_dk = Cp
                
            # ---------- Also check the 4 edges ----------
            if axis == 'x':
                A = np.array([c, a1min, a2min])
                B = np.array([c, a1max, a2min])
                C = np.array([c, a1max, a2max])
                D = np.array([c, a1min, a2max])
            elif axis == 'y':
                A = np.array([a1min, c, a2min])
                B = np.array([a1max, c, a2min])
                C = np.array([a1max, c, a2max])
                D = np.array([a1min, c, a2max])
            else:
                A = np.array([a1min, a2min, c])
                B = np.array([a1max, a2min, c])
                C = np.array([a1max, a2max, c])
                D = np.array([a1min, a2max, c])

            edges = [(A, B), (B, C), (C, D), (D, A)]

            for e0, e1 in edges:
                Cl_e, Cp_e = closest_point_line_segment(self.P0, self.tdir, e0, e1)

                # clamp line point to TPC
                t_e = np.dot(self.tdir, Cl_e - self.P0)
                t_e, Cl_e = clamp_line_to_tpc_point(t_e)

                dist_e = np.linalg.norm(Cl_e - Cp_e)
                if dist_e < best_dist:
                    best_dist = dist_e
                    best_cl = Cl_e
                    best_cp = Cp_e


                #t_e = np.dot(self.tdir, Cl_e - self.P0)
                t_e = clamp_line_to_segment(t_e)
                Cl_e_dk = self.P0 + t_e * self.tdir

                dist_e_dk = np.linalg.norm(Cl_e_dk - Cp_e)
                if dist_e_dk < best_dist_dk:
                    best_dist_dk = dist_e_dk
                    best_cl_dk = Cl_e_dk
                    best_cp_dk = Cp_e
                    
            direction = best_cl-best_cp
            cos_theta = np.dot(direction, pds_normal) / best_dist
            cos_theta = np.abs(np.clip(cos_theta, -1.0, 1.0))

            direction = best_cl_dk-best_cp_dk
            cos_theta_dk = np.dot(direction, pds_normal) / best_dist_dk
            cos_theta_dk = np.abs(np.clip(cos_theta_dk, -1.0, 1.0))


        return best_cl, best_dist, cos_theta, best_cl_dk, best_dist_dk, cos_theta_dk, best_cp




    def solid_angle_triangle_vec(self, a, b, c):
        # norms
        la = np.linalg.norm(a, axis=1)
        lb = np.linalg.norm(b, axis=1)
        lc = np.linalg.norm(c, axis=1)

        # triple product
        num = np.einsum('ij,ij->i', a, np.cross(b, c))
        
        # denominator (vectorized)
        den = (la*lb*lc
               + np.einsum('ij,ij->i', a, b)*lc
               + np.einsum('ij,ij->i', a, c)*lb
               + np.einsum('ij,ij->i', b, c)*la)

        return np.fabs(2.0 * np.arctan2(num, den))

    
    def quad_solid_angle(self, A, B, C, D, point):
        rA = A - point
        rB = B - point
        rC = C - point
        rD = D - point

        tri1 = self.solid_angle_triangle_vec(rA, rB, rC)
        tri2 = self.solid_angle_triangle_vec(rA, rC, rD)
        return np.abs(tri1 + tri2)
    
    def predict_light_geometric(self, pds_mod):

        # Precompute panel geometry once
        x_center = cf.pds_x_center[pds_mod]
        y_center = cf.pds_y_center[pds_mod]
        z_center = cf.pds_z_center[pds_mod]
        pds_eff   = cf.pds_eff[pds_mod]
    
        dx = cf.pds_x_length[pds_mod]/2.
        dy = cf.pds_y_length[pds_mod]/2.
        dz = cf.pds_z_length[pds_mod]/2.

        xmin, xmax = x_center-dx, x_center+dx
        ymin, ymax = y_center-dy, y_center+dy
        zmin, zmax = z_center-dz, z_center+dz

        # Build vertices once
        if xmin == xmax:
            A = np.array([xmin, ymin, zmin])
            B = np.array([xmin, ymax, zmin])
            C = np.array([xmin, ymax, zmax])
            D = np.array([xmin, ymin, zmax])
            pds_normal = np.array([1, 0, 0])
            
        elif ymin == ymax:
            A = np.array([xmin, ymin, zmin])
            B = np.array([xmax, ymin, zmin])
            C = np.array([xmax, ymin, zmax])
            D = np.array([xmin, ymin, zmax])
            pds_normal = np.array([0, 1, 0])
        else:
            A = np.array([xmin, ymin, zmin])
            B = np.array([xmax, ymin, zmin])
            C = np.array([xmax, ymax, zmin])
            D = np.array([xmin, ymax, zmin])
            pds_normal = np.array([0, 0, 1])
            
        # Vectorize points along the track
        points = self.P0 + np.outer(self.ts, self.tdir)
        points_dk = self.P0 + np.outer(self.ts_dk, self.tdir)

        # Distances to center (vectorized)
        pds_center = np.array([x_center, y_center, z_center])
        closest_dist = np.linalg.norm(points - pds_center, axis=1)
        closest_dist_dk = np.linalg.norm(points_dk - pds_center, axis=1)

        # Fast vectorized solid angle
        omega = self.quad_solid_angle(A, B, C, D, points) / (4*np.pi)
        omega_dk = self.quad_solid_angle(A, B, C, D, points_dk) / (4*np.pi)

        
        npe = np.exp(-closest_dist/self.labs) \
            * self.n_photons_per_cm[0] * self.ds \
            * pds_eff * omega

        npe_dk = np.exp(-closest_dist_dk/self.labs) \
            * self.n_photons_per_cm[0] * self.ds_dk \
            * pds_eff * omega_dk

        if(len(npe)>0):
            max_val = np.argmax(npe)
        else:
            max_val=0
        t_max = self.ts[max_val]
        point_max = self.P0 + t_max*self.tdir
        direction = point_max - pds_center
        dist_point_max = np.linalg.norm(point_max - pds_center)

        if(len(npe_dk)>0):
            dk_max_val = np.argmax(npe_dk)
        else:
            dk_max_val = 0
        dk_t_max = self.ts[dk_max_val]
        dk_point_max = self.P0 + dk_t_max*self.tdir
        dk_direction = dk_point_max - pds_center
        dk_dist_point_max = np.linalg.norm(dk_point_max - pds_center)

        """
        cos_theta = np.dot(direction, pds_normal) / dist_point_max
        cos_theta = np.abs(np.clip(cos_theta, -1.0, 1.0))
        """

        return npe.sum(), npe_dk.sum(), point_max, dist_point_max, dk_point_max, dk_dist_point_max#, cos_theta


    
    def correct_detector_effect_dummy(self, chan, dist):
        return 1.

    def correct_detector_effect_polynomial4(self, chan, dist):
        """ 4th order polynomial function as a function of track-pds closest distance """
        params = self.correction_params[chan]
        p4 = np.poly1d(params)
        return p4(dist)
    

    def correct_detector_effect_modified_gaisser_hillas(self, chan, dist):
        """ modified gaisser-hillas function as a function of track-pds closest distance """
        """ gets two lambda parameters for d < dmax or d>dmax """

        params = self.correction_params[chan]
        if(params[0] < 0):
            return 1.
        Nmax = params[0]
        dmax = params[1]
        d0   = params[2]
        lambda_val = params[3] if dist < dmax else params[4]
        term = (dist-d0) / (dmax-d0)
        if(term <= 0): term = 1e-12

        correction = Nmax * term **((dmax-d0)/lambda_val) * np.exp((dmax-dist)/lambda_val)
        return correction
