import data_containers as dc
import config as cf
import numpy as np
from rtree import index


class R_tree:
    def __init__(self, rcut):
        #self.dX = cf.ChanPitch/2.
        self.rcut = rcut

    def create_index(self, view):
        self.dX = cf.view_pitch[view]
        self.tree = index.Index()

    def insert_hit(self, hit, idx):
        stop = min(hit.Z_stop, hit.Z_start)
        start = max(hit.Z_stop, hit.Z_start)
        self.tree.insert(idx, (hit.X-self.dX, stop, hit.X+self.dX, start))

    def remove_hit(self, hit, idx):
        stop = min(hit.Z_stop, hit.Z_start)
        start = max(hit.Z_stop, hit.Z_start)
        self.tree.delete(idx, (hit.X-self.dX, stop, hit.X+self.dX, start))
    def n_hits(self):
        return self.tree.get_size()

    def infos(self):
        print(self.tree)

    def nearest_id(self, hit, n):
        stop = min(hit.Z_stop, hit.Z_start)
        start = max(hit.Z_stop, hit.Z_start)                
        return list(self.tree.nearest((hit.X-self.dX, stop, hit.X+self.dX, start), n))

    def overlap_in_time(self, hA, hB):
        b, a = hA.Z_start, hA.Z_stop
        d, c = hB.Z_start, hB.Z_stop
        return not(b < c or d < a)

    def distance(self, hA, hB):
        dx = hB.X - hA.X
        #dz = hB.Z - hA.Z
        dz = self.short_distance_z(hA, hB)
        return np.sqrt(dx*dx + dz*dz)

    def short_distance_z(self, hA, hB):
        b, a = hA.Z_start, hA.Z_stop
        d, c = hB.Z_start, hB.Z_stop
        return min(np.fabs(a-c), np.fabs(a-d), np.fabs(b-d), np.fabs(b-c))

    def peak_distance(self, hA, hB):
        dx = hB.X - hA.X
        dz = hB.Z - hA.Z
        return np.sqrt(dx*dx + dz*dz)


    def close_enough(self, hA, hB):
        if(hA.ID == hB.ID):
            """ same hit """
            return False

        #if(self.distance_z(hA, hB) < ):#self.overlap_in_time(hA, hB)):
        return self.distance(hA, hB) < self.rcut
