import math
""" computations based on 
Pierre Billoir Nucl.Instrum.Meth. A225 (1984) 352-366
information matrix is the inverse of the covariance matrix
"""
class PFilter:
    def __init__(self, erry, errs, pbeta):
        self.erry = erry
        self.errs = errs
        self.pbeta = pbeta
        self.reset()

    def reset(self):
        self.opt = [0.0, 0.0]
        self.info = [0.0, 0.0, 0.0]
        self.chi2 = 0.0

    def initiate(self, y, a):
        self.reset()
        self.opt[0] = y
        self.opt[1] = a
        self.info[0] = 1.0 / (4.0 * self.erry * self.erry)
        self.info[1] = 2.0 if a == 0.0 else 1.0 / (0.25 * a * a)
        self.info[2] = 0.0

    def det(self, m):
        return m[0] * m[1] - m[2] * m[2]

    def invert(self, m):
        d = self.det(m)
        return [m[1]/d, m[0]/d, -m[2]/d] if d != 0 else [0.0, 0.0, 0.0]

    def getY(self): return self.opt[0]
    def getSlope(self): return self.opt[1]

    def getYerr(self):
        d = self.det(self.info)
        return math.sqrt(self.info[1] / d) if d != 0 else float('inf')

    def getSlopeErr(self):
        d = self.det(self.info)
        return math.sqrt(abs(self.info[0] / d)) if d != 0 else float('inf')

    def getCorr(self):
        erry = self.getYerr()
        errs = self.getSlopeErr()
        d = self.det(self.info)
        return (self.info[2] / d) / (erry * errs) if erry and errs else 0.0

    def predict(self, step):
        return self.opt[0] + step * self.opt[1]

    def delta_y(self, ymeas, step):
        return abs(self.predict(step) - ymeas)

    def computeChi2(self, ymeas, step):
        res = self.predict(step) - ymeas
        det = self.det(self.info)
        errcov = self.info[1] / det if det != 0 else 10.0
        return res * res / (self.erry * self.erry + errcov)

    def chi2_if_update(self, ymeas, step):
        ypred, apred = self.predict(step), self.opt[1]
        cov = self.multScatt(step)
        inv_info = self.invert(self.info)
        info = self.invert([i + c for i, c in zip(inv_info, cov)])

        step2 = step * step
        info0 = info[0]
        info2 = info[2] = -info0 * step + info[2]
        info1 = info[1] = info0 * step2 - 2 * info2 * step + info[1]

        m00_inv = 1.0 / (self.erry * self.erry)
        i_m = [info0 + m00_inv, info1, info2]
        det = self.det(i_m)

        common1 = m00_inv * ymeas + info0 * ypred + info2 * apred
        common2 = info2 * ypred + info1 * apred

        yopt = (info1 * common1 - info2 * common2) / det
        aopt = (-info2 * common1 + i_m[0] * common2) / det

        ydelta, adelta = yopt - ypred, aopt - apred
        chi2meas = i_m[0] * ydelta * ydelta + info1 * adelta * adelta + 2.0 * info2 * ydelta * adelta
        return chi2meas

    def update(self, ymeas, step):
        ypred, apred = self.predict(step), self.opt[1]
        cov = self.multScatt(step)
        inv_info = self.invert(self.info)
        info = self.invert([i + c for i, c in zip(inv_info, cov)])

        step2 = step * step
        info0 = info[0]
        info2 = info[2] = -info0 * step + info[2]
        info1 = info[1] = info0 * step2 - 2.0 * info2 * step + info[1]

        m00_inv = 1.0 / (self.erry * self.erry)
        i_m = [info0 + m00_inv, info1, info2]
        det = self.det(i_m)

        common1 = m00_inv * ymeas + info0 * ypred + info2 * apred
        common2 = info2 * ypred + info1 * apred

        yopt = (info1 * common1 - info2 * common2) / det
        aopt = (-info2 * common1 + i_m[0] * common2) / det

        self.opt[0] = yopt
        self.opt[1] = aopt

        self.info[0] = info0 + m00_inv
        self.info[1] = info1
        self.info[2] = info2

        ydelta, adelta = yopt - ypred, aopt - apred
        chi2meas = self.info[0] * ydelta * ydelta + self.info[1] * adelta * adelta + 2.0 * self.info[2] * ydelta * adelta
        self.chi2 += chi2meas
        return chi2meas

    def multScatt(self, step):
        """ from 
        The Kalman Filter Technique applied to Track Fitting in GLAST
        by Jose Hernando 
        """

        if self.pbeta == 0.0 or step == 0.0:
            return [0.0, 0.0, 0.0]

        X0 = 14.0  # LAr radiation length
        theta2 = (0.0136 / self.pbeta) ** 2 * abs(step) / X0
        angle = self.opt[1]
        incFac = (1.0 + angle * angle) ** 2.5 if not math.isnan(angle) else 1.0
        err = theta2 * incFac

        step_abs = abs(step)
        return [
            err * step * step / 3.0,
            err,
            err * step_abs / 2.0
        ]

    def getChi2(self):
        return self.chi2
