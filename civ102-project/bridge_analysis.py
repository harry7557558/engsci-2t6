import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import beam_analysis
import cross_section_analysis as csa


# matboard parameters

from rectpack import newPacker  # https://github.com/secnot/rectpack

MATBOARD_W = 1016
MATBOARD_H = 813

def pack_rect(rects, labels=None, ax=None):
    packer = newPacker()
    if labels is None:
        for r in rects:
            packer.add_rect(*r)
    else:
        for r, label in zip(rects, labels):
            packer.add_rect(*r, label)
    packer.add_bin(MATBOARD_W, MATBOARD_H)
    packer.pack()
    abin = packer[0]
    if len(abin) < len(rects):
        return False
    if ax is not None:  # plot
        def plot_rect(x, y, w, h, fmt='-', label=''):
            ax.plot([x, x+w, x+w, x, x], [y, y, y+h, y+h, y], fmt)
        plot_rect(0, 0, MATBOARD_W, MATBOARD_H, 'k-', label='matboard')
        legend = ['matboard']
        for rect in abin:
            plot_rect(rect.x, rect.y, rect.width, rect.height, label=rect.rid)
            legend.append(rect.rid)
            #ax.annotate(rect.rid, (rect.x+0.5*rect.width, rect.y+0.5*rect.height))
        ax.axis('equal')
        ax.legend(legend, ncol=3)
    return True


# bridge parameters

MAX_REACTION, (MAX_SHEAR_X, MAX_SHEAR_V), (MAX_BEND_X, MAX_BEND_V) = \
              beam_analysis.plot_max_responses(False)

def get_max_shear(x0, x1):
    x = np.linspace(x0, x1)
    sf = np.interp(x, MAX_SHEAR_X, MAX_SHEAR_V)
    return np.amax(sf)

def get_max_bend(x0, x1):
    x = np.linspace(x0, x1)
    sf = np.interp(x, MAX_BEND_X, MAX_BEND_V)
    return np.amax(sf)


# cross section

class BridgeCrossSection:

    def __init__(self, label, parts, glues, x0, x1):
        assert x0 < x1
        self.solved = False
        self.label = label
        self.x0 = x0
        self.x1 = x1
        self.parts = [[np.array(p) for p in part] for part in parts]
        self.glues = [[np.array(p) for p in glue] for glue in glues]
        if len(parts) == 0:
            return
        self.perimeter, (self.xc, self.yc), (self.Ix, self.I) \
                        = csa.calc_geometry(self.parts)
        self.area = self.perimeter * (self.x1 - self.x0)

    def solve(self):
        if self.solved:
            return
        self.bend_max = get_max_bend(self.x0, self.x1)
        self.shear_max = get_max_shear(self.x0, self.x1)
        self.max_bm = csa.calc_max_bending_moment(self.parts, self.yc, self.I)
        self.max_bm_b = csa.calc_buckling_moment(self.parts, self.yc, self.I)
        self.shear_f = csa.calc_shear_factor(self.parts, self.yc, self.I)
        self.max_shear = 1.0/self.shear_f.get_optim(absolute=True)[1]
        self.fos_bend = self.max_bm / self.bend_max
        self.fos_buckle = self.max_bm_b / self.bend_max
        self.fos_shear = self.max_shear / self.shear_max
        self.fos = min(self.fos_bend, self.fos_buckle, self.fos_shear)
        self.solved = True

    def get_rects(self):
        res = []
        for part in self.parts:
            peri = 0
            for (p1, p2) in zip(part[:-1], part[1:]):
                peri += np.linalg.norm(p2-p1)
            res.append((peri, self.x1-self.x0))
        return res


# bridge analysis

def vandercorput(n, b):
    x = 0.0
    e = 1.0 / b
    while n != 0:
        d = n % b
        x += d * e
        e /= b
        n //= b
    return x

PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37,
          41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]


class Bridge:

    def __init__(self,
                 calc_cross_section: 'Callable',
                 calc_diaphrams: 'Callable',
                 param_domains, param_labels,
                 initial_params):
        """
            @calc_cross_section:
                receives *params
                returns list[BridgeCrossSection]
                returns None if parameters violate constraints
            @calc_diaphrams:
                receives *params
                returns list[BridgeCrossSection]
                diaphram is at (x0+x1)/2
                two strips around the diaphram
            @param_domains
                [(a0, a1), (b0, b1), ...]
            @initial_params
                [a, b, ...]
        """
        self.calc_cross_section = calc_cross_section
        self.calc_diaphrams = calc_diaphrams
        self.param_domains = param_domains
        self.param_labels = param_labels
        self.params = initial_params[:]

    def calc_glues(self, cross_sections):
        """Calculate glue joints 
            set glues of the parameter to empty
            returns list[BridgeCrossSection] with only glues
            label is True if it is a concern of flexural shear failure"""
        for cs in cross_sections:
            cs.glues = []
        # might include duplicate glue joints
        # underestimate is better than overestimate
        res = []
        for j in range(len(cross_sections)):
            cs2 = cross_sections[j]
            for i in range(j):
                cs1, cs2 = cross_sections[i], cross_sections[j]
                if (cs1.x0, cs1.x1) > (cs2.x0, cs2.x1):
                    cs1, cs2 = cs2, cs1
                if cs2.x0 >= cs1.x1:
                    continue
                glues = csa.intersection_pieces(cs1.parts, cs2.parts)
                cs = BridgeCrossSection(cs1.x1 <= cs2.x1, [], glues,
                                        cs2.x0, min(cs1.x1, cs2.x1))
                res.append(cs)
        return res

    def analyze(self, params, plot=False):
        cross_sections = self.calc_cross_section(*params)
        if cross_sections is None:
            if plot:
                print("Cross section constraint violation.")
            return -1

        diaphrams = self.calc_diaphrams(*params)
        if diaphrams is None:
            if plot:
                print("Diaphram constraint violation.")
            return -1
        diaphrams_cs = [0.0, 1250.0]
        for d in diaphrams:
            c = 0.5*(d.x0+d.x1)
            cross_sections += [
                #BridgeCrossSection(d.label, d.parts, d.glues, d.x0, c),
                #BridgeCrossSection(d.label, d.parts, d.glues, c, d.x1),                BridgeCrossSection(d.label, d.parts, d.glues, d.x0, c),
                BridgeCrossSection(d.label, d.parts, d.glues, d.x0, d.x1)
            ]
            diaphrams_cs.append(c)
        diaphrams_cs.sort()

        # must fit the size of the matboard
        tot_area = sum([cs.area for cs in cross_sections])
        if tot_area > MATBOARD_W*MATBOARD_H:
            if plot:
                print("Area too large.")
            return -1
        rects = sum([cs.get_rects() for cs in cross_sections], []) + \
                sum([csa.cross_section_range(d.parts) for d in diaphrams], [])
        if not pack_rect(rects):
            if plot:
                print("Can't pack into matboard.")
            return -1

        # glue joints
        glues = self.calc_glues(cross_sections)

        # divide into sections
        xs = diaphrams_cs[:]
        for cs in cross_sections:
            xs += [cs.x0, cs.x1]
        xs = sorted(set(xs))

        if plot:
            cs_x = []
            cs_bend, cs_bend_buckle = [], []
            cs_shear, cs_shear_buckle = [], []
            cs_flexshear = []

        # for each cross section
        min_fos = float('inf')
        for (x0, x1) in zip(xs[:-1], xs[1:]):
            
            # bending moment and shear
            csa.BM_MAX = max(get_max_bend(x0, x1), 1)
            csa.SHEAR_MAX = max(get_max_shear(x0, x1), 1)
            csa.LENGTH = 0
            for dc1, dc2 in zip(diaphrams_cs[:-1], diaphrams_cs[1:]):
                if dc1 <= x0 < x1 <= dc2:
                    csa.LENGTH = dc2 - dc1
                    break
            assert csa.LENGTH != 0
            parts = []
            for cs in cross_sections:
                if x0 >= cs.x0 and x1 <= cs.x1:
                    parts += cs.parts
            #if len(parts) == 0:
            #    return -1
            fos_bend, fos_bend_buckle, fos_shear, fos_shear_buckle = \
                      csa.analyze_cross_section(
                parts, [], return_full=True)
            
            # shear at glue joints due to flexual stress
            fos_flexshear = float('inf')
            peri, (xc, yc), (Ix, I) = csa.calc_geometry(parts)
            for glue in glues:
                if glue.label == False:
                    continue
                if not glue.x0 <= x0 < x1 <= glue.x1:
                    assert x0 >= glue.x1 or x1 <= glue.x0
                    continue
                for (p1, p2) in glue.glues:
                    maxy = max(abs(p1[1]-yc), abs(p2[1]-yc))
                    dFdl = csa.BM_MAX*maxy/I * csa.LINDEN_M
                    h = dFdl / (glue.x1-glue.x0)
                    if plot:
                        #print(maxy, csa.BM_MAX, csa.BM_MAX*maxy/I)
                        #print(glue.x1-glue.x0)
                        pass
                    fos_flexshear = min(fos_flexshear, csa.SHEAR_C/h)
                    
            # apply
            if True:  # varying FoS for each type
                fos_bend /= 2
                fos_bend_buckle /= 3
                fos_shear /= 1.5
                fos_shear_buckle /= 3
                fos_flexshear /= 40  # don't think I calculated this one correctly
            min_fos = min(min_fos,
                          fos_bend, fos_bend_buckle,
                          fos_shear, fos_shear_buckle,
                          fos_flexshear)
            if plot:
                #print("fos_flexshear", fos_flexshear)
                #print(x0, x1, fos_bend)
                xs = np.linspace(x0, x1)
                one = np.ones(len(xs))
                bms = np.interp(xs, MAX_BEND_X, MAX_BEND_V)
                sfs = np.interp(xs, MAX_SHEAR_X, MAX_SHEAR_V)
                c_bend = fos_bend * csa.BM_MAX / bms
                c_bend_buckle = fos_bend_buckle * csa.BM_MAX / bms
                c_shear = fos_shear * csa.SHEAR_MAX / sfs
                c_shear_buckle = fos_shear_buckle * csa.SHEAR_MAX / sfs
                c_flexshear = fos_flexshear * csa.BM_MAX / bms
                cs_x += xs.tolist()
                cs_bend += c_bend.tolist()
                cs_bend_buckle += c_bend_buckle.tolist()
                cs_shear += c_shear.tolist()
                cs_shear_buckle += c_shear_buckle.tolist()
                cs_flexshear += c_flexshear.tolist()

        if plot:
            fig, (ax1, ax2) = plt.subplots(2, 1)
            one = np.float64(1)
            ax1.plot(cs_x, one/cs_bend, label="flexural fos⁻¹")
            ax1.plot(cs_x, one/cs_bend_buckle, label="bend buckle fos⁻¹")
            ax1.plot(cs_x, one/cs_shear, label="shear fos⁻¹")
            ax1.plot(cs_x, one/cs_shear_buckle, label="shear buckle fos⁻¹")
            ax1.plot(cs_x, one/cs_flexshear, label="flex shear fos⁻¹")
            ax1.legend()
            rects = sum([cs.get_rects() for cs in cross_sections], []) + \
                    sum([csa.cross_section_range(d.parts) for d in diaphrams], [])
            labels = sum([[cs.label]*len(cs.parts) for cs in cross_sections], []) + \
                     sum([[d.label]*len(d.parts) for d in diaphrams], [])
            pack_rect(rects, labels, ax2)
            ax2.set(xlim=(-100, 3000), ylim=(-100, MATBOARD_H+100))
            plt.show()

        return min_fos

    def random_param(self, seed):
        params = []
        for i in range(len(self.param_domains)):
            t = vandercorput(seed+100000, PRIMES[i])
            c0, c1 = self.param_domains[i]
            params.append(c0+c1*t)
        return params

    def rand_normal(self, seed, mu, sigma):
        params = [*mu]
        for i in range(len(params)):
            r1 = vandercorput(seed, PRIMES[2*i])
            r2 = vandercorput(seed, PRIMES[2*i+1])
            randn = np.sqrt(-2.0*np.log(1.0-r1)) * np.sin(2.0*np.pi*r2)
            params[i] += sigma * max(params[i], 1.0) * randn
        return np.array(params)

    def optimize_params(self):
        opt_params = self.params
        opt_fos = self.analyze(self.params)
        print(opt_params, opt_fos)

        # brute force search
        for seed in range(0):
            params = self.random_param(seed)
            fos = self.analyze(params)
            if fos > opt_fos:
                print(seed, fos)
                opt_params, opt_fos = params, fos
                if fos > 1:
                    break  # uncomment
        opt_params = np.array(opt_params)

        # optimize - what the heck?
        maxi = 1000
        conv_params, conv_fos = opt_params, opt_fos
        for i in range(maxi):
            alive = 0.5 * 0.01**(i/maxi)
            params = self.rand_normal(i, conv_params, alive)
            fos = self.analyze(params)
            if not fos > 0.0:
                continue
            if fos > opt_fos:
                opt_params, opt_fos = params, fos
                print(f"{i+1}/{maxi}", fos)
            #w = 0.0 if fos > conv_fos else 1.0
            w = 0.001 ** max(fos / conv_fos - (1.0-alive), 0)
            conv_params = conv_params * w + params * (1.0-w)
            conv_fos = conv_fos * w + fos * (1.0-w)

        # do a "proper" optimization
        niter = 20
        for i in range(niter):
            optres = scipy.optimize.minimize(
                lambda params: -self.analyze(params),
                x0=opt_params,
                method='Nelder-Mead',
                options={ 'maxfev': 100 })
            prev_fos = opt_fos
            opt_params, opt_fos = optres.x, -optres.fun
            print(f"{i+1}/{niter}", opt_fos)
            if i >= 5 and opt_fos / prev_fos < 1.002:
                break
            if opt_fos == prev_fos:
                break
        
        for label, val in zip(self.param_labels, opt_params):
            print(label, '=', val)
        print("FoS =", self.analyze(opt_params))
        print(opt_params.tolist())
        self.analyze(opt_params, plot=True)

    def plot_3d(self, zoom=1.0):
        ax = plt.axes(projection='3d')

        cross_sections = self.calc_cross_section(*self.params)
        if cross_sections is None:
            print("Cross section constraint violation")
            cross_sections = []
        for cs in cross_sections:
            x0, x1 = cs.x0, cs.x1
            for part in cs.parts:
                for p in part:
                    ax.plot3D([x0, x1], [p[0], p[0]], [p[1], p[1]], 'gray')
                ys = [p[0] for p in part]
                zs = [p[1] for p in part]
                ax.plot3D([x0]*len(part), ys, zs, 'gray')
                ax.plot3D([x1]*len(part), ys, zs, 'gray')
                for xe in [50, 1200]:
                    if x0 <= xe <= x1:
                        ax.plot3D([xe]*len(part), ys, zs, 'red')

        diaphrams = self.calc_diaphrams(*self.params)
        if diaphrams is None:
            print("Diaphram constraint violation")
            diaphrams = []
        for d in diaphrams:
            x0, x1 = d.x0, d.x1
            xc = 0.5*(x0+x1)
            for part in d.parts:
                for p in part:
                    ax.plot3D([x0, x1], [p[0], p[0]], [p[1], p[1]], 'black')
                ys = [p[0] for p in part]
                zs = [p[1] for p in part]
                ax.plot3D([x0]*len(part), ys, zs, 'black')
                ax.plot3D([x1]*len(part), ys, zs, 'black')
                ax.plot3D([xc]*len(part), ys, zs, 'black')

        # equal aspect ratio - https://stackoverflow.com/a/63625222/16318343
        limits = np.array([
            ax.get_xlim3d(),
            ax.get_ylim3d(),
            ax.get_zlim3d(),
        ])
        center = np.mean(limits, axis=1)
        radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0])) / zoom
        ax.set_xlim3d([center[0] - radius, center[0] + radius])
        ax.set_ylim3d([center[1] - radius, center[1] + radius])
        ax.set_zlim3d([center[2] - radius, center[2] + radius])

        plt.show()


if __name__ == "__main__":
    pass
