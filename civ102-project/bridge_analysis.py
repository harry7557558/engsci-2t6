import numpy as np
import matplotlib.pyplot as plt

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
        def plot_rect(x, y, w, h, fmt='-'):
            ax.plot([x, x+w, x+w, x, x], [y, y, y+h, y+h, y], fmt)
        plot_rect(0, 0, MATBOARD_W, MATBOARD_H, 'k-')
        legend = ['matboard']
        for rect in abin:
            plot_rect(rect.x, rect.y, rect.width, rect.height)
            legend.append(rect.rid)
        ax.axis('equal')
        ax.legend(legend)
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

    def __init__(self, label: str, parts, glues, x0, x1):
        assert x0 < x1
        self.label = label
        self.x0 = x0
        self.x1 = x1
        self.parts = [[np.array(p) for p in part] for part in parts]
        self.glues = [[np.array(p) for p in glue] for glue in glues]
        self.perimeter, self.yc, self.I = csa.calc_geometry(self.parts, self.glues)
        self.area = self.perimeter * (self.x1 - self.x0)
        self.solved = False

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
                 param_domains, param_labels,
                 initial_params):
        """
            @calc_cross_section:
                receives *params
                returns list[BridgeCrossSection]
                returns None if parameters violate constraints
            @param_domains
                [(a0, a1), (b0, b1), ...]
            @initial_params
                [a, b, ...]
        """
        self.calc_cross_section = calc_cross_section
        self.param_domains = param_domains
        self.param_labels = param_labels
        self.params = initial_params[:]
        #self.analyze(self.params, plot=True)
        self.optimize_params()

    def analyze(self, params, plot=False):
        cross_sections = self.calc_cross_section(*params)
        if cross_sections is None:
            return -1
        for cs in cross_sections:
            #cs.solve()
            pass

        # must fit the size of the matboard
        tot_area = sum([cs.area for cs in cross_sections])
        if tot_area > MATBOARD_W*MATBOARD_H:
            return -1
        rects = [(cs.perimeter, cs.x1-cs.x0) for cs in cross_sections]
        if not pack_rect(rects):
            return -1

        # divide into sections
        xs = []
        for cs in cross_sections:
            xs += [cs.x0, cs.x1]
        xs = sorted(set(xs))

        if plot:
            cs_x = []
            cs_bend, cs_buckle, cs_shear = [], [], []

        min_fos = float('inf')
        for (x0, x1) in zip(xs[:-1], xs[1:]):
            csa.BM_MAX = get_max_bend(x0, x1)
            csa.SHEAR_MAX = get_max_shear(x0, x1)
            csa.MAX_PERI = MATBOARD_W*MATBOARD_H/(x1-x0)
            parts, glues = [], []
            for cs in cross_sections:
                if x0 >= cs.x0 and x1 <= cs.x1:
                    parts += cs.parts
                    glues += cs.glues
            if len(parts) == 0:
                return -1
            fos_bend, fos_buckle, fos_shear = \
                      csa.analyze_cross_section(
                parts, glues, return_full=True)
            min_fos = min(min_fos, fos_bend, fos_buckle, fos_shear)
            if plot:
                #print(x0, x1, fos_bend)
                xs = np.linspace(x0, x1)
                bms = np.interp(xs, MAX_BEND_X, MAX_BEND_V)
                sfs = np.interp(xs, MAX_SHEAR_X, MAX_SHEAR_V)
                c_bend = fos_bend * csa.BM_MAX / bms
                c_buckle = fos_buckle * csa.BM_MAX / bms
                c_shear = fos_shear * csa.SHEAR_MAX / sfs
                cs_x += xs.tolist()
                cs_bend += c_bend.tolist()
                cs_buckle += c_buckle.tolist()
                cs_shear += c_shear.tolist()

        if plot:
            fig, (ax1, ax2) = plt.subplots(2, 1)
            one = np.float64(1)
            ax1.plot(cs_x, one/cs_bend, label="bend fos⁻¹")
            ax1.plot(cs_x, one/cs_buckle, label="buckle fos⁻¹")
            ax1.plot(cs_x, one/cs_shear, label="shear fos⁻¹")
            ax1.legend()
            rects = [(cs.perimeter, cs.x1-cs.x0) for cs in cross_sections]
            labels = [cs.label for cs in cross_sections]
            pack_rect(rects, labels, ax2)
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
        params = mu[:]
        for i in range(len(params)):
            r1 = vandercorput(seed, PRIMES[2*i])
            r2 = vandercorput(seed, PRIMES[2*i+1])
            randn = np.sqrt(-2.0*np.log(1.0-r1)) * np.sin(2.0*np.pi*r2)
            params[i] += sigma * max(params[i], 1.0) * randn
        return params

    def optimize_params(self):
        opt_params = self.params
        opt_fos = self.analyze(self.params)
        print(opt_params, opt_fos)
        # brute force search
        for seed in range(10000):
            params = self.random_param(seed)
            fos = self.analyze(params)
            if fos > opt_fos:
                print(seed, fos)
                opt_params, opt_fos = params, fos
        # anneal
        maxi = 1000
        for i in range(maxi):
            temp = 0.5 * 0.01**(i/maxi)
            params = self.rand_normal(i, opt_params, temp)
            fos = self.analyze(params)
            if fos > opt_fos:
                print(i, fos)
                opt_params, opt_fos = params, fos
        for label, val in zip(self.param_labels, opt_params):
            print(label, '=', val)
        self.analyze(opt_params, plot=True)


if __name__ == "__main__":
    pass
