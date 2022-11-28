# Failure analysis + packing to the matboard + optimization
# Closely related to our design

import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import beam_analysis
import cross_section_analysis as csa


# if this is True, use different FoS for different failure modes
VARYING_FOS = False


# matboard parameters

from rectpack import newPacker  # https://github.com/secnot/rectpack, version 0.2.2

LENGTH = beam_analysis.LENGTH  # beam length
MATBOARD_W = 1016
MATBOARD_H = 813


# component labeling convention
def is_diaphragm_label(label):
    if label.endswith('='):  # diaphragm padding
        return False
    if '%' in label:  # put more pieces into one rectangle
        return True
    return label.startswith('d:') or label.startswith('de:')

def plot_rect_color(label):
    """For plotting"""
    if label in ['matboard', '']:
        return '#ccc'
    if 'support' in label:
        return '#ffd'
    if '%' in label:
        return '#fef'
    if label.startswith('d:'):
        return '#dff'
    return '#fff'

def pack_rect(rects, labels=None, ax=None):
    """Returns a boolean, whether the rectangles can pack into the matboard
        Plot the packed rectangles if @ax is not None"""
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
        assert len(rects) == len(labels)
        def plot_rect(x, y, w, h, fmt='-', label=''):
            xs = [x, x+w, x+w, x, x]
            ys = [y, y, y+h, y+h, y]
            ax.fill(xs, ys, plot_rect_color(label))
            ax.plot(xs, ys, fmt)
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


# cross section class

class BridgeCrossSection:

    def __init__(self, label, parts, glues=[], x0=None, x1=None, offset=0.0):
        assert (x0 is None and x1 is None) or x0 < x1
        self.label = label
        self.x0 = x0
        self.x1 = x1
        if type(offset) in [int, float]:
            offset = [offset] * len(parts)
        self.offset = offset
        self.parts = [[np.array(p) for p in part] for part in parts]
        if offset != 0:
            self.parts_offset = [self.calc_offset(part, o)
                                 for (part, o) in zip(self.parts, self.offset)]
        else:
            self.parts_offset = [part[:] for part in self.parts]
        self.glues = [[np.array(p) for p in glue] for glue in glues]
        if len(parts) == 0:
            return
        if x0 is None and x1 is None:
            return
        self.perimeter, (self.xc, self.yc), (self.Ix, self.I) \
                        = csa.calc_geometry(self.parts)
        self.area = self.perimeter * (self.x1 - self.x0)

    def get_rects(self):
        res = []
        for part in self.parts_offset:
            peri = 0
            for (p1, p2) in zip(part[:-1], part[1:]):
                peri += np.linalg.norm(p2-p1)
            res.append((peri, self.x1-self.x0))
        return res

    def get_folds(self):
        res = []
        for part in self.parts_offset:
            fold = []
            width = self.x1-self.x0
            peri = 0
            for (p1, p2) in zip(part[:-1], part[1:]):
                peri += np.linalg.norm(p2-p1)
                fold.append([
                    np.array([peri, 0]),
                    np.array([peri, width])
                ])
            fold = fold[:-1]
            if self.label.endswith('='):
                fold.append([
                    np.array([0, 0.5*width]),
                    np.array([peri, 0.5*width])
                ])
            res.append(fold)
        return res

    def assert_ccw(self):
        """make sure the points are in counter-clockwise order
            this only checks parts, not glues"""
        for i in range(len(self.parts_offset)):
            part = self.parts_offset[i][:]
            if len(part) > 2 and \
               np.linalg.norm(part[-1]-part[0]) > 1e-6:
                part.append(part[0])
            sA = 0
            for (p1, p2) in zip(part[:-1], part[1:]):
                sA += p1[0]*p2[1]-p1[1]*p2[0]
            if not sA > -1e-6:
                print(f"Assert CCW fail: {self.label} at index {i} (area={sA})")
                assert sA > -1e-6

    def assert_nointersect(self):
        """assert each part don't intersect each other
            assumes closed convex ccw polygons
            assumes two polygons are not identical"""
        # check if polygon edges intersect
        segs = []
        for i in range(len(self.parts_offset)):
            part = self.parts_offset[i]
            segs += list(zip(part[:-1], part[1:],
                             [i]*(len(part)-1)))
        for j in range(len(segs)):
            for i in range(j):
                p1, d1 = segs[i][0], segs[i][1]-segs[i][0]
                p2, d2 = segs[j][0], segs[j][1]-segs[j][0]
                if abs(d1[0]*d2[1]-d1[1]*d2[0]) < 1e-12:
                    continue
                t1, t2 = np.linalg.solve(np.transpose([d1, -d2]), p2-p1)
                eps = 1e-6
                if eps < t1 < 1-eps and eps < t2 < 1-eps:
                    print(f"Polygons intersect: {self.label} at indices {segs[i][2]},{segs[j][2]}")
                    assert False
        # check if one polygon is inside another polygon
        for j in range(len(self.parts_offset)):
            for i in range(len(self.parts_offset)):
                if i == j:
                    continue
                separator_found = False
                for (p1, p2) in zip(self.parts_offset[j][:-1], self.parts_offset[j][1:]):
                    n = np.array([p2[1]-p1[1], p1[0]-p2[0]])
                    is_separating = True
                    for p in self.parts_offset[i]:
                        if n.dot(p-p1) < -1e-6:
                            is_separating = False
                            break
                    if is_separating:
                        separator_found = True
                        break
                if not separator_found:
                    print(f"Polygons overlap: {self.label} at indices {i},{j}")
                    assert False

    def calc_offset(self, part, o):
        """inflate the part by o(ffset) (usually matboard thickness),
            negative o -> deflation,
            assume the cross section is ccw"""
        assert len(part) >= 2
        closed = np.linalg.norm(part[0]-part[-1])<1e-6
        # calculate normals
        normals = []
        for p1, p2 in zip(part[:-1], part[1:]):
            d = p2-p1
            d /= np.linalg.norm(d)
            n = np.array([d[1], -d[0]])
            normals.append(n)
        # offset
        def offset_endpoint(p, n):
            if abs(n[0]) > abs(n[1]) and o > 0:  # for multi-piece joints
                return p + np.array([o/n[0], 0]) + np.array([-n[1], n[0]]) * o * np.sign(n[0])
            else:
                return p + o * n
        new_part = [offset_endpoint(part[0], normals[0])]
        for (p1, p2, n1, n2) in zip(
                part[:-2], part[1:-1], normals[:-1], normals[1:]):
            p = np.linalg.solve([n1, n2],
                                [n1.dot(p1)+o, n2.dot(p2)+o])
            new_part.append(p)
        new_part.append(offset_endpoint(part[-1], normals[-1]))
        if len(part) <= 3:
            return new_part
        # resolve endpoints
        part = new_part
        d1, d2 = part[1]-part[0], part[-2]-part[-1]
        if abs(d1[0]*d2[1]-d1[1]*d2[0]) > 1e-12:
            t1, t2 = np.linalg.solve(np.transpose([d1, -d2]),
                                     part[-1] - part[0])
            eps = 1e-6
            if closed or (-eps < t1 < 1+eps and -eps < t2 < 1+eps):
                part[0] += d1 * t1
                part[-1] += d2 * t2
        return part


# bridge analysis

# quasi-random sequence for optimization
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
                 calc_diaphragms: 'Callable',
                 param_domains, param_labels,
                 initial_params):
        """
            @calc_cross_section:
                receives *params
                returns list[BridgeCrossSection]
                returns None if parameters violate constraints
            @calc_diaphragms:
                receives *params
                returns list[BridgeCrossSection]
                diaphragm is at (x0+x1)/2
                two strips around the diaphragm
            @param_domains
                [(a0, a1), (b0, b1), ...]
            @initial_params
                [a, b, ...]
        """
        self.calc_cross_section = calc_cross_section
        self.calc_diaphragms = calc_diaphragms
        self.param_domains = param_domains
        self.param_labels = param_labels
        self.params = initial_params[:]

    @staticmethod
    def merge_diaphragms(diaphragms, return_marks=False):
        """For calculating bounding rectangle"""
        merged = {}
        for d in diaphragms:
            if '%' not in d.label:
                assert d.label not in merged
                merged[d.label] = [d]
                continue
            mi = d.label[d.label.find('%'):]
            if mi not in merged:
                merged[mi] = []
            merged[mi].append(d)
        res = []
        marks = []
        labels = []
        for (mi, ds) in merged.items():
            labels.append('\n'.join(sum([
                [d.label[:(d.label+'%').find('%')]]*len(d.parts_offset)
                for d in ds], [])))
            if '%%' not in mi:
                parts = sum([d.parts_offset for d in ds], [])
                res.append(BridgeCrossSection(mi, parts))
                if return_marks:
                    ps = sum(parts, [])
                    minx = min([p[0] for p in ps])
                    miny = min([p[1] for p in ps])
                    dp = -np.array([minx, miny])
                    marks.append([[p+dp for p in part] for part in parts])
                continue
            # assume congruent trapezoids with the same orientation
            parts = []
            markst = []
            for i in range(len(ds)):
                assert len(ds[i].parts_offset) == 1
                ps = [p[:] for p in ds[i].parts_offset[0]]
                minx = min([p[0] for p in ps])
                miny = min([p[1] for p in ps])
                maxy = max([p[1] for p in ps])
                ps = [p-[minx, miny] for p in ps]
                maxy -= miny
                if i == 0:
                    parts.append(ps)
                    markst.append(ps)
                    continue
                if i % 2 == 1:  # turn it upside down
                    maxx = max([p[0] for p in ps])
                    ps = [[maxx,maxy]-p for p in ps]
                x0 = sorted(set([p[0] for p in parts[-1]]), reverse=True)[1]  # second largest x
                dp = np.array([x0, 0])
                ps = [p+dp for p in ps]
                parts.append(ps)
                markst.append(ps)
            res.append(BridgeCrossSection(mi, parts))
            marks.append(markst)
        if return_marks:
            return res, marks, labels
        return res

    def assert_ccw(self):
        """Make sure the pieces are in CCW, required for offsetting"""
        cross_sections = self.calc_cross_section(*self.params)
        if cross_sections is None:
            print("`assert_ccw`: Cross section constraint violation.")
            cross_sections = []
        for cs in cross_sections:
            cs.assert_ccw()
        diaphragms = self.calc_diaphragms(*self.params)
        if diaphragms is None:
            print("`assert_ccw`: Diaphragm constraint violation.")
            diaphragms = []
        for d in diaphragms:
            d.assert_ccw()
        # non-intersecting assertion
        merged = self.merge_diaphragms(diaphragms)
        for mcs in merged:
            mcs.assert_nointersect()

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

    @staticmethod
    def generate_rects(cross_sections, diaphragms,
                       require_labels: bool, require_marks=False):
        """Get a list of rectangles for packing into the matboard"""
        merged = Bridge.merge_diaphragms(diaphragms, require_marks)
        if require_marks:
            merged, merged_marks, merged_labels = merged
        rects = sum([cs.get_rects() for cs in cross_sections], []) + \
                [csa.cross_section_range(m.parts_offset, True) for m in merged]
        if not require_labels:
            return rects
        labels = sum([[cs.label[:(cs.label+'%').find('%')]]*len(cs.parts_offset)
                      for cs in cross_sections], []) + \
                 (merged_labels if require_marks else [m.label for m in merged])
        if not require_marks:
            assert len(rects) == len(labels)
            return rects, labels
        folds = sum([cs.get_folds() for cs in cross_sections], []) + \
                merged_marks
        assert len(labels) == len(folds)
        return rects, labels, folds

    def analyze(self, params, plot=False, show=True):
        """Failure load of the bridge, plot"""
        cross_sections = self.calc_cross_section(*params)
        if cross_sections is None:
            if plot:
                print("Cross section constraint violation.")
            return -1

        diaphragms = self.calc_diaphragms(*params)
        if diaphragms is None:
            if plot:
                print("Diaphragm constraint violation.")
            return -1
        diaphragms_cs = [0.0, LENGTH]
        for d in diaphragms:
            c = 0.5*(d.x0+d.x1)
            cross_sections += [
                BridgeCrossSection(d.label[:(d.label+'%').find('%')]+'=',
                                   d.parts, d.glues, d.x0, d.x1, offset=-csa.LINDEN_M),
            ]
            diaphragms_cs.append(c)
        diaphragms_cs.sort()

        # must fit the size of the matboard
        tot_area = sum([cs.area for cs in cross_sections])
        if tot_area > MATBOARD_W*MATBOARD_H:
            if plot:
                print("Area too large.")
            return -1
        rects = self.generate_rects(cross_sections, diaphragms, False)
        if not pack_rect(rects):
            if plot:
                print("Can't pack into matboard.")
            return -1

        # glue joints
        glues = self.calc_glues(cross_sections)
        flexshear_keypoints = []
        for cs in cross_sections:
            if 'beam' in cs.label:
                flexshear_keypoints += [
                    (cs.x0, cs.x1),
                    (cs.x1, cs.x0)
                ]
        flexshear_keypoints.sort(key=lambda p: p[0])
        flexshear_keypoints = dict(flexshear_keypoints)
        assert len(flexshear_keypoints) % 2 == 0

        # divide into sections
        xs = diaphragms_cs[:]
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
        has_flexshear = 0
        for (x0, x1) in zip(xs[:-1], xs[1:]):
            
            # bending moment and shear
            csa.BM_MAX = max(get_max_bend(x0, x1), 1)
            csa.SHEAR_MAX = max(get_max_shear(x0, x1), 1)
            csa.LENGTH = 0
            for dc1, dc2 in zip(diaphragms_cs[:-1], diaphragms_cs[1:]):
                if dc1 <= x0 < x1 <= dc2:
                    csa.LENGTH = dc2 - dc1
                    break
            assert csa.LENGTH != 0
            parts, parts_offset = [], []
            for cs in cross_sections:
                if x0 >= cs.x0 and x1 <= cs.x1:
                    parts += cs.parts
                    parts_offset += cs.parts_offset
            #if len(parts) == 0:
            #    return -1
            fos_bend, fos_bend_buckle, fos_shear, fos_shear_buckle = \
                      csa.analyze_cross_section(
                parts, parts_offset, return_full=True)
            
            # shear at glue joints due to flexual stress
            fos_flexshear = float('inf')
            if x0 in flexshear_keypoints:
                fx0, fx1 = x0, flexshear_keypoints[x0]
                has_flexshear += np.sign(fx1-fx0)
            if has_flexshear >= 2:
                peri, (xc, yc), (Ix, I) = csa.calc_geometry(parts)
                glue_to_check = []
                glue_area = 0.0
                for glue in glues:
                    if glue.label == False:
                        continue
                    if not glue.x0 <= x0 < x1 <= glue.x1:
                        assert x0 >= glue.x1 or x1 <= glue.x0
                        continue
                    glue_to_check.append(glue)
                    dx = csa.cross_section_range(glue.glues)
                    glue_area += (glue.x1-glue.x0)*sum([d[0] for d in dx])
                for glue in glue_to_check:
                    for (p1, p2) in glue.glues:
                        maxy = max(abs(p1[1]-yc), abs(p2[1]-yc))
                        dFdl = csa.BM_MAX*maxy/I * csa.LINDEN_M
                        h = dFdl / (glue.x1-glue.x0)  # failure stress for flexural shear
                        tf = 400/glue_area  # tension??
                        fos_flexshear = min(fos_flexshear,
                                            csa.SHEAR_C/np.hypot(h, tf))

            # apply
            if VARYING_FOS:  # varying FoS for each type
                fos_bend /= 2
                fos_bend_buckle /= 3
                fos_shear /= 1.5
                fos_shear_buckle /= 3
                fos_flexshear /= 30  # don't think I calculated this one correctly
            else:
                fos_flexshear /= 15
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
            print("FoS =", min_fos)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            one = np.float64(1)
            ax1.plot(cs_x, one/cs_bend, label="flexural fos⁻¹")
            ax1.plot(cs_x, one/cs_bend_buckle, label="bend buckle fos⁻¹")
            ax1.plot(cs_x, one/cs_shear, label="shear fos⁻¹")
            ax1.plot(cs_x, one/cs_shear_buckle, label="shear buckle fos⁻¹")
            #ax1.plot(cs_x, one/cs_flexshear, label="flex shear fos⁻¹")
            ax1.legend()
            rects, labels = self.generate_rects(cross_sections, diaphragms, True)
            pack_rect(rects, labels, ax2)
            ax2.set(xlim=(-100, 3000), ylim=(-100, MATBOARD_H+100))
            if show:
                plt.show()

        return min_fos

    # optimization stuff

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
        for seed in range(10000 if opt_fos < 0 else 0):
            params = self.random_param(seed)
            fos = self.analyze(params)
            if fos > opt_fos:
                print(seed, fos)
                opt_params, opt_fos = params, fos
                if fos > 1:
                    break  # uncomment
        opt_params = np.array(opt_params)

        # optimize - simulated annealing with momentum
        maxi = 10000
        conv_params, conv_fos = opt_params, opt_fos
        for i in range(maxi):
            if (i+1)%1000 == 0:
                print(f"{i+1}/{maxi}")
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

    def plot_3d(self, zoom=1.0, show=True):
        plt.figure(figsize=(8, 8))
        ax = plt.axes(projection='3d')

        cross_sections = self.calc_cross_section(*self.params)
        if cross_sections is None:
            print("Cross section constraint violation")
            cross_sections = []
        for cs in cross_sections:
            x0, x1 = cs.x0, cs.x1
            for part in cs.parts_offset:
                for p in part:
                    ax.plot3D([x0, x1], [p[0], p[0]], [p[1], p[1]], 'gray')
                ys = [p[0] for p in part]
                zs = [p[1] for p in part]
                ax.plot3D([x0]*len(part), ys, zs, 'gray')
                ax.plot3D([x1]*len(part), ys, zs, 'gray')
                for xe in [0.5*LENGTH-575, 0.5*LENGTH+575]:
                    if x0 <= xe <= x1:
                        ax.plot3D([xe]*len(part), ys, zs, 'red')

        diaphragms = self.calc_diaphragms(*self.params)
        if diaphragms is None:
            print("Diaphragm constraint violation")
            diaphragms = []
        for d in diaphragms:
            x0, x1 = d.x0, d.x1
            xc = 0.5*(x0+x1)
            for part in d.parts_offset:
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

        if show:
            plt.show()


if __name__ == "__main__":
    pass
