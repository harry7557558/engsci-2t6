import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

from PiecewisePolynomial import PiecewisePolynomial

# material properties
# length: mm; weight: N; stress: MPa
LINDEN_M = 1.27  # thickness of the matboard
SIGMA_T = 30.  # tensile strength of the matboard
SIGMA_C = 6.  # compressive strength of the matboard
SHEAR_M = 4.  # shear strength of the matboard
MU = 0.2  # Poisson's ratio of the matboard
E = 4000.  # Young's modulus of the matboard
MAX_PERI = 800.  # maximum allowed perimeter
LINDEN_C = 0.  # thickness of contact cement, assume cement has the same density as matboard
SHEAR_C = 2.  # shear strength of contact cement

# calculated from beam analysis
REACTION_MAX = 273
SHEAR_MAX = 255
BM_MAX = 66400

# variable?
LENGTH = __import__('beam_analysis').LENGTH  # cross section length

# misc?
TRAIN_WIDTH = 75


def calc_max_bending_moment(parts, yc, I):
    """Maximum allowed bending moment
        @parts: same as @parts in `analyze_cross_section`
        @yc: centroidal axis
        @I: second moment of area calculated at @yc
    """
    max_bm = float('inf')
    for part in parts:
        for p in part:
            y = p[1] - yc
            bml = max_bm
            if y > 0:  # compression
                bml = SIGMA_C * I / y
            elif y < 0:  # tension
                bml = SIGMA_T * I / -y
            max_bm = min(max_bm, bml)
    return max_bm


# split tensile and compressive calculation into two functions for the purpose of this assignment
SPLIT_BM = False

def calc_max_tension(parts, yc, I):
    max_bm = float('inf')
    for part in parts:
        for p in part:
            y = p[1] - yc
            if y < 0:
                max_bm = min(max_bm, SIGMA_T*I/-y)
    return max_bm

def calc_max_compression(parts, yc, I):
    max_bm = float('inf')
    for part in parts:
        for p in part:
            y = p[1] - yc
            if y > 0:
                max_bm = min(max_bm, SIGMA_C*I/y)
    return max_bm


def calc_shear_factor(parts, yc, I):
    """Maximum allowed shear force
            same parameters as `calc_max_bending_moment()`
            neglect glue
        Calculates Q(y) = -∫ydA as a PiecewisePolynomial;
        Calculates 1/b(y) as a PiecewisePolynomial;
        Returns Q/Ib
    """
    # keypoints
    key_ys = []  # (y, b(y) delta)
    for part in parts:
        for i in range(len(part)-1):
            (x1, y1), (x2, y2) = part[i], part[i+1]
            dA = LINDEN_M * np.hypot(x2-x1, y2-y1)
            if abs(y2-y1) < LINDEN_M:
                ym = 0.5*(y1+y2)
                y1, y2 = ym-0.5*LINDEN_M, ym+0.5*LINDEN_M
            dy = abs(y2-y1)
            key_ys.append((min(y1, y2), dA/dy))
            key_ys.append((max(y1, y2), -dA/dy))
    key_ys.sort(key=lambda x: x[0])
    key_ys1 = []
    for (y, b) in key_ys:
        if len(key_ys1) > 0 and y-key_ys1[-1][0] < 1e-8:
            key_ys1[-1][1] += b
            continue
        key_ys1.append([y, b])
    key_ys = key_ys1

    # get b = dA/dy
    b = 0
    y_prev = key_ys[0][0]
    ys, bs = [], []
    for (y, db) in key_ys:
        # integrate
        b += db
        y_prev = y
        # add piece
        ys.append(y)
        bs.append(b)
    while abs(bs[0]) < 1e-6:  # happens with bad optimization
        ys, bs = ys[1:], bs[1:]

    # integration
    dAdy_pieces, inv_b_pieces = [], []
    for (y, b) in zip(ys, bs):
        if y != ys[-1]:
            dAdy_pieces.append([b, 0.0])
            inv_b_pieces.append([1/max(b,2*LINDEN_M), 0.0])
    dAdy = PiecewisePolynomial(ys, dAdy_pieces)
    ydAdy = dAdy.polymul(PiecewisePolynomial([ys[0], ys[-1]], [[-yc, 1]]))
    Q = ydAdy.integrate().mul(-1)
    inv_b = PiecewisePolynomial(ys, inv_b_pieces)
    #plt.plot(*inv_b.get_plot_points()); plt.show()
    Q_Ib = Q.polymul(inv_b).mul(1/I)

    return Q_Ib


def calc_buckling_moment(pieces, yc, I):
    """Maximum allowed bending moment so no buckling occurs"""
    # bending moment
    max_bm = float('inf')
    for (p1, p2, count) in pieces:
        x1, y1, x2, y2 = p1[0], p1[1]-yc, p2[0], p2[1]-yc
        if y2 < y1:
            x1, x2, y1, y2 = x2, x1, y2, y1
        if not y2 > 0:
            continue
        k = 4.0
        if y1 < 0:
            t = -y1 / (y2-y1)
            x1 += (x2-x1) * t
            y1 += (y2-y1) * t + 1e-12
            #k = 6.0
        k = 6.0 - 2.0 * (y1/y2)**0.5  # underestimate is better than overestimate
        t = count * LINDEN_M
        b = np.hypot(x2-x1, y2-y1)
        sigma_crit = k*np.pi**2*E/12.0/(1.0-MU**2)*(t/b)**2
        #print(t, b, k, sigma_crit, y2)
        max_bm = min(max_bm, sigma_crit * I / y2)
    return max_bm


def calc_buckling_shear(pieces, yc, Q_Ib):
    """The worst case when a==infty"""
    max_shear = float('inf')
    for (p1, p2, count) in pieces:
        (x1, y1), (x2, y2) = p1, p2
        if y2 < y1:
            x1, x2, y1, y2 = x2, x1, y2, y1
        if y1 == y2:
            continue
        t = count * LINDEN_M
        h = y2 - y1
        tau_crit = 5.0*np.pi**2*E/12.0/(1.0-MU**2) * ((t/h)**2+(t/LENGTH)**2)
        max_shear = min(max_shear, tau_crit/Q_Ib)
    return max_shear


def calc_geometry(parts):
    """Returns (perimeter, (xc, yc), (Ix, Iy))
        Assumes the I matrix is diagonal"""
    peri_m, peri_c = 0.0, 0.0
    sA, sxA, syA, sx2A, sy2A = [0.0]*5
    for part in parts:
        for i in range(len(part)-1):
            p1, p2 = part[i], part[i+1]
            dl = np.linalg.norm(p2-p1)
            dA = dl * LINDEN_M
            peri_m += dl
            sA += dA
            sxA += 0.5*(p1[0]+p2[0]) * dA
            syA += 0.5*(p1[1]+p2[1]) * dA
            sx2A += (p1[0]**2+p1[0]*p2[0]+p2[0]**2)/3 * dA
            sy2A += (p1[1]**2+p1[1]*p2[1]+p2[1]**2)/3 * dA
    xc = sxA / sA
    yc = syA / sA  # centroidal axis
    Ix = sx2A - sA*xc**2
    Iy = sy2A - sA*yc**2  # second moment of area
    assert Ix > 0 and Iy > 0  # or there is a bug (and we also don't want ==0)
    return (peri_m, (xc, yc), (Ix, Iy))


def overlap_pieces(pieces0):
    """Divide/merge overlapping pieces"""
    pieces0.sort(key=lambda ps: np.linalg.norm(ps[1]-ps[0]))
    colinears = []  # n, d, pieces
    ok = lambda n, d, p: abs(n[0]*p[0]+n[1]*p[1]-d) < 1e-6
    for (p1, p2) in pieces0:
        n = (-(p2[1]-p1[1]), p2[0]-p1[0])
        n = (n[0]/np.hypot(n[0],n[1]), n[1]/np.hypot(n[0],n[1]))
        d = n[0]*p1[0] + n[1]*p1[1]
        colinears.append([n, d, [[p1, p2]]])
        for i in range(len(colinears)-1):
            n, d, pieces = colinears[i]
            if ok(n, d, p1) and ok(n, d, p2):
                pieces.append([p1, p2])
                del colinears[-1]
                break
    pieces = []  # p1, p2, count
    for (n, d, pairs) in colinears:
        comp = lambda p: n[0]*p[1]-n[1]*p[0]
        ps = []
        for (p1, p2) in pairs:
            if comp(p1) > comp(p2):
                p1, p2 = p2, p1
            ps += [(p1, 1), (p2, -1)]
        ps.sort(key=lambda p: comp(p[0]))
        count = 0
        for ((p1, d1), (p2, d2)) in zip(ps[:-1], ps[1:]):
            count += d1
            assert count >= 0
            if count == 0 or comp(p2)-comp(p1) < 1e-6:
                continue
            pieces.append((p1, p2, count))
    return pieces


def intersection_pieces(pieces1, pieces2):
    """Find the parts shared by both pieces"""
    pieces1 = sum([list(zip(p[:-1], p[1:])) for p in pieces1], [])
    pieces1.sort(key=lambda ps: np.linalg.norm(ps[1]-ps[0]))
    pieces2 = sum([list(zip(p[:-1], p[1:])) for p in pieces2], [])
    pieces2.sort(key=lambda ps: np.linalg.norm(ps[1]-ps[0]))
    # get colinear lines
    colinears = []  # n, d, pieces
    ok = lambda n, d, p: abs(n[0]*p[0]+n[1]*p[1]-d) < 1e-6
    def add_pieces(pieces, pid):
        for (p1, p2) in pieces:
            n = (-(p2[1]-p1[1]), p2[0]-p1[0])
            n = (n[0]/np.hypot(n[0],n[1]), n[1]/np.hypot(n[0],n[1]))
            d = n[0]*p1[0] + n[1]*p1[1]
            colinears.append([n, d, [[p1, p2, pid]]])
            for i in range(len(colinears)-1):
                n, d, pieces = colinears[i]
                if ok(n, d, p1) and ok(n, d, p2):
                    pieces.append([p1, p2, pid])
                    del colinears[-1]
                    break
    add_pieces(pieces1, 0)
    add_pieces(pieces2, 1)
    # check each straight line
    pieces = []  # p1, p2
    for (n, d, pairs) in colinears:
        comp = lambda p: n[0]*p[1]-n[1]*p[0]
        ps = []  # p, increment, pid
        for (p1, p2, pid) in pairs:
            if comp(p1) > comp(p2):
                p1, p2 = p2, p1
            ps += [(p1, 1, pid), (p2, -1, pid)]
        ps.sort(key=lambda p: (comp(p[0]), p[2]))
        count = [0, 0]
        for ((p1, d1, pid1), (p2, d2, pid2)) in zip(ps[:-1], ps[1:]):
            count[pid1] += d1
            assert min(count) >= 0
            #assert comp(p2) >= comp(p1)
            if comp(p2)-comp(p1) < 1e-6:
                continue
            if min(count) > 0:
                pieces.append((p1, p2))
        assert count == [0, 1] or count == [1, 0]
    return pieces


def cross_section_range(parts, bound_only=False):
    """Calculates the range of the cross section, for packing rectangles
        @bound_only is True: returns (width, height)
        @bound_only is False: returns list[(width, height)] for individual pieces"""
    res = []
    minx, miny, maxx, maxy = np.inf, np.inf, -np.inf, -np.inf
    for part in parts:
        x = [p[0] for p in part]
        y = [p[1] for p in part]
        if bound_only:
            minx = min(minx, min(x))
            miny = min(miny, min(y))
            maxx = max(maxx, max(x))
            maxy = max(maxy, max(y))
        else:
            res.append((max(x)-min(x), max(y)-min(y)))
    if bound_only:
        return (maxx-minx, maxy-miny)
    return res


def analyze_cross_section(parts_raw, parts, plot=False, return_full=False):
    """
        @parts: list of continuous point lists,
                first point equals last point for a closed polygon
        @parts_raw: parts without considering offset due to matboard thickness
                    used in determining glue joints
    """
    # represent points using NumPy arrays
    parts = [[np.array(p) for p in part] for part in parts]

    # calculate geometric properties
    peri_m, (xc, yc), (Ix, I) = calc_geometry(parts)
    I_buckle = I
    if Ix < I:  # buckling sideways?
        xr = max([r[0] for r in cross_section_range(parts)])
        yd = max([p[1] for p in sum(parts, [])])
        xd = max(0.5*xr-0.5*TRAIN_WIDTH, 0)  # how far the train can go sideways
        xd = 0.5*xr  # underestimate is better than overestimate
        c, s = yd/np.hypot(xd, yd), xd/np.hypot(xd, yd)
        I_buckle = c*c*I + s*s*Ix  # hope the inertia tensor formula still applies here

    # glue the pieces together
    pieces = []
    for part in parts_raw:
        for (p1, p2) in zip(part[:-1], part[1:]):
            pieces.append([p1, p2])
    pieces = overlap_pieces(pieces)
    # doesn't work with "cut" pieces, but better than nothing
    for part, part_raw in zip(parts, parts_raw):
        for (p01, p02, p1, p2) in zip(part_raw[:-1], part_raw[1:],
                                      part[:-1], part[1:]):
            for i in range(len(pieces)):
                q1, q2 = pieces[i][:2]
                same = lambda p1, p2: np.linalg.norm(p2-p1)<1e-6
                if (same(q1, p01) and same(q2, p02)) or \
                   (same(q1, p02) and same(q2, p01)):
                    pieces[i] = (p1, p2, pieces[i][2])

    # maximum allowed bending moment and shear force
    max_bm = calc_max_bending_moment(parts, yc, I)
    max_bm_b = calc_buckling_moment(pieces, yc, I_buckle)
    sff = calc_shear_factor(parts, yc, I)
    #max_sf = 1.0 / sff.eval(yc)
    max_sf_y, Q_Ib = sff.get_optim(absolute=True)
    max_sf = SHEAR_M / Q_Ib
    max_sf_b = calc_buckling_shear(pieces, yc, Q_Ib)
    fos_bm = max_bm/BM_MAX
    fos_bm_b = max_bm_b/BM_MAX
    fos_shear = max_sf/SHEAR_MAX
    fos_shear_b = max_sf_b/SHEAR_MAX

    if plot:
        print("yc:", yc, "mm")
        print("A:", peri_m*LINDEN_M, "mm²")
        print("I:", I, "mm⁴")
        print("Q/Ib:", Q_Ib, "mm³")
        print("Max BM:", 0.001*max_bm, "N⋅m", "\tFoS =", fos_bm)
        print("Max buckle BM:", 0.001*max_bm_b, "N⋅m", "\tFoS =", fos_bm_b)
        print("Max shear:", max_sf, "N", "\tFoS =", fos_shear)
        print("Max buckle shear:", max_sf_b, "N", "\tFoS =", fos_shear_b)
        fig, (ax1, ax2) = plt.subplots(
            1, 2, gridspec_kw={'width_ratios': [3, 1]})
        for part in parts:
            part = np.array(part)
            ax1.plot(part[:, 0], part[:, 1], '-')
        ax1.plot(ax1.get_xlim(), [yc, yc], '--')
        ax1.set_aspect('equal')
        ax1.set_xlabel("(mm)")
        ax1.set_ylabel("(mm)")
        ys, sffs = sff.get_plot_points()
        ax2.plot(sffs, ys)
        ax2.plot(SHEAR_M/max_sf, max_sf_y, 'o')
        ax2.set_xlabel("maxV⁻¹ (N⁻¹)")
        plt.show()

    if return_full:
        return (fos_bm, fos_bm_b, fos_shear, fos_shear_b)
    penalty = -1e6*max(peri_m/MAX_PERI-1, 0)**2
    return min(fos_bm, fos_bm_b, fos_shear, fos_shear_b)+penalty
