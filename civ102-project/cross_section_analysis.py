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
LINDEN_C = 0.1  # thickness of contact cement, assume cement has the same density as matboard
SHEAR_C = 2.  # shear strength of contact cement

# calculated from beam analysis
REACTION_MAX = 273
SHEAR_MAX = 255
BM_MAX = 66400


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


def calc_shear_factor(parts, yc, I):
    """Maximum allowed shear force
            same parameters as `calc_max_bending_moment()`
            neglect glue
        Calculates Q(y) = -∫ydA as a PiecewisePolynomial;
        Calculates 1/b(y) as a PiecewisePolynomial;
        Returns the reciprocal of failure shear force
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
            inv_b_pieces.append([1/b, 0.0])
    dAdy = PiecewisePolynomial(ys, dAdy_pieces)
    ydAdy = dAdy.polymul(PiecewisePolynomial([ys[0], ys[-1]], [[-yc, 1]]))
    Q = ydAdy.integrate().mul(-1)
    inv_b = PiecewisePolynomial(ys, inv_b_pieces)
    Q_Ib = Q.polymul(inv_b).mul(1/I)

    return Q_Ib.mul(1/SHEAR_M)
    #return Q_Ib.mul(1/SHEAR_C)


def calc_buckling_moment(parts, yc, I):
    """Maximum allowed bending moment so no buckling occurs"""
    # glue the pieces together
    pieces0 = []
    for part in parts:
        for (p1, p2) in zip(part[:-1], part[1:]):
            pieces0.append([p1, p2])
    pieces0.sort(key=lambda ps: -np.linalg.norm(ps[1]-ps[0]))
    pieces = []
    for (p1, p2) in pieces0:
        n = [-(p2[1]-p1[1]), p2[0]-p1[0]]
        d = n[0]*p1[0] + n[1]*p1[1]
        ok = lambda p: abs(n[0]*p[0]+n[1]*p[1]-d) < 1e-6
        pieces.append([p1, p2, 1])
        for i in range(len(pieces)-1):
            q1, q2, count = pieces[i]
            if ok(q1) and ok(q2):
                pieces[i][2] += 1
                del pieces[-1]
                break
    # bending moment
    max_bm = float('inf')
    for (p1, p2, count) in pieces:
        x1, y1, x2, y2 = p1[0], p1[1]-yc, p2[0], p2[1]-yc
        if y2 < y1:
            x1, x2, y1, y2 = x2, x1, y2, y1
        if not y2 > 0:
            continue
        # formula likely not right
        # underestimate is better than overestimate
        k = 4.0
        if y1 < 0:
            t = -y1 / (y2-y1)
            x1 += (x2-x1) * t
            y1 += (y2-y1) * t + 1e-12
            #k = 6.0
        k = 6.0 - 2.0 * (y1/y2)**0.5  # ??
        t = count * LINDEN_M
        b = np.hypot(x2-x1, y2-y1)
        sigma_crit = k*np.pi**2*E/12.0/(1.0-MU**2)*(t/b)**2
        max_bm = min(max_bm, sigma_crit * I / y2)
    return max_bm


def calc_geometry(parts, glues):
    """Returns (perimeter, yc, I)"""
    peri_m, peri_c = 0.0, 0.0
    sA, syA, sy2A = 0.0, 0.0, 0.0
    for part in parts:
        for i in range(len(part)-1):
            p1, p2 = part[i], part[i+1]
            dl = np.linalg.norm(p2-p1)
            dA = dl * LINDEN_M
            peri_m += dl
            sA += dA
            syA += 0.5*(p1[1]+p2[1]) * dA
            sy2A += (p1[1]**2+p1[1]*p2[1]+p2[1]**2)/3 * dA
    for glue in glues:
        p1, p2 = glue
        dl = np.linalg.norm(p2-p1)
        dA = dl * LINDEN_C
        peri_c += dl
        sA += dA
        syA += 0.5*(p1[1]+p2[1]) * dA
        sy2A += (p1[1]**2+p1[1]*p2[1]+p2[1]**2)/3 * dA
    yc = syA / sA  # centroidal axis
    I = sy2A - sA*yc**2  # second moment of area
    assert I > 0  # or there is a bug
    return (peri_m, yc, I)


def analyze_cross_section(parts, glues, plot=False, return_full=False):
    """
        @parts: list of continuous point lists,
                first point equals last point for a closed polygon
        @glues: list of glue joints, each has a list of two points

    Currently returns the maximum allowed bending moment
    """
    # represent points using NumPy arrays
    parts = [[np.array(p) for p in part] for part in parts]
    glues = [[np.array(p) for p in glue] for glue in glues]

    # calculate geometric properties
    peri_m, yc, I = calc_geometry(parts, glues)

    # maximum allowed bending moment and shear force
    max_bm = calc_max_bending_moment(parts, yc, I)
    max_bm_b = calc_buckling_moment(parts, yc, I)
    sff = calc_shear_factor(parts, yc, I)
    #max_sf = 1.0 / sff.eval(yc)
    max_sf_y, max_sf = sff.get_optim(absolute=True)
    max_sf **= -1.0
    fos_bm = max_bm/BM_MAX
    fos_bm_b = max_bm_b/BM_MAX
    fos_shear = max_sf/SHEAR_MAX

    if plot:
        print("I:", I, "mm⁴")
        print("Max BM:", 0.001*max_bm, "N⋅m", "\tFoS =", fos_bm)
        print("Max buckle BM:", 0.001*max_bm_b, "N⋅m", "\tFoS =", fos_bm_b)
        print("Max shear:", max_sf, "N", "\tFoS =", fos_shear)
        fig, (ax1, ax2) = plt.subplots(
            1, 2, gridspec_kw={'width_ratios': [3, 1]})
        for part in parts:
            part = np.array(part)
            ax1.plot(part[:, 0], part[:, 1], '-')
        for glue in glues:
            glue = np.array(glue)
            ax1.plot(glue[:, 0], glue[:, 1], 'k--')
        ax1.plot(ax1.get_xlim(), [yc, yc], '--')
        ax1.set_aspect('equal')
        ax1.set_xlabel("(mm)")
        ax1.set_ylabel("(mm)")
        ys, sffs = sff.get_plot_points()
        ax2.plot(sffs, ys)
        ax2.plot(1.0/max_sf, max_sf_y, 'o')
        ax2.set_xlabel("maxV⁻¹ (N⁻¹)")
        plt.show()

    if return_full:
        return (fos_bm, fos_bm_b, fos_shear)
    penalty = -1e6*max(peri_m/MAX_PERI-1, 0)**2
    return min(fos_bm, fos_bm_b, fos_shear)+penalty
