import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

from PiecewisePolynomial import PiecewisePolynomial

# matboard properties
# length: mm; weight: N; stress: MPa
LINDEN_M = 1.27  # thickness of the matboard
SIGMA_T = 30.  # tensile strength of the matboard
SIGMA_C = 6.  # compressive strength of the matboard
SHEAR_M = 4.  # shear strength of the matboard
MAX_PERI = 800.  # maximum allowed perimeter
E = 4000.  # Young's modulus of the matboard
LINDEN_C = 0.1  # thickness of contact cement, assume cement has the same density as matboard
SHAER_C = 2.  # shear strength of contact cement


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
        Returns τ(y)/V = Q(y)/(I*b(y))
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

    # get ydA and 1/b
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

    #return Q_Ib.mul(1/SHEAR_M)
    return Q_Ib.mul(1/SHAER_C)


def analyze_cross_section(parts, glues, plot=False):
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

    # maximum allowed bending moment and shear force
    max_bm = calc_max_bending_moment(parts, yc, I)
    sff = calc_shear_factor(parts, yc, I)
    max_sf = 1.0 / sff.eval(yc)

    if plot:
        print("I:", I, "mm⁴")
        print("Max BM:", 0.001*max_bm, "N⋅m")
        print("Max shear:", max_sf, "N")
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
        ax2.plot(1.0/max_sf, yc, 'o')
        ax2.set_xlabel("maxV⁻¹ (N⁻¹)")
        plt.show()

    if peri_m > MAX_PERI:
        return -1.0  # impossible

    return max_bm


def analyze_trapezoid(wt, wb, h, glue_loc, plot=False):
    """Equilateral trapezoid
        @wt: width at the top
        @wb: width at the bottom
        @h: height
    Currently returns the maximum bending moment
    """
    assert glue_loc in ['top', 'bottom']

    # hard constraint(s)
    if not max(wt, wb, h) > 0:
        return -1  # must be a valid geometry
    if wt < 75:
        return -1  # must hold the train
    if True:  # not analyzing a "triangle"
        if wb < 50 or wt > 2 * wb:
            return -1  # must stand on the platform
        if h > 2 * wb:
            return -1  # must not be too "thin"

    # analyze
    if glue_loc == 'bottom':
        points = [
            (-0.5*wb, 0),
            (0.5*wb, 0),
            (0.5*wt, h),
            (-0.5*wt, h),
            (-0.5*wb, 0),
            (0.5*wb, 0)
        ]
        glues = [
            (-0.5*wb, 0),
            (0.5*wb, 0)
        ]
    elif glue_loc == 'top':
        points = [
            (0.5*wt, h),
            (-0.5*wt, h),
            (-0.5*wb, 0),
            (0.5*wb, 0),
            (0.5*wt, h),
            (-0.5*wt, h)
        ]
        glues = [
            (-0.5*wt, h),
            (0.5*wt, h)
        ]
    return analyze_cross_section([points], [glues], plot)


def optimize_trapezoid(glue_loc):
    """Find values for wt, wb, h to maximize allowed bending moment

    Perimeter: wt + wb + 2 hypot(h, (wt-wb)/2)
    search wt, wb, h from 0 to MAX_PERI/2
    """
    assert glue_loc in ['top', 'bottom']
    objfun = lambda params: -analyze_trapezoid(*params, glue_loc)

    # bruteforce grid search
    optimx = scipy.optimize.brute(
        objfun,
        ([50, MAX_PERI/2], [50, MAX_PERI/2], [10, MAX_PERI/2]),
        Ns=20)
    optimv = -objfun(optimx)
    print("Brute grid search:", optimx, optimv)

    # optimize
    optres = scipy.optimize.minimize(
        objfun, method='Nelder-Mead', x0=optimx)
    optimx, optimv = optres.x, -optres.fun
    print("Optimize:", optimx, optimv)

    analyze_trapezoid(*optimx, glue_loc, plot=True)


def analyze_pi(wt, ht, wb, hb, xb, plot=False):
    """Pi-shaped, three squares
        @wt: width of the top piece
        @ht: height of the top piece
        @wb: width of each of the bottom piece
        @hb: height of each of the bottom piece
        @xb: deviation of the left of the bottom piece from the middle
    Currently returns the maximum bending moment
    """

    # hard constraint(s)
    if not max(wt, ht, wb, hb, xb) > 0:
        return -1  # must be a valid geometry
    if xb + wb > 0.5*wt:
        return -1  # bottom must hold top
    if 2*(xb+wb) < 50:
        return -1  # must stand on the platform

    # analyze
    top_piece = [
        (0.5*wt, hb+ht),
        (-0.5*wt, hb+ht),
        (-0.5*wt, hb),
        (0.5*wt, hb),
        (0.5*wt, hb+ht), # glue at top based on trapezoid analysis result
        (-0.5*wt, hb+ht)
    ]
    bottom_piece_1 = [
        (xb, 0),
        (xb+wb, 0),
        (xb+wb, hb),
        (xb, hb),
        (xb, 0),  # glue at bottom to resist tensile stress
        (xb+wb, 0)
    ]
    bottom_piece_2 = np.array(bottom_piece_1) * (-1, 1)
    glues = [
        [top_piece[4], top_piece[5]],
        [bottom_piece_1[4], bottom_piece_1[5]],
        [bottom_piece_2[4], bottom_piece_2[5]],
        [bottom_piece_1[2], bottom_piece_1[3]],
        [bottom_piece_2[2], bottom_piece_2[3]]
    ]
    return analyze_cross_section(
        [top_piece, bottom_piece_1, bottom_piece_2],
        glues, plot)


def optimize_pi():
    """Find values for wt, ht, wb, hb, xb to maximize allowed bending moment
    """
    objfun = lambda params: -analyze_pi(*params)

    # bruteforce grid search
    optimx = scipy.optimize.brute(
        objfun,
        ([50, MAX_PERI/2], [10, MAX_PERI/2],
         [20, MAX_PERI/2], [10, MAX_PERI/2], [0, MAX_PERI/2]),
        Ns=10)
    optimv = -objfun(optimx)
    print("Brute grid search:", optimx, optimv)

    # optimize
    optres = scipy.optimize.minimize(
        objfun, method='Nelder-Mead', x0=optimx)
    optimx, optimv = optres.x, -optres.fun
    print("Optimize:", optimx, optimv)

    analyze_pi(*optimx, plot=True)


def analyze_trapezoid_triangle(wt, wb, h, trig_loc, plot=False):
    """Trapezoid with triangle strengthening"""
    assert trig_loc in ['bottom', 'side']

    # hard constraint(s)
    if not max(wt, wb, h) > 0:
        return -1  # must be a valid geometry
    if wt < 75:
        return -1  # must hold the train
    if wb < 50 or wt > 2 * wb:
        return -1  # must stand on the platform

    # analyze
    points = [[
        (0.5*wt, h),
        (-0.5*wt, h),
        (-0.5*wb, 0),
        (0.5*wb, 0),
        (0.5*wt, h),
        (-0.5*wt, h)
    ]]
    glues = [[
        (-0.5*wt, h),
        (0.5*wt, h)
    ]]
    if trig_loc == 'bottom':
        points.append([
            (-0.5*wt, h),
            (0, h),
            (-0.5*wb, 0),
            (0.5*wb, 0),
            (0, h),
            (0.5*wt, h)
        ])
        glues += [
            [(-0.5*wt, h), (0, h)],
            [(0, h), (0.5*wt, h)],
            [(-0.5*wb, 0), (0.5*wb, 0)]
        ]
    if trig_loc == 'side':
        points.append([
            (-0.5*wt, h),
            (0, h),
            (-0.5*wb, 0),
            (-0.5*wt, h)
        ])
        points.append([
            (0.5*wt, h),
            (0, h),
            (0.5*wb, 0),
            (0.5*wt, h)
        ])
        glues += [
            [(-0.5*wt, h), (0, h)],
            [(-0.5*wb, 0), (-0.5*wt, h)],
            [(0.5*wt, h), (0, h)],
            [(0.5*wb, 0), (0.5*wt, h)]
        ]
    return analyze_cross_section(points, glues, plot)


def optimize_trapezoid_triangle(trig_loc):
    """Find values for wt, wb, h to maximize allowed bending moment
    """
    assert trig_loc in ['bottom', 'side']
    objfun = lambda params: -analyze_trapezoid_triangle(*params, trig_loc)

    # bruteforce grid search
    optimx = scipy.optimize.brute(
        objfun,
        ([50, MAX_PERI/2], [50, MAX_PERI/2], [10, MAX_PERI/2]),
        Ns=20)
    optimv = -objfun(optimx)
    print("Brute grid search:", optimx, optimv)

    # optimize
    optres = scipy.optimize.minimize(
        objfun, method='Nelder-Mead', x0=optimx)
    optimx, optimv = optres.x, -optres.fun
    print("Optimize:", optimx, optimv)

    analyze_trapezoid_triangle(*optimx, trig_loc, plot=True)

if __name__ == "__main__":
    #optimize_trapezoid(glue_loc='top')
    #optimize_trapezoid(glue_loc='bottom')
    #optimize_pi()
    optimize_trapezoid_triangle(trig_loc='bottom')
    #optimize_trapezoid_triangle(trig_loc='side')

