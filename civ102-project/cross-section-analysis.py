import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

# matboard properties
# length: mm; weight: N
LINDEN_M = 1.27  # thickness of the matboard
SIGMA_T = 30.  # tensile strength of the matboard
SIGMA_C = 6.  # compressive strength of the matboard
SHEAR_M = 4.  # shear strength of the matboard
MAX_PERI = 800.  # maximum allowed perimeter
E = 4000.  # Young's modulus of the matboard
LINDEN_C = 0.5  # thickness of contact cement, mass ratio to matboard
SHAER_C = 2.  # shear strength of contact cement

LINDEN_M = 2.5


def calc_max_bending_moment(parts, yc, I):
    """
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

    if plot:
        plt.clf()
        for part in parts:
            part = np.array(part)
            plt.plot(part[:, 0], part[:, 1], '-')
        for glue in glues:
            glue = np.array(glue)
            plt.plot(glue[:, 0], glue[:, 1], 'k--')
        plt.plot(plt.xlim(), [yc, yc], '--')
        plt.show()

    if peri_m > MAX_PERI:
        return -1.0  # impossible

    max_bm = calc_max_bending_moment(parts, yc, I)
    #print("Maximum bending moment:", 0.001*max_bm, "Nm")
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
    if wb < 50 or wt > 2 * wb:
        return -1  # must stand on the platform

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

