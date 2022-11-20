"""DEPRECATED: This code is no longer compactible
    after changes on `cross_section_analysis.py`.
"""


from cross_section_analysis import *


def analyze_trapezoid(wt, wb, h, glue_loc, no_triangle=True, plot=False):
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
    if no_triangle:  # not analyzing a "triangle"
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


def optimize_trapezoid(glue_loc, no_triangle=True):
    """Find values for wt, wb, h to maximize allowed bending moment

    Perimeter: wt + wb + 2 hypot(h, (wt-wb)/2)
    search wt, wb, h from 0 to MAX_PERI/2
    """
    assert glue_loc in ['top', 'bottom']
    objfun = lambda params: -analyze_trapezoid(*params, glue_loc, no_triangle)

    # bruteforce grid search
    optimx = scipy.optimize.brute(
        objfun,
        ([50, MAX_PERI/2], [50, MAX_PERI/2], [10, MAX_PERI/2]),
        Ns=12)
    optimv = -objfun(optimx)
    print("Brute grid search:", optimx, optimv)

    # optimize
    optres = scipy.optimize.minimize(
        objfun, method='Nelder-Mead', x0=optimx)
    optimx, optimv = optres.x, -optres.fun
    print("Optimize:", optimx, optimv)

    analyze_trapezoid(*optimx, glue_loc, no_triangle, plot=True)


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
        Ns=8)
    optimv = -objfun(optimx)
    print("Brute grid search:", optimx, optimv)

    # optimize
    optres = scipy.optimize.minimize(
        objfun, method='Nelder-Mead', x0=optimx)
    optimx, optimv = optres.x, -optres.fun
    print("Optimize:", optimx, optimv)

    analyze_pi(*optimx, plot=True)


def analyze_trapezoid_triangle(wt, wb, h, trig_loc, no_triangle=True, plot=False):
    """Trapezoid with triangle strengthening"""
    assert trig_loc in ['bottom', 'side']

    # hard constraint(s)
    if not max(wt, wb, h) > 0:
        return -1  # must be a valid geometry
    if wt < 75:
        return -1  # must hold the train
    if no_triangle:
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


def optimize_trapezoid_triangle(trig_loc, no_triangle=True):
    """Find values for wt, wb, h to maximize allowed bending moment
    """
    assert trig_loc in ['bottom', 'side']
    objfun = lambda params: -analyze_trapezoid_triangle(*params, trig_loc, no_triangle)

    # bruteforce grid search
    optimx = scipy.optimize.brute(
        objfun,
        ([50, MAX_PERI/2], [50, MAX_PERI/2], [10, MAX_PERI/2]),
        Ns=10)
    optimv = -objfun(optimx)
    print("Brute grid search:", optimx, optimv)

    # optimize
    optres = scipy.optimize.minimize(
        objfun, method='Nelder-Mead', x0=optimx)
    #print(optres)
    optimx, optimv = optres.x, -optres.fun
    print("Optimize:", optimx, optimv)

    analyze_trapezoid_triangle(*optimx, trig_loc, no_triangle, plot=True)


def analyze_semicircle(wt, l, glue_loc, plot=False):
    """Flat with length @wt at the top and circular arc with length @l below"""
    assert glue_loc in ['top', 'bottom']
    # hard constraint(s)
    if not max(wt, l) > 0 or l <= wt:
        return -1  # must be a valid geometry
    if wt < 75:
        return -1  # must hold the train
    # parameters
    r = scipy.optimize.brentq(
        lambda r: 2*r*np.sin(l/(2*r))-wt,
        0.15*l, max(0.2, l/(l-wt))*l)
    theta = 0.5*l/r
    # construct
    points = []
    sign = 1.0 if glue_loc == 'bottom' else -1.0
    for a in np.linspace(-theta, theta, 64):
        points.append([r*np.sin(a), sign*r*np.cos(a)])
    points = [points[-1]] + points + [points[0]]
    glues = [points[0], points[-1]]
    # analyze
    return analyze_cross_section([points], [glues], plot)


def optimize_semicircle(glue_loc):
    assert glue_loc in ['top', 'bottom']
    objfun = lambda params: -analyze_semicircle(*params, glue_loc)

    # bruteforce grid search
    optimx = scipy.optimize.brute(
        objfun,
        ([50, MAX_PERI/2], [50, MAX_PERI/2]),
        Ns=12)
    optimv = -objfun(optimx)
    print("Brute grid search:", optimx, optimv)

    # optimize
    optres = scipy.optimize.minimize(
        objfun, method='Nelder-Mead', x0=optimx)
    optimx, optimv = optres.x, -optres.fun
    print("Optimize:", optimx, optimv)

    analyze_semicircle(*optimx, glue_loc, plot=True)


def analyze_curve(wt, points, plot=False):
    if not wt > 0:
        return -1
    glue = [[0.5*wt, 0], [-0.5*wt, 0]]
    points = glue + points + glue
    return analyze_cross_section([points], [glue], plot)


def optimize_curve(N):
    """This optimization fails
        Result is very close to a truncated circle"""
    assert N > 0
    
    def decode_params(params):
        wt = params[0]
        points = params[1:].reshape(N, 2).tolist()
        return wt, points

    def objfun(params):
        wt, points = decode_params(params)
        return -analyze_curve(wt, points)

    r, theta = 100.0, 0.8*np.pi
    a = np.linspace(-theta, theta, N+2)[1:N+1]
    points = np.array([r*np.sin(a), -r*np.cos(a)+r*np.cos(theta)]).T
    params = np.concatenate(([2.0*r*np.sin(theta)], points.flatten()))
    print("Initial:", params)
    optres = scipy.optimize.minimize(
        objfun, x0=params)
    print(optres.message)
    optimx, optimv = optres.x, -optres.fun
    print("Optimize:", optimx, optimv)
    analyze_curve(*decode_params(params), plot=True)


if __name__ == "__main__":
    #optimize_trapezoid(glue_loc='top')
    #optimize_trapezoid(glue_loc='top', no_triangle=False)
    #optimize_trapezoid(glue_loc='bottom')
    #optimize_pi()
    #optimize_trapezoid_triangle(trig_loc='bottom', no_triangle=True)
    #optimize_trapezoid_triangle(trig_loc='side', no_triangle=True)
    optimize_semicircle(glue_loc='top')
    #optimize_curve(10)


