# Plot SFD/BMD, print key points

import numpy as np
import matplotlib.pyplot as plt

from PiecewisePolynomial import PiecewisePolynomial


def format_float(x, s=3):
    """Slide-rule precision"""
    if abs(x) <= 1e-12:
        return '0'
    sigfig = s+1 if np.log10(abs(x))%1.0 < np.log10(2) else s
    return f"{{:.{sigfig}g}}".format(x)


def solve_beam(
        x1, x2,  # start/end of the beam
        p1, p2,  # supports
        point_loads,  # (x, f), upward -> positive
        uniform_loads,  # ((x1, x2), p)
        EI  # φ = M/EI
    ):
    assert x2 > x1
    assert x1 <= p1 < p2 <= x2
    for ((a, b), p) in uniform_loads:
        assert b > a

    # solve for reaction forces
    vert = 0  # total vertical external load
    m_p1 = 0  # total external moment about p1
    for (x, f) in point_loads:
        vert += f
        m_p1 += f * (x-p1)
    for ((a, b), p) in uniform_loads:
        f = (b-a) * p
        vert += f
        m_p1 += f * (0.5*(a+b)-p1)
    f2 = -m_p1 / (p2-p1)
    f1 = -vert - f2
    print("Reaction forces:", format_float(-f1), format_float(-f2))

    # key points
    # (x, point_load, uniform_load_delta)
    keypoints_dict = {
        p1: [f1, 0],
        p2: [f2, 0]
    }
    def add_keypoint(x, f, dp):
        if x not in keypoints_dict:
            keypoints_dict[x] = [0, 0]
        keypoints_dict[x][0] += f
        keypoints_dict[x][1] += dp
    for (x, f) in point_loads:
        add_keypoint(x, f, 0)
    for ((a, b), p) in uniform_loads:
        add_keypoint(a, 0, p)
        add_keypoint(b, 0, -p)
    keypoints = sorted(keypoints_dict.keys())
    for i in range(len(keypoints)):
        x = keypoints[i]
        keypoints[i] = (x, keypoints_dict[x][0], keypoints_dict[x][1])
    keypoints = [(x1, 0, 0)] + keypoints + [(x2, 0, 0)]

    # SFD: integrate key point forces
    # BMD: integrate SFD (while calculating bm1)
    cul = 0.0  # current uniform load
    cld = 0.0  # current load

    poly_keypoints = [x1]
    poly_pieces = []  # linear [b, m]

    x_prev = x1
    for (x, pl, dul) in keypoints[1:]:

        # update piecewise polynomial
        m = cul
        b = cld - cul * x_prev
        poly_keypoints.append(x)
        poly_pieces.append([b, m])

        # update loads
        cld += cul * (x-x_prev)
        cul += dul
        if pl != 0.0:
            cld += pl
        x_prev = x

    sfd = PiecewisePolynomial(poly_keypoints, poly_pieces)
    bmd = sfd.integrate()
    phi = bmd.mul(1.0/EI)
    slope = phi.integrate()
    deflection = slope.integrate()
    y1 = deflection.eval(p1)
    y2 = deflection.eval(p2)
    corr_m = (y2-y1)/(p2-p1)
    corr_b = y1 - corr_m*p1
    slope = slope.sub([corr_m])
    deflection = deflection.sub([corr_b, corr_m])

    # plot graph
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(7.5, 7.5))
    ax4.set_xlabel("position (mm)")
    ax1.set_ylabel("shear (N)")
    ax2.set_ylabel("bending (N⋅mm)")
    ax3.set_ylabel("tangent (rad)")
    ax4.set_ylabel("deflection (mm)")
    ax1.plot(*sfd.get_plot_points(), '-')
    #ax1.plot(*sfd.get_plot_key_points(), 'o')
    ax2.plot(*bmd.get_plot_points(), '-')
    #ax2.plot(*bmd.get_plot_key_points(), 'o')
    ax2.set_ylim(ax2.get_ylim()[::-1])
    ax3.plot(*slope.get_plot_points(), '-')
    #ax3.plot(*slope.get_plot_key_points(), 'o')
    ax4.plot(*deflection.get_plot_points(), '-')
    #ax4.plot(*deflection.get_plot_key_points(), 'o')
    plt.show()


if __name__ == "__main__":

    # length unit: mm
    # weight unit: N

    length = 1260
    board_mass = 0.75*9.81

    train_weight = 400
    train_joints = [52, 228, 392, 568, 732, 908]

    EI = 4000 * 10e6

    solve_beam(
        0, length,
        50, length-50,
        [[j+100, -train_weight/6] for j in train_joints],
        [((0, length), -board_mass/length)],
        EI
    )


