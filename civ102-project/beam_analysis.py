# Plot shear/bending/tangent/deflection of a beam
# Print extreme values

import numpy as np
import matplotlib.pyplot as plt

from PiecewisePolynomial import PiecewisePolynomial


# strictly follow the instruction
# do not include self weight and deflection
# use 1mm increment
STRICT_MODE = (__name__ == "__main__")
#STRICT_MODE = False


LENGTH = 1260.  # length of the beam


def solve_beam(
        x1, x2,  # start/end of the beam
        p1, p2,  # supports
        point_loads,  # (x, f), upward -> positive
        uniform_loads,  # ((x1, x2), p)
        EI,  # φ = M/EI
        plot=False
    ):
    """Returns four tuples (max_argx, max_val)
        - reaction force (N) (max_x==0)
        - shear force (N)
        - bending momemnt (N⋅mm)
        - deflection (mm)
    """
    assert x2 > x1
    assert x1 <= p1 < p2 <= x2
    for ((a, b), p) in uniform_loads:
        assert b > a

    # solve for reaction forces
    vert = 0  # total vertical external load
    m_p1 = 0  # total external moment about p1
    for (x, f) in point_loads:
        if x < x1 or x > x2:
            continue
        vert += f
        m_p1 += f * (x-p1)
    for ((a, b), p) in uniform_loads:
        assert x1 <= a < b <= x2
        f = (b-a) * p
        vert += f
        m_p1 += f * (0.5*(a+b)-p1)
    f2 = -m_p1 / (p2-p1)
    f1 = -vert - f2
    #print("Reaction forces:", f1, f2)

    # key points
    # (x, point_load, uniform_load_delta)
    keypoints_dict = {
        p1: [f1, 0],
        p2: [f2, 0]
    }
    def add_keypoint(x, f, dp):
        if x < x1 or x > x2:
            return
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

    # integrate to get sfd, bmd, slope, delta
    sfd = PiecewisePolynomial(poly_keypoints, poly_pieces)
    bmd = sfd.integrate()
    phi = bmd.mul(1.0/EI)
    slope = phi.integrate()
    delta = slope.integrate()
    # choose integral constants to make deflection at supports zero
    y1 = delta.eval(p1)
    y2 = delta.eval(p2)
    corr_m = (y2-y1)/(p2-p1)
    corr_b = y1 - corr_m*p1
    slope = slope.sub([corr_m])
    delta = delta.sub([corr_b, corr_m])

    # plot graph
    if plot:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(7.5, 7.5))
        ax4.set_xlabel("position (mm)")

        ax1.set_ylabel("shear (N)")
        ax1.plot(*sfd.get_plot_points(), '-')
        #ax1.plot(*sfd.get_keypoints(), 'o')
        maxsfd = sfd.get_optim(absolute=True)
        ax1.plot(maxsfd[0], maxsfd[1], 'o')
        print("Max shear", maxsfd[1])    

        ax2.set_ylabel("bending (N⋅mm)")
        ax2.plot(*bmd.get_plot_points(), '-')
        #ax2.plot(*bmd.get_keypoints(), 'o')
        maxbmd = bmd.get_optim()[1]
        print("Max bending", maxbmd)
        ax2.plot(maxbmd[0], maxbmd[1], 'o')
        ax2.set_ylim(ax2.get_ylim()[::-1])

        ax3.set_ylabel("tangent (rad)")
        ax3.plot(*slope.get_plot_points(), '-')
        #ax3.plot(*slope.get_keypoints(), 'o')

        ax4.set_ylabel("deflection (mm)")
        ax4.plot(*delta.get_plot_points(), '-')
        #ax4.plot(*delta.get_keypoints(), 'o')
        maxdelta = delta.get_optim()[0]
        print("Max deflection", maxdelta)
        ax4.plot(maxdelta[0], maxdelta[1], 'o')

        plt.show()

    xs = np.linspace(x1+1e-12, x2-1e-12, int(x2-x1)// \
                     (1 if STRICT_MODE else 10)+1)
    return (
        (np.array([p1, p2]), np.array([f1, f2])),
        (xs, abs(sfd.evals(xs))),
        (xs, bmd.evals(xs)),
        (xs, delta.evals(xs))
    )


def get_responses(train_x, plot=False):

    # length unit: mm
    # weight unit: N

    board_mass = 0.75*9.81

    train_weight = 400
    train_joints = [52, 228, 392, 568, 732, 908]

    EI = 4000 * 1e6

    if STRICT_MODE:
        return solve_beam(
            0, LENGTH,
            0.5*LENGTH-600, 0.5*LENGTH+600,
            [[j+train_x, -train_weight/6] for j in train_joints],
            [],
            EI, plot
        )

    return solve_beam(
        0, LENGTH,
        0.5*LENGTH-575, 0.5*LENGTH+575,
        [[j+train_x, -train_weight/6] for j in train_joints],
        #[((0, LENGTH), -board_mass/LENGTH)],
        [],
        EI,
        plot
    )


def plot_max_responses(plot=True):

    # generate a list of possible train left positions
    train_x1 = -960
    train_x2 = int(LENGTH)
    xs = np.array(range(train_x1, train_x2+1,
                        1 if STRICT_MODE else 2), dtype=np.float64)

    # find the maximum reaction across all positions
    msfx, mbmx = train_x1, train_x1
    for x in xs:
        (rx, rv), (sfx, sfv), (bmx, bmv), (dfx, dfv) = get_responses(x)
        if x == xs[0]:
            mrv, msfv, mbmv, mdfv = rv, sfv, bmv, dfv
        else:
            mrv = np.maximum(mrv, rv)
            msfv = np.maximum(msfv, sfv)
            mbmv = np.maximum(mbmv, bmv)
            mdfv = np.minimum(mdfv, dfv)
            if plot:
                if np.amax(msfv) == np.amax(sfv):
                    msfx = x
                if np.amax(mbmv) == np.amax(bmv):
                    mbmx = x

    if not plot:
        return np.amax(mrv), (sfx, msfv), (bmx, mbmv)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7))

    print("Max reaction", np.amax(mrv), "N")
    print("Max shear", np.amax(msfv), "N",
          "at train_x =", msfx,
          "at x =", sfx[np.argmax(msfv)])
    print("Max bending", np.amax(0.001*mbmv), "N⋅m",
          "at train_x =", mbmx,
          "at x =", bmx[np.argmax(mbmv)])

    ax1.set_title("max shear (N)")
    ax1.plot(sfx, msfv, '-')

    ax2.set_title("max bending (×10³ N⋅mm)")
    ax2.plot(bmx, 0.001*mbmv, '-')

    plt.show()


if __name__ == "__main__":

    #get_responses(-123, True)
    plot_max_responses()
