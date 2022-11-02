# Plot SFD/BMD, print key points

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad


def format_float(x, s=3):
    """Slide-rule precision"""
    if abs(x) <= 1e-12:
        return '0'
    sigfig = s+1 if np.log10(abs(x))%1.0 < np.log10(2) else s
    return f"{{:.{sigfig}g}}".format(x)


def solve_beam(
        x1, x2,  # start/end of the beam
        p1, p2,  # supports
        point_loads,  # (x, f), downward -> positive
        uniform_loads  # ((x1, x2), p)
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
    cbm = 0.0  # current bending moment
    bm1 = 0.0  # bending moment at x1
    xs = [x1]
    sfds = [cld]
    bmds = [bm1]
    bmdcxs = np.array([], dtype=np.float64)
    bmdcs = np.array([], dtype=np.float64)
    for (x, pl, dul) in keypoints[1:]:
        dx = x-xs[-1]
        # due to uniformly distributed load
        bm1 += cul * dx * (0.5*(x+xs[-1])-x1)
        bmdcdx = np.linspace(0, dx)  # generate a smooth curve for plotting
        bmdc = cbm + cld * bmdcdx - 0.5 * cul * bmdcdx**2
        bmdcxs = np.concatenate((bmdcxs, xs[-1]+bmdcdx))
        bmdcs = np.concatenate((bmdcs, bmdc))
        if cul != 0.0:
            dxopt = cld / cul
            if 0 < dxopt < dx:
                bmopt = bmds[-1] + cld * dxopt - 0.5 * cul * dxopt**2
                xs.append(xs[-1]+dxopt)
                sfds.append(cld-cul*dxopt)
                bmds.append(bmopt)
        cbm += cld * dx - 0.5 * cul * dx**2
        cld -= cul * dx
        cul += dul
        xs.append(x)
        sfds.append(cld)
        bmds.append(cbm)
        # due to point load
        if pl != 0.0:
            bm1 += pl * (x-x1)
            cld -= pl
            xs.append(x)
            sfds.append(cld)
            bmds.append(cbm)
    assert abs(bm1) < 1e-12

    # print info
    maxsf, maxsfx, maxbm, maxbmx = 0.0, [], 0.0, []
    print("x sf bm")
    for (x, sf, bm) in list(zip(xs, sfds, bmds))[1:len(xs)-1]:
        sf = float(format_float(sf))
        bm = float(format_float(bm))
        print(format_float(x), sf, bm)
        if abs(sf) > abs(maxsf):
            maxsf, maxsfx = sf, [x]
        elif abs(sf) == abs(maxsf):
            maxsfx.append(x)
        if abs(bm) > abs(maxbm):
            maxbm, maxbmx = bm, [x]
        elif abs(bm) == abs(maxbm):
            maxbmx.append(x)
    maxsfx = sorted(set(maxsfx))
    maxbmx = sorted(set(maxbmx))
    print("Maximum shearing stress:", format_float(maxsf),
          "at", 'x='+','.join([format_float(x) for x in maxsfx]))
    print("Maximum bending moment:", format_float(maxbm),
          "at", 'x='+','.join([format_float(x) for x in maxbmx]))

    # plot graph
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.set_title("SFD and BMD of the given beam")
    ax2.set_xlabel("position (m)")
    ax1.set_ylabel("shear force (kN)")
    ax2.set_ylabel("bending moment (kN⋅m)")
    ax1.plot(xs, sfds, 'o-')
    ax2.plot(bmdcxs, bmdcs, '-')
    ax2.plot(xs, bmds, 'o')
    ax2.set_ylim(ax2.get_ylim()[::-1])
    plt.show()


def assignment_7_problem_1():
    solve_beam(
        0, 30,
        0, 30,
        [(6, 100), (12, 100), (18, 100), (24, 100)],
        []
    )

def assignment_7_problem_3a():
    solve_beam(
        0, 6,
        0, 4,
        [(3, 10), (6, 10)],
        []
    )

def assignment_7_problem_4():
    solve_beam(
        0, 8,
        0, 8,
        [(3, 200), (5, 200)],
        [((0, 8), 60)]
    )

def assignment_7_problem_5():
    solve_beam(
        0, 7,
        1, 6,
        [],
        [((0, 7), 90)]
    )

def assignment_7_problem_6a():
    solve_beam(
        0, 9,
        2, 7,
        [(0, 20), (9, 30)],
        [((2, 7), 10)]
    )

def assignment_7_problem_6b():
    solve_beam(
        0, 9,
        #2, 7,  # copy-paste typo, graph looks cool
        0, 9,
        [(6, 20)],
        [((3, 9), 8)]
    )

def assignment_7_problem_7():
    solve_beam(
        0, 10,
        0, 10,
        [(2, 5), (4, 1), (6, 7), (8, 15)],
        []
    )

if __name__ == "__main__":

    #assignment_7_problem_1()
    #assignment_7_problem_3a()
    #assignment_7_problem_4()
    #assignment_7_problem_5()
    assignment_7_problem_6a()
    #assignment_7_problem_6b()
    #assignment_7_problem_7()
    
    
