# Bridge Design Source Files listed in Chronological Order


## `PiecewisePolynomial.py` and `beam_analysis.py`

Calculates the maximum reaction force, maximum shear, and maximum bending moment as a function of location across the bridge. Prints the values and plots the graphs.

This file is based on a passion experiment of integrating piecewise polynomials that was written before the project was introduced. It also calculates a deflection diagram assuming an uniform $EI$.

In the first checkin: We considered the self-weight of the bridge; We used the edge (instead of middle) of the support for the location of the reaction force (reason: think about deflection). My group got mark deducted because the TA got different numbers and we were frustrated.

Change made after the incident: add a global parameter `STRICT_MODE`. Strictly follow the instruction if it is `True` to make TAs happy in the report, and use the more practically accurate model if it is `False`.

```py
import numpy as np


class PiecewisePolynomial():

    def __init__(self, keypoints, pieces):
        """
            @keypoints: a list of key points
            @linear_pieces: a list of [b, m] between keypoints
            len(keypoints) should be len(linear_pieces)+1
            All lists must be Python list, not NumPy
        """
        self.num_pieces = len(keypoints)-1
        self.keypoints = keypoints[:]
        self.pieces = pieces[:]

    def __len__(self):
        return self.num_pieces

    # evaluation

    @staticmethod
    def _eval_piece(piece, x):
        """evaluate one polynomial piece"""
        y = 0
        p = 1
        for c in piece:
            y += c * p
            p *= x
        return y

    def eval(self, x, leq=True):
        """evaluate the function at a given x"""
        l = len(self)
        for i in range(l):
            x1, x2 = self.keypoints[i], self.keypoints[i+1]
            if ((leq or i == 0) and x1<=x<x2) or \
                ((not leq or i == l-1) and x1<x<=x2):
                return self._eval_piece(self.pieces[i], x)
        return None

    def evals(self, xs, leq=True):
        return np.array([self.eval(x, leq) for x in xs])

    def get_plot_points(self, num_splits = 1000):
        """get a list of keypoints for plotting"""
        delta = (self.keypoints[-1] - self.keypoints[0]) / num_splits
        xs, ys = [], []
        for i in range(len(self)):
            x1, x2 = self.keypoints[i], self.keypoints[i+1]
            n_dif = int(np.ceil((x2-x1)/delta))
            x = np.linspace(x1, x2, n_dif+1)
            y = self._eval_piece(self.pieces[i], x)
            if len(ys) != 0 and abs(y[0]-ys[-1]) < 1e-12:
                x, y = x[1:], y[1:]
            xs = np.concatenate((xs, x))
            ys = np.concatenate((ys, y))
        return xs, ys

    # integration

    def integrate(self) -> 'PiecewisePolynomial':
        """zero vertical displacement at left endpoint"""
        poly = PiecewisePolynomial(self.keypoints, self.pieces)
        sum_y = 0
        for i in range(len(self)):
            x1, x2 = self.keypoints[i], self.keypoints[i+1]
            piece = [0.0] + self.pieces[i]
            for k in range(1, len(piece)):
                piece[k] /= k
            y1 = self._eval_piece(piece, x1)
            y2 = self._eval_piece(piece, x2)
            piece[0] = sum_y - y1
            sum_y += y2 - y1
            poly.pieces[i] = piece
        return poly

    # optimization

    @staticmethod
    def _piece_optim(piece, x1, x2):
        """optimal x values, does not include endpoints"""
        piece = piece[:]
        for k in range(1, len(piece)):
            piece[k] *= k
        piece = piece[1:]
        roots = np.roots(piece[::-1]).astype(np.complex128)
        res = []
        for x in roots:
            if abs(x.imag) < 1e-8 and x1+1e-12 < x.real < x2-1e-12:
                res.append(x.real)
        return sorted(res)

    def get_keypoints(self):
        """endpoints + optimum"""
        xs, ys = [], []
        for i in range(len(self)):
            x1, x2 = self.keypoints[i], self.keypoints[i+1]
            xm = self._piece_optim(self.pieces[i], x1, x2)
            x = np.concatenate(([x1], xm, [x2]))
            y = self._eval_piece(self.pieces[i], x)
            if len(ys) != 0 and abs(y[0]-ys[-1]) < 1e-12:
                x, y = x[1:], y[1:]
            xs = np.concatenate((xs, x))
            ys = np.concatenate((ys, y))
        return xs, ys

    def get_optim(self, absolute=False):
        xs, ys0 = self.get_keypoints()
        ys = np.abs(ys0) if absolute else ys0
        maxi, mini = np.argmax(ys), np.argmin(ys)
        maxy, miny = ys[maxi], ys[mini]
        maxs = (np.mean(xs[np.where(abs(ys-np.full(len(ys),maxy))<1e-6)]), maxy)
        mins = (np.mean(xs[np.where(abs(ys-np.full(len(ys),miny))<1e-6)]), miny)
        if absolute:
            return (maxs[0], ys0[maxi])
        return mins, maxs

    # arithmic - not using operator overload to keep it clear

    def mul(self, c) -> 'PiecewisePolynomial':
        """multiply by a constant"""
        poly = PiecewisePolynomial(self.keypoints, self.pieces)
        for i in range(len(poly.pieces)):
            piece = self.pieces[i][:]
            for k in range(len(piece)):
                piece[k] *= c
            poly.pieces[i] = piece
        return poly

    @staticmethod
    def _sub_piece(piece1, piece2):
        res = [0] * max(len(piece1), len(piece2))
        for i in range(len(piece1)):
            res[i] += piece1[i]
        for i in range(len(piece2)):
            res[i] -= piece2[i]
        return res

    def sub(self, piece) -> 'PiecewisePolynomial':
        """subtract by a smooth polynomial"""
        poly = PiecewisePolynomial(self.keypoints, self.pieces)
        for i in range(len(poly.pieces)):
            poly.pieces[i] = self._sub_piece(poly.pieces[i], piece)
        return poly

    @staticmethod
    def _mul_piece(p, q):
        r = [0]*(len(p)+len(q)-1)
        for kp in range(len(p)):
            for kq in range(len(q)):
                r[kp+kq] += p[kp]*q[kq]
        return r

    def polymul(this, that) -> 'PiecewisePolynomial':
        """multiply two piecewise polynomials"""
        keypoints, pieces = [], []
        xthis_map, xthat_map = {}, {}
        for i in range(len(this)+1):
            x = this.keypoints[i]
            xthis_map[x] = [1, 0] if i == len(this) else this.pieces[i]
            keypoints.append(x)
        for i in range(len(that)+1):
            x = that.keypoints[i]
            xthat_map[x] = [1, 0] if i == len(that) else that.pieces[i]
            keypoints.append(x)
        keypoints = sorted(set(keypoints))
        pieces = []
        piece_this, piece_that = [1, 0], [1, 0]
        for i in range(len(keypoints)-1):
            x = keypoints[i]
            if x in xthis_map:
                piece_this = xthis_map[x]
            if x in xthat_map:
                piece_that = xthat_map[x]
            pieces.append(this._mul_piece(piece_this, piece_that))
        return PiecewisePolynomial(keypoints, pieces)
```

```py
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
```


## `cross_section_analysis.py`

Analyze a cross section: calculating geometric parameters, computing the individual and overall failure load based on material strengths. Used by `bridge_analysis.py`.

Takes the minimum FoS of the following four failure modes:
 - Flexural: `min(tensile, compressive)` calculate them at once for convenience
    - Compression is going to be the biggest one because the matboard's compressive strength is much lower than its tensile strength
 - Thin plate buckling: uses the formula provided in the course notes. We use $k=4$ for two fixed ends and $k=6$ for half-compression pieces. We don't have a formula for $k$ for "trapezoidal" stress distributions so we use a guessed formula that avoids overestimation.
 - Shear of material: $\tau=VQ/Ib$
 - Shear buckling: to set $a$, set the global parameter `LENGTH`

Parts that we think there's no practical value for the program to calculate:
 - Buckling at open flanges: two layers of matboards with a width of $10\mathrm{mm}$ is unlikely a problem.
 - Shear buckling at glue joints: we found this part to be difficult to calculate based on the assumption. We consider this not an issue because glue joints are close to top and bottom that have low $Q$ values. In case it is an issue, we prestress the bridge while waiting for contact cement to cure.

Assumption: The thickness of the matboard is small compared to the length and width of the components. They are treated as line segments with uniform line density.
 - A comparison shows this assumption affects the buckling failure loads by up to $\pm5\%$.

Note that when running this file, `STRICT_MODE` in `beam_analysis.py` is `False`.

```py
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
```


## `cross_section_analysis_examples.py`

As an early example to explore the optimal bridge cross section. Includes models based on parameters and optimizers for the parameters. Turned out an inverted trapezoid glued at the top yields the highest flexural strength. A circular geometry has a higher FoS but we don't trust it intuitively.

This file no longer runs after `cross_section_analysis.py` was updated. Check GitHub commit history if you want to run it.

```py
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
```


## `cross_section_models.py`

Predefined cross-section geometries used in the bridge.

```py
# Defines common cross section geometry.
# All function return (points, glues) = (list[list[point]], [])

# Points are listed in CCW convention


import math


def trapezoid(wt, wb, h):
    """You can guess what these parameters are"""
    assert max(wt, wb, h) > 0
    points = [
        (0.5*wt, h),
        (-0.5*wt, h),
        (-0.5*wb, 0),
        (0.5*wb, 0),
        (0.5*wt, h),
        (-0.5*wt, h)
    ]
    return [points], []


def trapezoid_nowrap(wt, wb, h):
    """No overlapping"""
    assert max(wt, wb, h) > 0
    points = [
        (0.5*wt, h),
        (-0.5*wt, h),
        (-0.5*wb, 0),
        (0.5*wb, 0),
        (0.5*wt, h)
    ]
    return [points], []


def trapezoid_glue_edge_2(wt, wb, h):
    """With a +2x10mm platform on the top"""
    assert max(wt, wb, h) > 0
    body = [
        (-0.5*wt, h),
        (-0.5*wb, 0),
        (0.5*wb, 0),
        (0.5*wt, h),
    ]
    top = [
        (0.5*wt+10, h),
        (-0.5*wt-10, h),
    ]
    strips = [
        [(0.5*wt+10, h), (0.5*wt, h)],
        [(0.5*wt, h), (-0.5*wt, h)],
        [(-0.5*wt, h), (-0.5*wt-10, h)],
    ]
    return [top, body] + strips, []


def trapezoid_glue_edge_1(wt, wb, h):
    """single layer at the top,
        bad for both middle and side beams"""
    assert max(wt, wb, h) > 0
    assert wt > 20
    body = [
        (-0.5*wt, h),
        (-0.5*wb, 0),
        (0.5*wb, 0),
        (0.5*wt, h),
    ]
    top = [
        (0.5*wt+10, h),
        (-0.5*wt-10, h),
    ]
    strips = [
        [(-0.5*wt, h), (-0.5*wt-10, h)],
        [(-0.5*wt+10, h), (-0.5*wt, h)],
        [(0.5*wt, h), (0.5*wt-10, h)],
        [(0.5*wt+10, h), (0.5*wt, h)]
    ]
    return [top, body] + strips, []


def trapezoid_rect_support(wt, wb, h):
    """Wrap a rectangle around the trapezoid to support the end"""
    assert wt > wb
    start = (wb/2+(wt-wb)/(2*h)*(0.8*h-10), 0.8*h-10)
    points = [
        (-start[0], start[1]),
        (-0.5*wt, h),
        (-0.5*wt, 0),
        (0.5*wt, 0),
        (0.5*wt, h),
        start,
    ]
    return [points], []


def trapezoid_rect_support_diaphragm(wt, wb, h):
    """Two pieces of triangles to accomodate the above function"""
    assert wt > wb
    parts = [
        [(0.5*wt, 0), (0.5*wt, h), (0.5*wb, 0), (0.5*wt, 0)],
        [(-0.5*wt, 0), (-0.5*wb, 0), (-0.5*wt, h), (-0.5*wt, 0)]
    ]
    return parts, []


def trapezoid_edge_strengthen(wt, wb, h, et, es):
    """Strengthen; See the final design to get what it looks like"""
    theta = math.atan((wt-wb)/(2*h))
    ex = 0.5*wt - es*math.sin(theta)
    ey = h - es*math.cos(theta)
    parts = [
        [(ex, ey), (0.5*wt, h), (0.5*wt-et, h)],
        [(-(0.5*wt-et), h), (-0.5*wt, h), (-ex, ey)]
    ]
    return parts, []


def trapezoid_wall_strengthen(wt, wb, h):
    """Strengthen to prevent shearing, not an issue"""
    parts = [
        [(0.5*wb, 0), (0.5*wt, h)],
        [(-0.5*wb, 0), (0.5*wt, h)]
    ]
    return parts, []
```


## `bridge_analysis.py`

The heaviest source file of this project. Contains data structure that represents the entire bridge, failure analysis, packing into the matboard, and optimization.

Note that the optimization part contains constraints and additional factor of safeties not shown in the plot. This is to make our intuition happy.

```py
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
```


## `design_s2.py`

First full-bridge design. Experiments with trapezoid orientation, strengthenings, and beam joinings. The result has a 2.5 FoS but looks "kinda sus."

```py
import bridge_analysis as ba
import cross_section_models as csm
from export_bridge import export_bridge


def mix(a, b, t):
    return a + (b-a) * t


# total 9 parameters:
# @wt, @wb, @h
# length of each side beam, @sl
# length of the central beam @cl
# strengthen middle edges @mel, @met, @mes, length, top width, side width

# Trapezoid body: folded
# Trapezoid top: folded, or glue the edges (10mm padding)
GLUE_EDGE = True
# Diaphragms: glue the edges

C = __import__('beam_analysis').LENGTH/2
E11, E12, E21, E22 = 10, 30, 50, 70
EC = 0.5*(E12+E21)
D1 = mix(EC, 2*C-EC, 0.16)
D2 = mix(EC, 2*C-EC, 0.38)

# six diaphragms
# two at the ends, four uniformly spaced


def calc_cross_section(wt, wb, h, sl, cl, mel, met, mes):
    # geometry constraints
    if not min(wt, wb, h, cl, mel, met, mes) > 0:
        return None
    if wt < (90 if GLUE_EDGE else 100):  # top must hold the train
        return None
    if wb < 20 or h > 200:  # hard constraints
        return None
    if mel > C:  # these can't be higher than half of the bridge length
        return None
    if 2*met > wt or met < 10:  # no overlapping?
        return None
    if cl > 2*C:  # must not longer than bridge
        return None
    if 2*sl+cl < 2*C+20*2:  # frame must be longer than bridge, with glue joints
        """Failure of glue joint due to tension
            400N / 2MPa = 200mm²
            200mm²/100mm = 2mm is theoretically enough
            Intuitively it doesn't make sense so put 20mm.
        """
        return None
    if wt < wb:  # force bottom large trapezoid
        return None
    if cl < 2*C-2*D1:  # optional, put all diaphragms in the middle span
        return None
    # beam cross sections
    offset = 1.5
    if GLUE_EDGE:
        tm = csm.trapezoid_glue_edge_2(wt, wb, h)
        #ts = csm.trapezoid_glue_edge_1(wt, wb, h)
        ts = csm.trapezoid_glue_edge_2(wt, wb, h)
    else:
        tm = csm.trapezoid(wt, wb, h)
        ts = csm.trapezoid(wt, wb, h)
    sp = csm.trapezoid_rect_support(wt, wb, h)
    tes = csm.trapezoid_edge_strengthen(wt, wb, h, met, mes)
    tws = csm.trapezoid_wall_strengthen(wt, wb, h)
    res = [
        ba.BridgeCrossSection('side_beam_1', *ts, 0, sl, [offset]+[0]*4),
        ba.BridgeCrossSection('side_beam_2', *ts, 2*C-sl, 2*C, [offset]+[0]*4),
        ba.BridgeCrossSection('central_beam', *tm, C-0.5*cl, C+0.5*cl, [2*offset]+[offset]*4),
        ba.BridgeCrossSection('mid_strengthen', *tes, C-0.5*mel, C+0.5*mel),
        ba.BridgeCrossSection('support_1', *sp, E11, E22, -offset),
        ba.BridgeCrossSection('support_2', *sp, 2*C-E22, 2*C-E11, -offset),
    ]
    """Buckling of flange
        Failure stress 0.425π²E / 12(1-μ²) * (1.27mm/10mm)² = 23.5 MPa
        Far higher than the board's compressive strength. Not a big issue.
    """
    return res


def calc_diaphragms(wt, wb, h, sl, cl, mel, met, mes):
    if wt < wb:  # force bottom large trapezoid
        return None
    x1 = 50
    x2 = 2*C - 50
    """Buckling
        each wheel 20mm*10mm = 200mm² (measured)
        pressure on surface (400N/12)/(200mm²) = 0.1667 MPa
        all area 1.27mm * (100mm * n + 960mm * 2) = 2440mm² + 127mm² * n
        max pressure (400N)/(2440mm²) = 0.1640 MPa
        thin plate buckling load 4π²E / 12(1-μ²) * (1.27mm/100mm)² = 2.21 MPa
        shear buckling load 5π²E / 12(1-μ²) * ((1.27mm/100mm)² + (1.27mm/100mm)²) = 5.53 MPa
        single layer of matboard is enough for resisting buckling
        at the end: 274N / (2*50mm*1.27mm) = 2.16 MPa (might be an issue)
    """
    offset = 1.0
    cs = csm.trapezoid_nowrap(wt, wb, h)
    sp = csm.trapezoid_rect_support_diaphragm(wt, wb, h)
    res = [
        ba.BridgeCrossSection('de:support_1_1%a', *sp, E11, E12, -0.5*offset),
        ba.BridgeCrossSection('de:support_1_2%b', *sp, E21, E22, -0.5*offset),
        ba.BridgeCrossSection('d:support_1%a', *cs, E12, E21, 0),
        ba.BridgeCrossSection('de:support_2_1%c', *sp, 2*C-E12, 2*C-E11, -0.5*offset),
        ba.BridgeCrossSection('de:support_2_2%d', *sp, 2*C-E22, 2*C-E21, -0.5*offset),
        ba.BridgeCrossSection('d:support_2%b', *cs, 2*C-E21, 2*C-E12, 0),
        ba.BridgeCrossSection('d:d1_1%c', *cs, D1-10, D1+10, 0.5*offset),
        ba.BridgeCrossSection('d:d1_2%d', *cs, 2*C-D1-10, 2*C-D1+10, 0.5*offset),
        ba.BridgeCrossSection('d:d2_1', *cs, D2-10, D2+10, 0.5*offset),
        ba.BridgeCrossSection('d:d2_2', *cs, 2*C-D2-10, 2*C-D2+10, 0.5*offset),
    ]
    return res


if __name__ == "__main__":

    if GLUE_EDGE:  # FoS = 2.53
        initial_params = [91, 30, 65, 180, 990, 500, 20, 15]
        initial_params = [90.27670975496106, 28.497081352430108, 83.70282958637046, 197.85696889711943, 1005.6537386401361, 568.2762296922637, 35.328454803561115, 9.2141692024721]
    else:  # not trustworthy
        initial_params = [101, 30, 65, 400, 600, 600, 20, 20]
        initial_params = [100.82165954525692, 64.36032702165264, 90.4114510065952, 652.6591967573131, 70.57551660710013, 41.45188009721109, 1.939084293269092, 85.55489634215272]

    #initial_params = [100, 40, 100, 180, 1000, 500, 20, 15]  # invalid
    bridge = ba.Bridge(
        calc_cross_section, calc_diaphragms,
        [[75, 150], [20, 100], [20, 100],
         [200, 630], [200, 1000],
         [0, 800], [0, 40], [0, 40],
        ],
        ['wt', 'wb', 'h', 'sl', 'cl', 'mel', 'met', 'mes'],
        initial_params
    )
    bridge.assert_ccw()

    #bridge.optimize_params()
    #bridge.analyze(bridge.params, plot=True)
    #bridge.plot_3d(zoom=1.5)

    description_ge = """
<p>
The bridge design consists of three segments symmetrical about the middle span.
The middle one is thicker than the two at the sides due to glue joints.
Glue joints are closer to the end for a lower bending moment and curvature.
</p>
<p>
The body of the bridge is an inverted trapezoid.
There are two layers of matboard at the top, with flanges extending beyond the top of the trapezoid
to increase the height of centroidal axis and therefore decrease <i>y</i> in the equation <i>My/I</i>
and increase compressive failure load.
Components are designed to strengthen the middle span and the supports of the bridge.
</p>
<p>
The bottom and sides of the trapezoid are folded.
Inspired by the winning group last year,
we decide to glue the edges between paddings instead of using folded tabs
for the top of the trapezoid and diaphragms.
These parts are mainly subjected to compression and are less likely to fail due to tension and shear.
Intuitively, folding tabs involves deforming the matboard, making it more likely to buckle.
</p>
<p>
Analysis based on the CIV102 course shows the bridge will first fail due to compression in the midspan.
However, intuition tells us the glue joint between beam segments may be the weakest part.
The bridge may also crush at the ends due to reaction force.
</p>
""".strip()

    export_bridge(
        bridge,
        "design_s2_ge" if GLUE_EDGE else "design_s2_fold",
        [100, 300, 600] if GLUE_EDGE else [],
        description_ge if GLUE_EDGE else ""
    )
```


## `design_nd.py`

Final full-bridge design. Add more diaphragms in the middle to resist the vertical compression of the beam due to flexural stress + deflection.

```py
import bridge_analysis as ba
import cross_section_models as csm
from export_bridge import export_bridge


def mix(a, b, t):
    return a + (b-a) * t


# total 9 parameters:
# @wt, @wb, @h
# length of each side beam, @sl
# length of the central beam @cl
# strengthen middle edges @mel, @met, @mes, length, top width, side width

# Trapezoid body: folded
# Trapezoid top: glue the edges (10mm padding)
# Diaphragms: glue the edges

C = __import__('beam_analysis').LENGTH/2
E11, E12, E21, E22 = 0, 20, 40, 60
EC = 0.5*(E12+E21)

# 7+2 diaphragms


def calc_cross_section(wt, wb, h, sl, cl, mel, met, mes):
    # geometry constraints
    if not min(wt, wb, h, cl, mel, met, mes) > 0:
        return None
    if wt < 90:  # top must hold the train
        return None
    if wb < 20 or h > 200:  # hard constraints
        return None
    if mel > C:  # these can't be higher than half of the bridge length
        return None
    if 2*met > wt:  # no overlapping?
        return None
    if cl > 2*C:  # must not longer than bridge
        return None
    if 2*sl+cl < 2*C+20*2:  # frame must be longer than bridge, with glue joints
        """Failure of glue joint due to tension
            400N / 2MPa = 200mm²
            200mm²/100mm = 2mm is theoretically enough
            Intuitively it doesn't make sense so put 20mm.
        """
        return None
    if wt < wb:  # force bottom large trapezoid
        return None
    # beam cross sections
    offset = 1.5
    tm = csm.trapezoid_glue_edge_2(wt, wb, h)
    ts = csm.trapezoid_glue_edge_2(wt, wb, h)
    sp = csm.trapezoid_rect_support(wt, wb, h)
    tes = csm.trapezoid_edge_strengthen(wt, wb, h, met, mes)
    tws = csm.trapezoid_wall_strengthen(wt, wb, h)
    res = [
        ba.BridgeCrossSection('side_beam_1', *ts, 0, sl, [offset]+[0]*4),
        ba.BridgeCrossSection('side_beam_2', *ts, 2*C-sl, 2*C, [offset]+[0]*4),
        ba.BridgeCrossSection('central_beam', *tm, C-0.5*cl, C+0.5*cl, [2*offset]+[offset]*4),
        ba.BridgeCrossSection('mid_strengthen', *tes, C-0.5*mel, C+0.5*mel),
        ba.BridgeCrossSection('support_1', *sp, E11, E22, -offset),
        ba.BridgeCrossSection('support_2', *sp, 2*C-E22, 2*C-E11, -offset),
    ]
    """Buckling of flange
        Failure stress 0.425π²E / 12(1-μ²) * (1.27mm/10mm)² = 23.5 MPa
        Far higher than the board's compressive strength. Not a big issue.
    """
    return res


def calc_diaphragms(wt, wb, h, sl, cl, mel, met, mes):
    if wt < wb:  # force bottom large trapezoid
        return None
    x1 = 50
    x2 = 2*C - 50
    """Buckling
        each wheel 20mm*10mm = 200mm² (measured)
        pressure on surface (400N/12)/(200mm²) = 0.1667 MPa
        all area 1.27mm * (100mm * n + 960mm * 2) = 2440mm² + 127mm² * n
        max pressure (400N)/(2440mm²) = 0.1640 MPa
        thin plate buckling load 4π²E / 12(1-μ²) * (1.27mm/100mm)² = 2.21 MPa
        shear buckling load 5π²E / 12(1-μ²) * ((1.27mm/100mm)² + (1.27mm/100mm)²) = 5.53 MPa
        single layer of matboard is enough for resisting buckling
    """
    D1 = mix(EC, 2*C-EC, 0.14)
    D2 = mix(EC, 2*C-EC, 0.3)
    D3 = mix(EC, 2*C-EC, 0.4)
    D4 = mix(EC, 2*C-EC, 0.5)
    D1 = sl+10
    D2 = C-0.5*mel-10
    D3 = 0.5*(D2+D4)
    if D2-D1 < 20:
        return None
    offset = 1.0
    cs = csm.trapezoid_nowrap(wt, wb, h)
    sp = csm.trapezoid_rect_support_diaphragm(wt, wb, h)
    res = [
        ba.BridgeCrossSection('de:support_1%a', *cs, E12, E21, 0),
        ba.BridgeCrossSection('de:support_1_1%a', *sp, E11, E12, -0.5*offset),
        ba.BridgeCrossSection('de:support_1_2%c', *sp, E21, E22, -0.5*offset),
        ba.BridgeCrossSection('de:support_2%b', *cs, 2*C-E21, 2*C-E12, 0),
        ba.BridgeCrossSection('de:support_2_1%b', *sp, 2*C-E12, 2*C-E11, -0.5*offset),
        ba.BridgeCrossSection('de:support_2_2%d', *sp, 2*C-E22, 2*C-E21, -0.5*offset),
        ba.BridgeCrossSection('d:d1_1%c', *cs, D1-10, D1+10, 0.5*offset),
        ba.BridgeCrossSection('d:d1_2%d', *cs, 2*C-D1-10, 2*C-D1+10, 0.5*offset),
        ba.BridgeCrossSection('d:d2_1%%e', *cs, D2-10, D2+10, 0.5*offset),
        ba.BridgeCrossSection('d:d2_2%%e', *cs, 2*C-D2-10, 2*C-D2+10, 0.5*offset),
        ba.BridgeCrossSection('d:d3_1%%f', *cs, D3-10, D3+10, 0.5*offset),
        ba.BridgeCrossSection('d:d3_2%%f', *cs, 2*C-D3-10, 2*C-D3+10, 0.5*offset),
        ba.BridgeCrossSection('d:d4', *cs, D4-10, D4+10, 0.5*offset),
    ]
    return res


if __name__ == "__main__":

    # FoS = 2.40
    initial_params = [91, 30, 65, 180, 990, 500, 20, 15]
    initial_params = [91.61580029591246, 46.463653533343106, 81.17993434406554, 219.58374917543213, 927.3505703015933, 402.58218749702394, 20.25507075560746, 8.39290511377031]

    bridge = ba.Bridge(
        calc_cross_section, calc_diaphragms,
        [[75, 150], [20, 100], [20, 100],
         [200, 630], [200, 1000],
         [0, 800], [0, 40], [0, 40],
        ],
        ['wt', 'wb', 'h', 'sl', 'cl', 'mel', 'met', 'mes'],
        initial_params
    )
    bridge.assert_ccw()

    #bridge.optimize_params()
    #bridge.analyze(bridge.params, plot=True)
    #bridge.plot_3d(zoom=1.5)

    description = """
<p>
The bridge design consists of three segments symmetrical about the middle span.
The middle one is thicker than the two at the sides due to glue joints.
Glue joints are closer to the end for a lower bending moment and curvature.
</p>
<p>
The body of the bridge is an inverted trapezoid.
There are two layers of matboard at the top, with flanges extending beyond the top of the trapezoid
to increase the height of centroidal axis and therefore decrease <i>y</i> in the equation <i>My/I</i>
and increase compressive failure load.
Components are designed to strengthen the middle span and the supports of the bridge.
</p>
<p>
The bottom and sides of the trapezoid are folded.
Inspired by the winning group last year,
we decide to glue the edges between paddings instead of using folded tabs
for the top of the trapezoid and diaphragms.
These parts are mainly subjected to compression and are less likely to fail due to tension and shear.
Intuitively, folding tabs involves deforming the matboard, making it more likely to buckle.
</p>
<p>
This bridge design has 7 diaphragms in the middle span.
Since the analysis shows shear buckling is not the main cause of the bridge's failure,
we consider there is no need for extra diaphragms at the ends.
The cross section of the middle of the bridge will likely shrink vertically due to bending moment
and therefore have a decreased second moment of area.
Diaphragms are placed in the middle to resist such deformation.
</p>
<p>
Analysis based on the CIV102 course shows the bridge will first fail due to compression in the midspan.
However, intuition tells us the glue joint between beam segments may be the weakest part.
The bridge may also crush at the ends due to reaction force.
</p>
""".strip()

    export_bridge(
        bridge,
        "design_nd",
        [100, 300, 600],
        description
    )
```


## `export_bridge.py`

Export the result of analysis and optimization to HTML. Makes it easier for us to cut the matboard and fold the shapes.

Example generated HTML: https://harry7557558.github.io/engsci-2t6/civ102-project/design_nd.html

```py
from bridge_analysis import *


def format_float(x, s=3):
    """Slide-rule precision"""
    if abs(x) < 1e-4:
        return "0"
    sigfig = s+1 if np.log10(abs(x))%1.0 < np.log10(2) else s
    tz = ""
    while x >= 10**sigfig:
        x = round(x/10)
        tz += "0"
    return f"{{:.{sigfig}g}}".format(x) + tz


def wrap_h2(text):
    h, p, m = 0, 1, 65521
    for c in text:  # rolling hash
        h = (h + ord(c)) * p % m
        p = p * 31 % m
    h = hex(h)[2:]
    return f'<h2 id="{h}"><a href="#{h}">{text}</a></h2>'


def wrap_svg(content, w, h):
    if isinstance(content, list):
        content = '\n'.join(content)
    return f"""
<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">
    {content}
</svg>"""


def wrap_table(contents):
    nrow = len(contents)
    ncol = max([len(row) for row in contents])
    trs = []
    for row in contents:
        tds = []
        for ele in row:
            tds.append("<td>"+ele+"</td>")
        trs.append("<tr>"+'\n'.join(tds)+"</tr>")
    return "<table>" + '\n'.join(trs) + "</table>"


def wrap_html(children):
    content = '\n'.join(children)
    return f"""<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <title>CIV102 Bridge Design</title>
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="robots" content="none" />
    <style>
        body{{ padding: 0px 10px; }}
        h2 a{{ color: black; }}
        img{{ max-width: 100%; max-height: 600px; display: block; }}
        td{{ padding: 10px; }}
    </style>
</head>
<body>
    {content}
</body>
</html>"""


def export_cross_section_svg(cs: BridgeCrossSection):
    t = csa.LINDEN_M  # thickness of the matboard
    children = []
    minx, maxx, miny, maxy = np.inf, -np.inf, np.inf, -np.inf
    for part in cs.parts_offset:
        points = []
        for p in part:
            minx, maxx = min(minx, p[0]), max(maxx, p[0])
            miny, maxy = min(miny, p[1]), max(maxy, p[1])
            points.append(','.join(map(str, p)))
        points = ' '.join(points)
        children.append(f'<polyline points="{points}" '
                        f'stroke-width="{t}" stroke="black" fill="none" />')
    pad, th = 10, 16
    minx, maxx, miny, maxy = minx-pad, maxx+pad, miny-pad, maxy+pad
    transform = ' '.join(map(str,
        [1, 0, 0, -1, -minx, maxy+th]))
    g = f'<g transform="matrix({transform})">' + '\n'.join(children) + '</g>'
    label = cs.label[:(cs.label+'%').find('%')]
    text = f'<text x="{0.5*(maxx-minx)}" y="{th}" text-anchor="middle">{label}</text>'
    return wrap_svg([g, text], maxx-minx, maxy-miny+th)


def export_cross_section_text(cs: BridgeCrossSection, show_length=True):
    lines = []
    if show_length:
        lines.append("length: " + format_float(cs.x1-cs.x0))
    else:
        lines.append("location: " + format_float(0.5*(cs.x0+cs.x1)) + " from one end")
    for part in cs.parts_offset:
        points = []
        for p in part:
            s = ', '.join([format_float(p[0]), format_float(p[1])])
            points.append(f"({s})")
        points = ', '.join(points) + ';'
        lines.append(points)
    return "<div>" + "<br/>\n".join(lines) + "</div>"


def export_bridge(bridge: Bridge,
                  filename: str,
                  manual_locations: list[float],
                  description: str):
    """filename should have no extension"""
    html_elements = [
        f"<h1>CIV102 Bridge Design <code>`{filename}`</code></h1>",
        description,
        "<hr/>"
    ]

    # figures
    if True:
        imgsrc = filename+"_fos.png"
        bridge.analyze(bridge.params, plot=True, show=False)
        plt.savefig(imgsrc)
        html_elements += [
            wrap_h2("FoS plots"),
            """<p><b>Top figure</b>:
the first four FoS are the actual FoS without dividing by factors (I asked Raymond).
Ignore flexural shear FoS when writing the report
(and possibly remove that plot in the future)
because I don't think my formula is correct.</p>""",
            """<p><b>Bottom figure</b>:
How the matboard will be cut. Gray is unused region.
Light yellow is support. Light blue is diaphragm.
A light pink rectangle contains multiple diaphragms.
A more detailed figure appears at the bottom of this page.</p>""",
            f'<img src="{imgsrc}" />',
            '<hr/>'
        ]
        imgsrc = filename+"_3d.png"
        bridge.plot_3d(zoom=1.5, show=False)
        plt.savefig(imgsrc)
        html_elements += [
            wrap_h2("3D bridge plot"),
            "<p>Gray is bridge body. Black is diaphragm. "
            "Red highlights the edge of the platform when the bridge is tested.</p>",
            f'<img src="{imgsrc}" />',
            '<hr/>'
        ]

    # cross sections
    html_elements += [
        wrap_h2("Cross-section components"),
        "<p>All lengths and coordinates are in mm. One pixel on this page is 1&nbsp;mm. "
        "Zoom in in your browser if you think the figures are too small.</p>",
        "<p>A cross section is represented as a list of point coordinates in order. "
        "This format makes it easier to analyze computationally.</p>",
    ]
    cross_sections = bridge.calc_cross_section(*bridge.params)
    if cross_sections is None:
        print("Cross section constraint violation.")
        cross_sections = []
    for cs in cross_sections:
        svg = export_cross_section_svg(cs)+"<br/>\n"
        text = export_cross_section_text(cs)
        html_elements.append(wrap_table([[svg, text]]))
    html_elements.append("<hr/>")

    # diaphragms
    html_elements += [
        wrap_h2("Diaphragm components"),
        "<p>All locations and coordinates are in mm. One pixel on this page is 1&nbsp;mm.</p>",
        "<p>A diaphragm a polygon represented as a list of point coordinates in order. "
        "The last point in the point list must be the same as the first point to enforce a closed polygon.</p>",
        "<p>In bridge construction, two 1cm strips at each side of a diaphragm and the thin edges of the diaphragm are glued.</p>",
    ]
    diaphragms = bridge.calc_diaphragms(*bridge.params)
    if diaphragms is None:
        print("Diaphragm constraint violation.")
        diaphragms = []
    diaphragms.sort(key=lambda d: d.x0)
    for d in diaphragms:
        svg = export_cross_section_svg(d)+"<br/>\n"
        text = export_cross_section_text(d, False)
        html_elements.append(wrap_table([[svg, text]]))
    html_elements.append("<hr/>")

    # manual calculation locations
    html_elements += [
        wrap_h2("Cross-sections to manually calculate"),
        "<p>Manually analyze these cross sections for the report.</p>",
        "<p>Note that lines may overlap in the diagram. "
        "The coordinates are more trustworthy. "
        "Zoom in in your browser if you think the figures are too small. ",
        "Glue joints can be identified by inspecting visually.</p>",
        "<p>Failure loads for <i>some</i> failure modes calculated from my computer program are given. "
        "The initial weight of the train is 400N. "
        "Since I considered additional factors, I think it is acceptable that a number calculated based on the course is different from the program's result "
        "(and I expect some hand results to be larger). "
        "Let me know if you get a number very different from the program's number so I can check for potential mistakes in my code.</p>",
    ]
    for x in manual_locations:
        # get parts
        parts, parts_offset = [], []
        for cs in cross_sections:
            if cs.x0 <= x <= cs.x1:
                parts += cs.parts
                parts_offset += cs.parts_offset
        assert len(parts) != 0

        # shear buckling `a`
        csa.LENGTH = 0  # for shear buckling
        for dc1, dc2 in zip(diaphragms[:-1], diaphragms[1:]):
            x1 = 0.5*(dc1.x0+dc1.x1)
            x2 = 0.5*(dc2.x0+dc2.x1)
            if x1 <= x <= x2:
                csa.LENGTH = x2 - x1
                break
        assert csa.LENGTH != 0

        # key points
        xs = []
        for cs in cross_sections:
            xs += [cs.x0, cs.x1]
        #for d in diaphragms:
        #    xs += [d.x0, 0.5*(d.x0+d.x1), d.x1]
        xs = sorted(set(xs))
        for x1, x2 in zip(xs[:-1], xs[1:]):
            if x1 <= x <= x2:
                break
        csa.BM_MAX = max(get_max_bend(x1, x2), 1)
        csa.SHEAR_MAX = max(get_max_shear(x1, x2), 1)

        # properties
        fos_bend, fos_bend_buckle, fos_shear, fos_shear_buckle = \
                    csa.analyze_cross_section(
                parts, parts_offset, return_full=True)
        cs = BridgeCrossSection(f"x={x}", parts_offset, [], x1, x2)
        svg = export_cross_section_svg(cs)
        text = export_cross_section_text(cs)
        W0 = 400
        fos = '<div>' + '<br/>\n'.join([
            "Location: (" + format_float(x1) + "&nbsp;mm, " + format_float(x2) + "&nbsp;mm)",
            "Max shear: " + format_float(csa.SHEAR_MAX) + "&nbsp;N",
            "Max bending: " + format_float(1e-3*csa.BM_MAX) + "&nbsp;N⋅m"
            "<hr/>"
            "Flexural: " + format_float(W0*fos_bend) + "&nbsp;N",
            "Buckling: " + format_float(W0*fos_bend_buckle) + "&nbsp;N",
            "Shear: " + format_float(W0*fos_shear) + "&nbsp;N",
            "Shear buckling: " + format_float(W0*fos_shear_buckle) + "&nbsp;N"
        ]) + '</div>'
        html_elements.append(wrap_table([[svg, text, fos]]))
    html_elements.append("<br/><hr/>")

    # how the matboard will be cut
    html_elements += [
        wrap_h2("How the matboard will be cut"),
        f"""<p>
The matboard is divided into rectangles.
Bottom left is (0, 0), top right is ({MATBOARD_W}, {MATBOARD_H}).
Light yellow is support. Light blue is diaphragm.
White is beam. Gray is unused region.
All units are in mm. One pixel in this diagram is 1&nbsp;mm.
</p><p>
On a desktop device, you can mouse hover a rectangle to read the numbers.
The second row contains the distance of the vertices of the rectangle from the edge of the matboard.
The third row contains the width and height of the rectangle on its own coordinate system.
Note that rectangles may be rotated.
A sequence of point coordinates in the rectangle's coordinate system
representing the cut or fold is followed (if cut or fold exists.)
</p><p>
When marking and cutting the matboard,
label each rectangle before cutting it off so we don't mess it up.
The bridge components are packed very compactly.
Make sure you make no mistake when marking and cutting the board -
there is no room for a mistake.
We have extra materials left, and we may consider using them to
strenghten the endpoints of the bridge that support the reaction force.
</p>""",
    ]
    html_elements += [
        '<div id="display" style="position:fixed;display:none;background-color:rgba(250,230,240,0.95);border:1px solid gray;padding:10px;max-width:500px;"></div>'
        """<script>
var mouseX = -1, mouseY = -1;
document.addEventListener('mousemove', function(event) {
    mouseX = event.clientX, mouseY = event.clientY;
});
function displayText(text) {
    let display = document.getElementById("display");
    if (text == '') { display.style.display = 'none'; return; }
    display.style.display = "block";
    if (mouseX < 0.5*window.innerWidth) {
        display.style.left = (mouseX+10)+'px';
        display.style.removeProperty('right');
    }
    else {
        display.style.right = (window.innerWidth-mouseX)+'px';
        display.style.removeProperty('left');
    }
    if (mouseY < 0.5*window.innerHeight) {
        display.style.top = (mouseY+10)+'px';
        display.style.removeProperty('bottom');
    }
    else {
        display.style.bottom = (window.innerHeight-mouseY)+'px';
        display.style.removeProperty('top');
    }
    display.innerHTML = text;
}
</script>"""
    ]
    svg_rects = []
    svg_marks = []
    svg_labels = []
    def add_label(xm, ym, x, y, rotate, label):
        nonlocal svg_labels
        svg_labels += [
            f'<text x="{x}" y="{y+4}" text-anchor="middle" style="pointer-events:none;"'
            f' transform="scale(1,-1) rotate({rotate},{xm},{ym})">{label}</text>'
        ]
    def add_rect(x, y, w, h, label, caption):
        nonlocal svg_rects
        fill = plot_rect_color('' if is_diaphragm_label(label) else label)
        svg_rects += [  
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}"'
            f' stroke="black" stoke-width="1px" fill="{fill}"\n'
            f' onmousemove=\'displayText("{caption}");\'/>'
        ]
        xm, ym = x+0.5*w, -(y+0.5*h)
        if '\n' in label:
            labels = []
            for l in label.strip().split('\n'):
                if l not in labels:
                    labels.append(l)
            rotate = -90 if w<8*max([len(l) for l in labels]) and h>w else 0
            for i in range(len(labels)):
                add_label(xm, ym, xm, ym+16*(i-0.5*len(labels)+0.5),
                          rotate, labels[i])
        else:
            rotate = -90 if w<8*len(label) and h>w else 0
            add_label(xm, ym, xm, ym, rotate, label)
    def add_mark(points, label):
        nonlocal svg_marks
        fill = 'none'
        if is_diaphragm_label(label):
            fill = plot_rect_color(label)
        s = []
        for p in points:
            s.append(','.join(map(str, p)))
        s = ' '.join(s)
        svg_marks += [
            f'<polyline points="{s}" style="pointer-events:none;"'
            f' stroke="gray" stroke-width="0.5" stroke-dasharray="2" fill="{fill}" />'
        ]
    add_rect(0, 0, MATBOARD_W, MATBOARD_H, '', '')
    for d in diaphragms:
        c = 0.5*(d.x0+d.x1)
        cross_sections += [
            BridgeCrossSection(d.label[:(d.label+'%').find('%')]+'=',
                               d.parts, d.glues, d.x0, d.x1, offset=-csa.LINDEN_M),
        ]
    rects, labels, marks = bridge.generate_rects(
        cross_sections, diaphragms,
        require_labels=True, require_marks=True)
    comps = list(zip(rects, labels, marks))
    packer = newPacker()
    for i in range(len(comps)):
        packer.add_rect(*comps[i][0], str(i))
    transform = ' '.join(map(str,
        [1, 0, 0, -1, 1, MATBOARD_H+1]))
    packer.add_bin(MATBOARD_W, MATBOARD_H)
    packer.pack()
    abin = packer[0]
    if len(abin) < len(rects):
        html_elements.append(
            "<h1 style='color:#b00'>Error: cannot pack all parts into the matboard."
            " The following diagram is incomplete.</h1>")
    for prect in abin:
        i = int(prect.rid)
        rect, label, marks = comps[i]
        # generate caption
        caption = [
            '; '.join(set(label.split('\n'))),
            "(left, bottom, right, top) = ({:.1f}, {:.1f}, {:.1f}, {:.1f})".format(
                prect.x, prect.y,
                MATBOARD_W-(prect.x+prect.width),
                MATBOARD_H-(prect.y+prect.height)),
            "(w, h) = ({:.1f}, {:.1f})".format(rect[0], rect[1]),
            "<hr/>"
        ]
        for mark in marks:
            caption.append(', '.join(
                    ["({:.1f}, {:.1f})".format(p[0], p[1]) for p in mark])+';')
        # add rect
        add_rect(prect.x, prect.y, prect.width, prect.height,
                 label, '<br/>'.join(caption).replace('<hr/><br/>','<hr/>'))
        p0 = np.array((prect.x, prect.y))
        A = np.array([[1,0,0],[0,1,0]] if rect[0]==prect.width
                     else [[0,-1,rect[1]],[1,0,0]])
        if '\n' not in label:
            label = '\n'.join([label]*len(marks))
        for mark, label in zip(marks, label.split('\n')):
            ps = [p0+A[:,0:2].dot(p)+A[:,2] for p in mark]
            add_mark(ps, label)
    svg = [f'<g transform="matrix({transform})">'] + \
          svg_rects + svg_marks + svg_labels + ["</g>"]
    svg = wrap_svg(svg, MATBOARD_W+2, MATBOARD_H+2)
    svg = f"""<div style="overflow-x:scroll" onmouseout="displayText('')">{svg}</div>"""
    html_elements.append(svg)

    # end
    html_elements += ["<br/>"]*5
    html = wrap_html(html_elements)
    open(filename+".html", 'wb').write(bytearray(html, 'utf-8'))
```

