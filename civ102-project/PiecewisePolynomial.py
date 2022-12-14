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
