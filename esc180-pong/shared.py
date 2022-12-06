from math import sqrt, sin, cos, pi
INF = float('inf')
NAN = float('nan')


# vector meth

class vec2:
    def __init__(self, x: float = 0.0, y: float = None):
        if y is None:
            if type(x) in [float, int]:
                y = x
            else:
                x, y = x
        self.x = x
        self.y = y
    def __getitem__(self, i) -> float:
        return [self.x, self.y][i%2]
    def __iter__(self):
        yield self.x
        yield self.y
    def __str__(self) -> str:
        return "({}, {})".format(self.x, self.y)
    def __repr__(self) -> str:
        return str(self)
    def __eq__(self, other) -> bool:
        return self.x == other.x and self.y == other.y
    def clone(self):
        return vec2(self.x, self.y)
    def rot(self):
        return vec2(-self.y, self.x)
    def __add__(self, other):
        return vec2(self.x + other.x, self.y + other.y)
    def __neg__(self):
        return vec2(-self.x, -self.y)
    def __sub__(self, other):
        return vec2(self.x - other.x, self.y - other.y)
    def __mul__(self, k):
        if type(k) == vec2:
            return vec2(k.x*self.x, k.y*self.y)
        else:
            return vec2(k*self.x, k*self.y)
    def __rmul__(self, k: float):
        return vec2(k*self.x, k*self.y)
    def __truediv__(self, k: float):
        if type(k) == vec2:
            return vec2(self.x/k.x, self.y/k.y)
        else:
            return vec2(self.x/k, self.y/k)
    def length(self) -> float:
        return sqrt(self.x**2 + self.y**2)
    def normalize(self):
        return self / self.length()

def dot(p, q):
    return p.x * q.x + p.y * q.y
def det(p, q):
    return p.x * q.y - p.y * q.x
def lerp(p, q, t):
    return p * (1.0-t) + q * t
def reflect(d, n):
    return d-2*dot(d,n)/dot(n,n)*n


# line segment, with ray intersection operation

class LineSegment:
    def __init__(self, p1, p2):
        self.p1 = vec2(p1)
        self.p2 = vec2(p2)
        self.d = p2-p1

    def distance_to(self, p):
        """shortest distance to a point"""
        pa = p - self.p1
        h = dot(pa,self.d)/dot(self.d,self.d)
        #h = max(min(h, 1.0), 0.0)
        return (pa-self.d*h).length()

    def intersect(self, ro, rd):
        """Returns the distance rd*t to intersection
        """
        # p1+d*t = ro+rd*s
        # [d -rd] [t s]ᵀ = ro-p1
        m = 1.0 / det(self.d, -rd)
        b = ro - self.p1
        t = m * det(b, -rd)
        if t < 0 or t > 1:
            return INF
        s = m * det(self.d, b)
        if s < 0:
            return INF
        return s

    @staticmethod
    def circle_intersect(c, r, ro, rd):
        """(ro+rd*t-c)^2 = r^2
            dot(p+rd*t, p+rd*t) = r^2
            dot(rd,rd)t^2 + 2dot(p,rd)t + dot(p,p)-r^2 = 0"""
        p = ro - c
        a = dot(rd,rd)
        b = dot(p,rd)
        c = dot(p,p)
        d = b*b-a*c
        if d < 0:
            return INF
        t1 = (-b-sqrt(d))/a
        t2 = (-b+sqrt(d))/a
        if t1<0: t1 = INF
        if t2<0: t2 = INF
        return min(t1, t2)
        
    def intersect_circle(self, c, r, v):
        """Returns (t, refl)
            either intersect at the line,
            or intersect at an endpoint"""
        n = self.d.rot().normalize()
        if dot(v, n) > 0:
            n = -n
        # intersect at the line
        t = self.intersect(c-r*n, v)
        if t < INF:
            return t, reflect(v, n)
        # intersect at a point
        t1 = self.circle_intersect(c, r, self.p1, -v)
        t2 = self.circle_intersect(c, r, self.p2, -v)
        if t1 == INF and t2 == INF:
            return INF, reflect(v, n)
        if t1 < t2:
            return t1, reflect(v, c+v*t1-self.p1)
        else:
            return t2, reflect(v, c+v*t2-self.p2)


# linear algebruh

try:
    import numpy as np
except ModuleNotFoundError:
    pass  # :(

def matvecmul(A, x):
    assert len(A[0]) == len(x)
    b = []
    for Ai in A:
        b.append(sum([Ai[j]*x[j] for j in range(len(x))]))
    return b
