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


# paddle collision and response

def generate_rect_walls(c, r):
    return [
        LineSegment(vec2(c.x-r.x,c.y-r.y), vec2(c.x-r.x,c.y+r.y)),
        LineSegment(vec2(c.x+r.x,c.y-r.y), vec2(c.x+r.x,c.y+r.y)),
        LineSegment(vec2(c.x-r.x,c.y-r.y), vec2(c.x+r.x,c.y-r.y)),
        LineSegment(vec2(c.x-r.x,c.y+r.y), vec2(c.x+r.x,c.y+r.y)),
    ]

def get_paddle_angle(center, radius, facing, y):
    rel_dist_from_c = ((y-center.y)/(2*radius.y))
    rel_dist_from_c = min(0.5, rel_dist_from_c)
    rel_dist_from_c = max(-0.5, rel_dist_from_c)
    sign = 1-2*facing
    return sign*rel_dist_from_c*45*pi/180

def get_paddle_collision_response(
        p_x, p_r, p_facing, b_x, v, b_r):
    theta = get_paddle_angle(p_x, p_r, p_facing, b_x.y)
    v /= 0.1
    v = vec2(cos(theta)*v.x-sin(theta)*v.y,
             sin(theta)*v.x+cos(theta)*v.y)
    v.x *= -1.0
    v = vec2(cos(-theta)*v.x-sin(-theta)*v.y,
             cos(-theta)*v.y+sin(-theta)*v.x)
    if v.x * (2*p_facing-1) < 1:
        v.y = (v.y/abs(v.y))*sqrt(max(v.x*v.x+v.y*v.y-1.0,0))
        v.x = 2.0*p_facing-1.0
    return v * 0.1 * 1.2


# main

def pong_ai(paddle_frect, other_paddle_frect, ball_frect, table_size):
    '''return "up" or "down", depending on which way the paddle should go to
    align its centre with the centre of the ball, assuming the ball will
    not be moving
    
    Arguments:
    paddle_frect: a rectangle representing the coordinates of the paddle
                  paddle_frect.pos[0], paddle_frect.pos[1] is the top-left
                  corner of the rectangle. 
                  paddle_frect.size[0], paddle_frect.size[1] are the dimensions
                  of the paddle along the x and y axis, respectively
    
    other_paddle_frect:
                  a rectangle representing the opponent paddle. It is formatted
                  in the same way as paddle_frect
    ball_frect:   a rectangle representing the ball. It is formatted in the 
                  same way as paddle_frect
    table_size:   table_size[0], table_size[1] are the dimensions of the table,
                  along the x and the y axis respectively
    
    The coordinates look as follows:
    
     0             x
     |------------->
     |
     |             
     |
 y   v
    '''

    # ball position
    ball_r = 0.5*vec2(ball_frect.size)
    ball_x = vec2(ball_frect.pos)+ball_r
    ball_r = sqrt(ball_r.x*ball_r.y)

    # paddles
    pr1 = 0.5*vec2(paddle_frect.size)
    pc1 = vec2(paddle_frect.pos)+pr1
    pr2 = 0.5*vec2(other_paddle_frect.size)
    pc2 = vec2(other_paddle_frect.pos)+pr2
    if pc1.x < pc2.x:
        fc1, fc2 = (1, 0)
    else:
        fc1, fc2 = (0, 1)

    # previous + velocity
    global prev_ball_x, prev_p1y, prev_p2y
    try:
        prev_ball_x
    except:
        prev_ball_x = ball_x
        prev_p1y = pc1.y
        prev_p2y = pc2.y
    ball_v = ball_x-prev_ball_x
    prev_ball_x = ball_x
    p1vy = pc1.y-prev_p1y
    prev_p1y = pc1.y
    p2vy = pc2.y-prev_p2y
    prev_p2y = pc2.y
    
    # default direction
    paddle_x = vec2(paddle_frect.pos) + 0.5 * vec2(paddle_frect.size)
    other_paddle_x = vec2(other_paddle_frect.pos) + 0.5*vec2(other_paddle_frect.size)
    if paddle_x.y < ball_x.y:
        default_dir = "down"
    else:
        default_dir = "up"
    #return default_dir

    if ball_v == vec2(0):
        return default_dir

    # tracing
    shift = 0.5  # positive when inward

    # paddles - bouncing is a bit troll?
    if pc1.x < pc2.x:
        pc1.x, pc2.x = pc1.x-shift, pc2.x+shift
    else:
        pc1.x, pc2.x = pc1.x+shift, pc2.x-shift
    def intersect_paddle(ro, rd, pc, pr, pfc):
        mint, minrefl = INF, vec2(1, 0)
        for wall in generate_rect_walls(pc, pr):
            try:
                t, refl = wall.intersect_circle(ro, ball_r, rd)
            except ZeroDivisionError:
                continue
            if t < mint:
                mint = t
                minrefl = get_paddle_collision_response(
                    pc, pr, pfc, ro+rd*t, rd, ball_r)
                #minrefl = refl
        return mint, minrefl
    
    # walls, simple bouncing
    if pc1.x < pc2.x:
        x0 = pc1.x+pr1.x
        x1 = pc2.x-pr2.x
    else:
        x0 = pc1.x-pr1.x
        x1 = pc2.x+pr2.x
    y0 = 0.0-shift
    y1 = table_size[1]+shift
    walls = [
        ('p1', LineSegment(vec2(x0, y0), vec2(x0, y1))),
        ('p2', LineSegment(vec2(x1, y0), vec2(x1, y1))),
        ('w', LineSegment(vec2(x0, y0), vec2(x1, y0))),
        ('w', LineSegment(vec2(x0, y1), vec2(x1, y1))),
    ]
    del x0, x1

    # ball tracing
    ro, rd = ball_x, ball_v
    hits = []  # ball location when hit, ball velocity before hit, estimate time
    tot_t = 0.0
    for rayi in range(40):
        ro = ro + 0.5 * rd
        minid, mint, minrefl = '', INF, vec2(1, 0)
        # walls
        for label, wall in walls:
            try:
                t, refl = wall.intersect_circle(ro, ball_r, rd)
            except ZeroDivisionError:
                continue
            if t < mint:
                if label.startswith('p'):
                    refl *= 1.2
                minid, mint, minrefl = label, t, refl
        # paddles
        t, refl = intersect_paddle(ro, rd, pc1, pr1, fc1)
        if t < mint+1e-4:
            minid, mint, minrefl = 'p1', t, refl
        t, refl = intersect_paddle(ro, rd, pc2, pr2, fc2)
        if t < mint+1e-4:
            minid, mint, minrefl = 'p2', t, refl
        # data
        params = [
                pc1.x, pc1.y, pc2.x, pc2.y, p1vy, p2vy,
                ball_x.x, ball_x.y, ball_v.x, ball_v.y,
                mint, minrefl.x, minrefl.y
            ]
        # update
        ro += rd * mint
        prev_rd, rd = rd, minrefl
        tot_t += mint
        if minid == 'p1':
            hits.append([ro, prev_rd, tot_t])
            if len(hits) >= 2:
                break
    if len(hits) == 0:
        return default_dir

    # where to go for the next hit so you:
    #   - won't miss the second hit
    #   - give your opponent a hard time

    # time constraint

    # suppose you go to y and the hits are (t1, y1), (t2, y2)
    # your speed is 1
    safe_dy = max(pr1.y-0.5*ball_r, 0)
    max_dy = pr1.y + 0.707*ball_r *0.0
    # to catch the hit, you go anywhere between y1-safe_dy and y1+safe_dy
    # to catch the second hit, |y-y2|/(t2-t1) < max_dy
    y1, t1 = hits[0][0].y, hits[0][2]
    y_range = [y1-safe_dy, y1+safe_dy]
    if len(hits) >= 2:
        y2, t2 = hits[1][0].y, hits[1][2]
        y2r = (t2-t1)*max_dy
        if y2 < y1:
            y_range[1] = min(y_range[1], y2+y2r)
        else:
            y_range[0] = max(y_range[0], y2-y2r)
    if y_range[1] < y_range[0]:
        y_range = [y1-safe_dy, y1+safe_dy]
    y_range[0] = max(y_range[0], 0)
    y_range[1] = min(y_range[1], table_size[1])
    #if min(abs(pc1.y-y_range[0]), abs(pc1.y-y_range[1])) > t1+ball_r:
    #    print("Oh no!", t1)

    # give your opponent a hard time
    opp_y = pc2.y
    target_y = 0.5*(y_range[0]+y_range[1])
    worst_opp = 0.0
    numchecks = 5
    for check_i in range(numchecks):
        # get initial hit
        test_y = lerp(y_range[0], y_range[1], check_i/(numchecks-1))
        test_y = round(test_y)
        ro = hits[0][0]
        rd = hits[0][1]
        t, rd = intersect_paddle(ro-0.5*rd, rd,
                                 vec2(pc1.x, test_y), pr1, fc1)
        # trace
        for rayi in range(10):
            ro = ro + 0.5 * rd
            minid, mint, minrefl = '', INF, vec2(1, 0)
            # walls
            for label, wall in walls:
                try:
                    t, refl = wall.intersect_circle(ro, ball_r, rd)
                except ZeroDivisionError:
                    continue
                if t < mint:
                    if label.startswith('p'):
                        refl *= 1.2
                    minid, mint, minrefl = label, t, refl
            ro += rd * mint
            prev_rd, rd = rd, minrefl
            tot_t += mint
            if minid == 'p2':
                opp_dist = abs(ro.y - opp_y)
                if opp_dist > worst_opp:
                    target_y = test_y
                    worst_opp = opp_dist
                break

    yc = 0.5*table_size[1]
    if ball_v.x * (pc1.x-0.5*table_size[0]) < 0.0 and (False\
         or min(abs(pc1.y-y_range[0]), abs(pc1.y-y_range[1])) > t1+ball_r \
         or abs(yc-hits[0][0].y)+ball_r < 0.8*t1
        ):
        target_y = yc

    return 'down' if paddle_x.y < target_y else 'up'

    return default_dir



# please don't reveal our real names on tournament thanks
pong_ai.team_name = [
    "Flying Jellyfish",
    "Mathy Nautilus",
    "Radioactive Coral",
    "Parametric Starfish",
    "Isotropic Crab",
    "Sinusoidal Mussel"
][__import__('random').randint(0, 5)]
#print(pong_ai.team_name)

