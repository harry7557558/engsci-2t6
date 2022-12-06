from shared import *

# paddle collision and response

def generate_rect_walls(c, r):
    return [
        LineSegment(vec2(c.x-r.x,c.y-r.y), vec2(c.x-r.x,c.y+r.y)),
        LineSegment(vec2(c.x+r.x,c.y-r.y), vec2(c.x+r.x,c.y+r.y)),
        LineSegment(vec2(c.x-r.x,c.y-r.y), vec2(c.x+r.x,c.y-r.y)),
        LineSegment(vec2(c.x-r.x,c.y+r.y), vec2(c.x+r.x,c.y+r.y)),
    ]

def get_paddle_angle(center_y, radius_y, facing, y):
    rel_dist_from_c = ((y-center_y)/(2*radius_y))
    rel_dist_from_c = min(0.5, rel_dist_from_c)
    rel_dist_from_c = max(-0.5, rel_dist_from_c)
    sign = 1-2*facing
    return sign*rel_dist_from_c*45*pi/180

def get_paddle_collision_response(
        p_x, p_r, p_facing, b_x, v, b_r):
    theta = get_paddle_angle(p_x.y, p_r.y, p_facing, b_x.y)
    v /= 0.1
    v = vec2(cos(theta)*v.x-sin(theta)*v.y,
             sin(theta)*v.x+cos(theta)*v.y)
    v.x *= -1.0
    v = vec2(cos(-theta)*v.x-sin(-theta)*v.y,
             cos(-theta)*v.y+sin(-theta)*v.x)
    if v.x * (2*p_facing-1) < 1:
        v.y = (v.y/abs(v.y))*sqrt(max(v.x*v.x+v.y*v.y-1.0,0))
        v.x = 2.0*p_facing-1.0
    return v * 0.1 * 1.2, b_x

def get_paddle_collision_response_accurate(
    p_x, p_r, p_facing, b_x, v, b_r):
    c = 0
    while abs(p_x.x-b_x.x) < p_r.x+b_r:
        b_x = b_x - 0.1*v.normalize()
        c += 1
    theta = get_paddle_angle(p_x.y, p_r.y, p_facing, b_x.y)
    v = vec2(cos(theta)*v.x-sin(theta)*v.y,
             sin(theta)*v.x+cos(theta)*v.y)
    v.x *= -1.0
    v = vec2(cos(-theta)*v.x-sin(-theta)*v.y,
             cos(-theta)*v.y+sin(-theta)*v.x)
    if v.x * (2*p_facing-1) < 1:
        v.y = (v.y/abs(v.y))*sqrt(max(v.x*v.x+v.y*v.y-1.0,0))
        v.x = 2.0*p_facing-1.0
    v *= 1.2
    while c > 0 and abs(p_x.x-b_x.x) < p_r.x+b_r:
        b_x = b_x + 0.12*v.normalize()
        c -= 1
    return v, b_x


# the sus part

def relu(x):
    return [max(xi,0.0) for xi in x]

def predict_move(params):
    W1 = [[0.002801, -0.023366, 0.001654, -0.020931, -0.002707, -0.02604, 0.010143, -0.003676, 0.005629, -0.005474, -0.003572, 0.009325, -0.018912, -0.02268, 0.011455, -0.017545, -0.006782, -0.01806, -0.004477, 0.011328, -0.032861, -0.015602, -0.003029, 0.002147, -0.004739, 0.004449, -0.020038, -0.003338, -0.02182, -0.024638, -0.043571, -0.016943, 0.99639, -0.004148], [-0.003415, 0.000624, -0.074878, 0.071383, -0.037166, 0.043042, -0.213639, 0.017183, -0.009182, 0.262439, 0.00011, -0.052401, -0.026509, 0.015399, -0.023103, -0.094079, 0.014956, -0.016267, -0.023089, 0.016782, -0.090945, -0.007287, 0.003387, 0.039441, -0.018309, -0.13831, -0.023954, 0.016802, -0.003213, 0.002026, -0.010094, -0.007208, -0.333331, 0.012063], [-0.023874, -0.014913, -0.061237, 0.050737, 0.018718, 0.000417, 0.010112, -0.006036, -0.011232, 0.006504, -0.013658, -0.063906, -0.012571, 0.001904, -0.086139, 0.321418, -0.035053, -0.004203, 0.00256, -0.023701, -0.093074, -0.05001, 0.008015, -0.020856, 0.024679, -0.056208, -0.03447, -0.004303, -0.000175, 0.062685, 0.084932, -0.045615, -0.082642, -0.007981], [-0.000177, -0.036365, -0.108822, 0.002085, -0.017226, -0.024412, 0.386565, -0.008219, 0.002882, -0.308511, 0.003411, -0.082089, -0.000356, 0.032697, 0.031443, 0.274466, 0.022909, -0.009572, 0.032666, -0.064312, -0.068401, 0.021369, -0.002688, -0.060713, 0.093643, 0.269462, 0.003586, 0.003964, 0.019914, 0.063108, -0.016486, 0.015844, 0.588987, -0.004464], [-0.011669, -0.011621, -0.009897, 0.013196, -0.011924, 0.000608, 0.013804, -0.009052, -0.02553, -0.007869, -0.012749, -0.012147, 0.000792, -0.017538, -0.001466, -0.008737, 0.008426, -0.022092, -0.029906, 0.002111, 0.009716, -0.009461, -0.001897, -0.006844, -0.004137, -0.015265, -0.006539, -0.016751, 0.000729, -0.015404, 0.003347, -0.014216, 0.003731, -0.005161], [-0.021663, 0.062092, -0.25539, 0.086498, -0.027075, 0.018364, 0.801454, -0.031269, 0.087959, -0.256408, -0.005471, -0.264019, -0.025583, 0.160227, 0.092917, 0.39062, -0.047882, -0.024131, 0.030976, 0.05813, -0.151926, 0.01585, -0.017762, 0.061717, 0.234671, 0.019643, -0.084258, -0.033813, 0.00611, -0.003485, -0.065387, 0.034917, 0.997169, -0.030631], [-0.018696, -0.010181, -0.007472, -0.001111, -0.009107, -0.01536, -0.003807, -0.00157, -0.012175, 0.014366, 0.013447, 0.006986, -0.008476, -0.012724, -0.0141, 0.007359, 0.002168, -0.010264, -0.008391, 0.0118, -0.017907, -0.008043, -0.022743, -0.023389, -0.005696, 0.004697, -0.025685, -0.009302, -9.9e-05, 0.014727, 0.001438, 0.002491, -0.001813, -0.008252], [0.030157, 0.036707, -0.044259, 0.03692, -0.01071, -0.052406, 0.677507, -0.007557, 0.113021, -0.344438, 0.019531, -0.096007, -0.033326, 0.097961, 0.025575, 0.474802, -0.081113, -0.023809, -0.011044, -0.116476, -0.099427, 0.017901, -0.047462, 0.041754, 0.217131, 0.002099, 0.029301, 0.004147, 0.020851, 0.112553, -0.043946, -0.000572, 0.904407, -0.028759], [-0.008639, 0.0151, 0.008746, -0.009815, -0.001431, 0.00718, 0.015149, -0.027725, -0.00657, 0.019813, 0.008547, -0.003015, -0.023684, 0.00075, -0.004538, 0.020223, 0.005486, -0.005439, 0.009494, 0.008552, -0.003087, -0.021705, 0.003001, 0.007282, -0.007897, 0.005467, -0.021987, -0.003674, -0.026745, 0.002186, -0.002948, -0.017707, -0.002317, -0.010224], [0.00503, 0.017279, 0.159324, 0.06162, -0.00184, 0.006124, 0.064992, 0.002669, -0.015999, 0.066949, 7.7e-05, -0.011313, -0.014884, 0.016182, -0.00508, -0.024956, -0.008565, -0.008182, -0.032208, -0.053052, -0.052815, -0.007568, -0.009559, 0.02095, 0.094694, 0.091059, -0.030373, -0.008442, -0.033363, 0.050533, 0.083565, -0.020087, -0.007605, 0.010946], [0.005657, -0.003433, -0.064015, -0.001225, -0.028742, -0.040164, 0.577744, -0.000666, 0.052299, -0.388372, 0.026985, -0.088208, -0.037441, 0.004648, -0.008913, 0.347031, -0.037229, -0.039634, 0.042517, -0.064134, -0.052278, -0.002168, 0.007732, -0.056439, 0.060053, 0.163726, 0.014474, 0.017131, 0.052698, 0.023325, 0.013517, 0.006624, 0.778437, -0.010889], [0.016437, 0.046753, 0.08175, 0.006117, 0.002076, 0.067553, -0.378737, 0.015571, 0.021783, 0.224965, 0.010397, 0.22805, 0.005562, -0.078976, -0.101155, 0.021224, -0.057002, 0.009007, 0.020716, 0.019041, 0.014891, -0.026138, 0.009159, 0.021078, -0.094661, -0.058829, 0.001096, 0.015001, 0.048824, -0.039937, 0.014024, -0.034354, -0.347525, 0.005906]]
    W2 = [[0.989846, -0.019164, -0.007918, -0.020411, 0.018487, -0.007137, -0.007595, -0.01145, 0.004533, -0.0148, -0.012813, -0.027148, -0.017286], [0.011188, 0.026242, 0.062571, -0.059026, -0.016341, -0.087245, 0.011316, -0.102394, -0.008486, -0.036892, -0.015551, 0.066317, 0.062823], [-0.01163, -0.024485, -0.00766, -0.004042, 0.018357, -0.004296, 0.008063, -0.014299, 0.000217, 0.005318, -0.03066, -0.005646, -0.018388], [0.007674, 0.030166, 0.036952, -0.025802, 0.004175, -0.108398, 0.005655, 0.018414, 0.013735, -0.012615, 0.004524, 0.033223, 0.055798], [-0.007069, -0.017328, -0.210264, 0.016849, 0.005489, 0.083978, 0.000144, 0.052749, -0.001605, -0.013889, 0.044327, 0.001002, -0.03125], [0.019794, 0.002681, 0.035614, -0.018205, -0.011107, -0.0733, 0.001452, -0.051678, 0.011183, -0.018948, -0.017091, 0.058293, 0.068873], [0.013322, -0.013306, 0.003757, 0.014845, 0.005126, -0.007729, -0.000321, -0.028528, -0.003491, -0.027721, -0.020751, -0.007943, -0.038359], [0.018667, 0.052365, 0.067939, -0.015935, -0.014316, 0.008689, -0.004299, 0.008118, -0.002836, 0.001728, -0.023515, -0.011794, 0.011069]]
    W3 = [[0.989962, -0.095015, -0.002054, -0.051441, 0.041075, -0.050036, -0.006386, -0.009448, -0.076831]]
    y = relu(matvecmul(W1, params+[1.0]))
    y = relu(matvecmul(W2, y+[1.0]))
    y = matvecmul(W3, y+[1.0])
    return y[0]



class ParameterGenerator:
    def __init__(self):
        self.paths = []
        self.parameter_log = []
        # ball
        self.ball_r = None
        self.prev_ball_x = None
        self.ball_x = None
        self.ball_v = None
        # paddles
        self.prev_p1y = None
        self.prev_p2y = None
        self.pc1, self.pr1, self.fc1 = [None]*3  # this paddle
        self.pc2, self.pr2, self.fc2 = [None]*3  # the opponent's paddle
        
    def intersect_paddle(self, ro, rd, pc, pr, pfc):
        pr = pr * vec2(1, 100) ##
        mint, minrefl, minro = INF, vec2(1, 0), vec2(0)
        if (pfc == 1 and ro.x < pc.x+pr.x) or \
            (pfc == 0 and ro.x > pc.x-pr.x):
                return mint, minrefl
        for wall in generate_rect_walls(pc, pr):
            try:
                t, refl = wall.intersect_circle(ro, self.ball_r, rd)
            except ZeroDivisionError:
                continue
            if t < mint:
                mint = t
                minrefl, minro = get_paddle_collision_response(
                    pc, pr, pfc, ro+rd*t, rd, self.ball_r)
                #minrefl = refl
        return mint+1, minrefl, minro

    def trace_ball(self, ro, rd, maxrayi):
        self.paths = [ro]
        hits = []  # ball location when hit, ball velocity before hit, estimate time
        hits_other = []
        tot_t = 0.0
        for rayi in range(maxrayi):
            ro = ro + 1e-2 * rd
            minid, mint, minrefl, minro = '', INF, vec2(1, 0), vec2(0)
            # walls
            for label, wall in self.walls:
                try:
                    t, refl = wall.intersect_circle(ro, self.ball_r, rd)
                except ZeroDivisionError:
                    continue
                if t < mint:
                    minro = ro+rd*t
                    if label.startswith('p'):
                        if label == 'p1':
                            p_x = self.pc1
                            p_r = self.pr1
                            pfc = self.fc1
                        else:
                            p_x = self.pc2
                            p_r = self.pr2
                            pfc = self.fc2
                        refl, minro = get_paddle_collision_response(
                            p_x, p_r, pfc, ro+rd*t, rd, self.ball_r)
                        refl *= 1.2
                    minid, mint, minrefl = label, t, refl
            # update
            ro = minro
            prev_rd, rd = rd, minrefl
            tot_t += mint
            self.paths.append(ro)
            if minid == 'p1':
                hits.append([ro, prev_rd, tot_t])
            if minid == 'p2':
                hits_other.append([ro, prev_rd, tot_t])
            if len(hits) >= 2 and len(hits_other) >= 2:
                break
        return hits, hits_other

    def get_action(self,
                   paddle_frect, other_paddle_frect, ball_frect, table_size):
        
        # ball position
        self.ball_r = 0.5*vec2(ball_frect.size)
        self.ball_x = vec2(ball_frect.pos)+self.ball_r
        self.ball_r = sqrt(self.ball_r.x*self.ball_r.y)

        # paddles
        self.pr1 = 0.5*vec2(paddle_frect.size)
        self.pc1 = vec2(paddle_frect.pos)+self.pr1
        self.pr2 = 0.5*vec2(other_paddle_frect.size)
        self.pc2 = vec2(other_paddle_frect.pos)+self.pr2
        if self.pc1.x < self.pc2.x:
            self.fc1, self.fc2 = (1, 0)
        else:
            self.fc1, self.fc2 = (0, 1)

        # previous + velocity
        if self.prev_ball_x is None:
            self.prev_ball_x = self.ball_x
            self.prev_p1y = self.pc1.y
            self.prev_p2y = self.pc2.y
        self.ball_v = self.ball_x-self.prev_ball_x
        self.prev_ball_x = self.ball_x
        self.p1vy = self.pc1.y-self.prev_p1y
        self.prev_p1y = self.pc1.y
        self.p2vy = self.pc2.y-self.prev_p2y
        self.prev_p2y = self.pc2.y
        
        # default direction
        if self.pc1.y < self.ball_x.y:
            default_dir = "down"
        else:
            default_dir = "up"
        if self.ball_v == vec2(0):
            return default_dir

        # walls, simple bouncing
        if self.pc1.x < self.pc2.x:
            x0 = self.pc1.x+self.pr1.x
            x1 = self.pc2.x-self.pr2.x
        else:
            x0 = self.pc1.x-self.pr1.x
            x1 = self.pc2.x+self.pr2.x
        y0 = 0.0
        y1 = table_size[1]
        self.walls = [
            ('p1', LineSegment(vec2(x0, y0), vec2(x0, y1))),
            ('p2', LineSegment(vec2(x1, y0), vec2(x1, y1))),
            ('w', LineSegment(vec2(x0, y0), vec2(x1, y0))),
            ('w', LineSegment(vec2(x0, y1), vec2(x1, y1))),
        ]
        del x0, x1

        # ball tracing
        hits1, hits2 = self.trace_ball(self.ball_x, self.ball_v, 25)
        if len(hits1) == 0:
            return default_dir
        action = strategy_1(self, hits1, table_size)
        if not (len(hits1) == 2 and len(hits2) == 2):
            return action
        params = [
            self.ball_x.x, self.ball_x.y, self.ball_v.x, self.ball_v.y,
            self.pc1.x, self.pc1.y, self.p1vy, self.pc2.x, self.pc2.y, self.p2vy,
            1.0 if self.pc1.x < self.pc2.x else -1.0,
            1.0 if hits1[0][2] < hits2[0][2] else -1.0,
            hits1[0][0].x, hits1[0][0].y, hits1[0][1].x, hits1[0][1].y, hits1[0][2],
            hits1[1][0].x, hits1[1][0].y, hits1[1][1].x, hits1[1][1].y, hits1[1][2],
            hits2[0][0].x, hits2[0][0].y, hits2[0][1].x, hits2[0][1].y, hits2[0][2],
            hits2[1][0].x, hits2[1][0].y, hits2[1][1].x, hits2[1][1].y, hits2[1][2],
            1.0 if action == 'up' else -1.0
        ]
        assert len(params) == 33
        #print(params[-1], predict_move(params))
        #return 'up' if predict_move(params) > 0 else 'down'
        #self.parameter_log.append(params)
        return action


def strategy_1(pg, hits, table_size):
    
    pr1, pc1 = pg.pr1, pg.pc1
    pr2, pc2 = pg.pr2, pg.pc2
    ball_x, ball_v, ball_r = pg.ball_x, pg.ball_v, pg.ball_r

    # time constraint

    # suppose you go to y and the hits are (t1, y1), (t2, y2)
    # your speed is 1
    safe_dy = max(pr1.y-0.5*ball_r, 0)
    max_dy = pr1.y + 0.707*ball_r *0.0
    # to catch the hit, you go anywhere between y1-safe_dy and y1+safe_dy
    # to catch the second hit, |y-y2|/(t2-t1) < max_dy
    y1, t1 = hits[0][0].y, hits[0][2]
    global y_range
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
    margin = 0.5*pr1.y+0.5*ball_r
    y_range[0] = max(y_range[0], margin)
    y_range[1] = min(y_range[1], table_size[1]-margin)
    target_y = y_range[0]
    yc = 0.5*table_size[1]
    if abs(y_range[1]-yc) < abs(target_y-yc):
        target_y = y_range[1]
    if y_range[0] < yc < target_y:
        target_y = yc

    # give your opponent a hard time
    opp_y = pc2.y
    target_y = 0.5*(y_range[0]+y_range[1])
    worst_opp, worst_opp_t = 0.0, 0.0
    numchecks = 10
    for check_i in range(numchecks):
        # get initial hit
        test_y = lerp(y_range[0], y_range[1], check_i/(numchecks-1))
        test_y = pc1.y + round(test_y-pc1.y)
        ro = hits[0][0]
        rd = hits[0][1]
        t, rd, ro = pg.intersect_paddle(ro-0.5*rd, rd,
            vec2(pg.pc1.x, test_y), pg.pr1, pg.fc1)
        # trace
        tot_t = t1
        for rayi in range(16):
            ro = ro + 1e-2 * rd
            minid, mint, minrefl, minro = '', INF, vec2(1, 0), vec2(0)
            # walls
            for label, wall in pg.walls:
                try:
                    t, refl = wall.intersect_circle(ro, ball_r, rd)
                except ZeroDivisionError:
                    continue
                if t < mint:
                    minro = ro+rd*t
                    if label.startswith('p'):
                        if label == 'p1':
                            p_x, p_r = pc1, pr1
                            pfc = pg.fc1
                        else:
                            p_x, p_r = pc2, pr2
                            pfc = pg.fc2
                        refl, minro = get_paddle_collision_response(
                            p_x, p_r, pfc, ro+rd*t, rd, ball_r)
                        refl *= 1.2
                    minid, mint, minrefl = label, t, refl
            ro = minro
            prev_rd, rd = rd, minrefl
            tot_t += mint
            if minid == 'p2':
                opp_dist = abs(ro.y - opp_y)
                if opp_dist > worst_opp:
                    target_y = test_y
                    worst_opp, worst_opp_t = opp_dist, tot_t
                break
    if worst_opp > worst_opp_t+1.5*ball_r and False:  # can't kill the opponent
        target_y = y_range[0]
        yc = 0.5*table_size[1]
        if abs(y_range[1]-yc) < abs(target_y-yc):
            target_y = y_range[1]
        if y_range[0] < yc < target_y:
            target_y = yc

    # if you are guaranteed to win or lose
    # or if you can safely catch the ball
    # and the ball is moving away, go to middle
    yc = 0.5*table_size[1]
    if ball_v.x * (pc1.x-0.5*table_size[0]) < 0.0 and (False\
         or min(abs(pc1.y-y_range[0]), abs(pc1.y-y_range[1])) > t1+ball_r \
         #or worst_opp < worst_opp_t  # hard to predict accurately
         or abs(yc-hits[0][0].y)+ball_r < 0.8*t1
         #or True
        ):
        #print("middle", t1)
        target_y = yc

    return 'down' if pc1.y < target_y-0.1 else 'up' if pc1.y > target_y+0.1 else None


pg = ParameterGenerator()
pg_other = ParameterGenerator()
TRAINING = False


# main
import pong_ai_221204
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
    return pong_ai_221204.pong_ai(paddle_frect, other_paddle_frect, ball_frect, table_size)

    if TRAINING:
        pg_other.get_action(other_paddle_frect, paddle_frect, ball_frect, table_size)

    return pg.get_action(paddle_frect, other_paddle_frect, ball_frect, table_size)



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
