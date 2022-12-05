from shared import *

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


# debug
DEBUG = False  # 🤬
paths = []
killpaths = []
hits = []
if DEBUG:
    dump = []  # p1x, p1y, p2x, p2y, p1vy, p2vy, bx, by, vx, vy, pred_t, prev_vx, prev_vy


# linear algebruh
# time is good, reflection is so bad
# I added rebounce after training this so it shouldn't be used now

def matvecmul(A, x):
    assert len(A[0]) == len(x)
    b = []
    for Ai in A:
        b.append(sum([Ai[j]*x[j] for j in range(len(x))]))
    return b

def relu(x):
    return [max(xi,0.0) for xi in x]

def refine_tv(params):
    W1 = [[0.018009327352046967, 0.009860044345259666, 0.03571648895740509, 0.04400981590151787, -0.05730670690536499, -0.3765850067138672, 0.10408633947372437, -0.1374954879283905, -0.240605890750885, 0.44493526220321655, 1.0126041173934937, 1.1203464269638062, -0.7077003717422485, 0.11097811907529831], [-0.10002420842647552, 0.1624055802822113, -0.015184879302978516, -0.1850682497024536, 0.2638918161392212, 0.1007724180817604, -0.10368762910366058, -0.013510343618690968, -0.04207005351781845, 0.058117467910051346, -0.10721270740032196, -0.0719519853591919, -0.26564693450927734, 0.008220215328037739], [0.0354294627904892, -0.030931983143091202, 0.0361412912607193, -0.035262059420347214, -0.0010997809004038572, 0.12827646732330322, -0.005571594461798668, 0.28037774562835693, -0.39407843351364136, 0.16521884500980377, 0.6682811379432678, 0.19024278223514557, 0.6887965202331543, -0.016882406547665596], [0.14890870451927185, -0.05706576630473137, 0.1755916029214859, 0.034269627183675766, 0.12000344693660736, 0.0840211734175682, -0.04819149523973465, -0.0304261464625597, -0.04689154028892517, 0.5332130789756775, 0.20821529626846313, 0.11842247098684311, -0.9902079105377197, -0.08033357560634613], [0.07023318111896515, 0.1608934849500656, 0.11748813092708588, 0.05222785845398903, 0.035272736102342606, -0.1740342229604721, 0.00781085342168808, 0.015560545027256012, -0.15259014070034027, 0.1783895641565323, 0.234432190656662, 2.909034490585327, -0.10555233806371689, -0.031298473477363586], [0.15235966444015503, -0.2568410336971283, 0.16880960762500763, 0.048493001610040665, 0.023971877992153168, -0.1267653852701187, 0.10537604987621307, 0.04235472157597542, -0.09464969485998154, -0.25288066267967224, -0.15196572244167328, -0.4098436236381531, 0.4385785460472107, 0.24858121573925018], [0.06830452382564545, 0.2773061692714691, 0.15749263763427734, 0.1330544799566269, 0.14415766298770905, 0.08416953682899475, 0.04990510642528534, -0.1258068084716797, -0.09768930077552795, -0.5713651180267334, 0.012949934229254723, -1.0614567995071411, 1.2220779657363892, 0.152230903506279], [-0.044875919818878174, 0.23795828223228455, 0.00526553625240922, 0.04019874706864357, -0.19641591608524323, -0.2503550946712494, 0.10718482732772827, 0.21454396843910217, -0.1579299122095108, 0.28388047218322754, -0.11881942301988602, -0.38272738456726074, -0.4839995205402374, 0.15506772696971893]]
    W2 = [[0.15626150369644165, 0.03624429181218147, 0.24261781573295593, -0.253577321767807, 0.463776558637619, 0.006333630066365004, -0.042032141238451004, -0.30030155181884766, 0.0049551320262253284], [0.2904116213321686, -0.34622660279273987, -0.20221132040023804, -0.33604076504707336, 0.03516446426510811, -0.04347337409853935, -0.33459344506263733, -0.20287224650382996, 0.1250324249267578], [-0.25321847200393677, -0.11015867441892624, -0.3793790340423584, 0.07723072916269302, -0.17759239673614502, 0.1873258799314499, 0.02864324487745762, -0.40161067247390747, -0.06767303496599197], [0.27114784717559814, -0.008542532101273537, -0.13054151833057404, 0.41859158873558044, 0.2249428927898407, -0.15681903064250946, -0.40734949707984924, 0.21654988825321198, -0.26843738555908203], [-0.13594667613506317, 0.19291773438453674, -0.31049203872680664, 0.2694686949253082, -0.28966888785362244, -0.2038547396659851, -0.16069918870925903, 0.022489845752716064, 0.23868200182914734], [-0.2819569408893585, -0.32279059290885925, 0.06753885746002197, -0.5146036744117737, -0.0003473401302471757, 0.20984527468681335, 0.2844017744064331, -0.21701183915138245, 0.03985549882054329], [0.18894638121128082, -0.219024196267128, -0.2833828330039978, -0.2957748472690582, -0.30057644844055176, -0.034106552600860596, -0.29745596647262573, -0.3075079917907715, -0.16903099417686462], [0.54311203956604, -0.1555502861738205, 0.6331169605255127, 0.18925176560878754, -0.5985279679298401, -0.2766903042793274, 0.2973949909210205, -0.15132789313793182, -0.2092406004667282]]
    W3 = [[0.3056219220161438, -0.09610988199710846, -0.03640097379684448, 0.04596646502614021, -0.22045676410198212, -0.32869723439216614, -0.20415355265140533, 0.8617072105407715, -0.78655606508255], [0.33298468589782715, 0.16978353261947632, 0.12450443208217621, 0.04539338871836662, 0.10156512260437012, -0.0797310471534729, 0.11668439954519272, -0.167338028550148, -1.9621198177337646], [0.3311026394367218, 0.0035577043890953064, -0.18381594121456146, -0.41074439883232117, -0.28998875617980957, 0.4605056643486023, 0.23143258690834045, -0.006304717622697353, 0.13073647022247314]]
    y = relu(matvecmul(W1, params+[1.0]))
    y = relu(matvecmul(W2, y+[1.0]))
    y = matvecmul(W3, y+[1.0])
    return y[0], vec2(y[1:3])


# main
import pong_ai_221204
def pong_ai(paddle_frect, other_paddle_frect, ball_frect, table_size):
    return pong_ai_221204.pong_ai(paddle_frect, other_paddle_frect, ball_frect, table_size)
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
    assert max(pr1)>0 and max(pr2)>0

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

    # paddles - bouncing is a bit troll?
    if pc1.x < pc2.x:
        pc1.x, pc2.x = pc1.x, pc2.x
    else:
        pc1.x, pc2.x = pc1.x, pc2.x
    def intersect_paddle(ro, rd, pc, pr, pfc):
        pr = pr * vec2(1, 100) ##
        mint, minrefl = INF, vec2(1, 0)
        if (pfc == 1 and ro.x < pc.x+pr.x) or \
            (pfc == 0 and ro.x > pc.x-pr.x):
                return mint, minrefl
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
    y0 = 0.0
    y1 = table_size[1]
    walls = [
        ('p1', LineSegment(vec2(x0, y0), vec2(x0, y1))),
        ('p2', LineSegment(vec2(x1, y0), vec2(x1, y1))),
        ('w', LineSegment(vec2(x0, y0), vec2(x1, y0))),
        ('w', LineSegment(vec2(x0, y1), vec2(x1, y1))),
    ]
    del x0, x1

    # ball tracing
    global paths, hits
    def trace_ball(ro, rd, maxrayi, record_idhit, exit_numhit, paths=None):
        if paths is None:
            paths = []
        else:
            while len(paths) > 0:
                del paths[-1]
        hits = []  # ball location when hit, ball velocity before hit, estimate time
        tot_t = 0.0
        for rayi in range(maxrayi):
            ro = ro + 1e-2 * rd
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
            # update
            ro += rd * mint
            prev_rd, rd = rd, minrefl
            tot_t += mint
            paths.append(ro)
            if minid == record_idhit:
                hits.append([ro, prev_rd, tot_t])
                if len(hits) >= exit_numhit:
                    break
        return hits
    hits = trace_ball(ball_x, ball_v, 40, 'p1', 2, paths)
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
    y_range[0] = max(y_range[0], 1)
    y_range[1] = min(y_range[1], table_size[1]-1)
    target_y = y_range[0]
    yc = 0.5*table_size[1]
    if abs(y_range[1]-yc) < abs(target_y-yc):
        target_y = y_range[1]
    if y_range[0] < yc < target_y:
        target_y = yc

    # give your opponent a hard time
    global killpaths
    opp_y = pc2.y
    target_y_kill = 0.5*(y_range[0]+y_range[1])
    worst_opp, worst_opp_t = 0.0, 0.0
    numchecks = 5
    for check_i in range(numchecks):
        # get initial hit
        test_y = lerp(y_range[0], y_range[1], check_i/(numchecks-1))
        test_y = pc1.y + round(test_y-pc1.y)
        ro = hits[0][0]
        rd = hits[0][1]
        t, rd = intersect_paddle(ro-0.5*rd, rd,
                                 vec2(pc1.x, test_y), pr1, fc1)
        # trace
        killpath = [ro]
        tot_t = t1
        for rayi in range(10):
            ro = ro + 1e-2 * rd
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
            killpath.append(ro)
            if minid == 'p2':
                opp_dist = abs(ro.y - opp_y)
                if opp_dist > worst_opp:
                    target_y_kill = test_y
                    worst_opp, worst_opp_t = opp_dist, tot_t
                    killpaths = killpath
                break
    if worst_opp < worst_opp_t+ball_r:  # can kill the opponent
        target_y = target_y_kill

    # if you are guaranteed to win or lose, go to middle
    if min(abs(pc1.y-y_range[0]), abs(pc1.y-y_range[1])) > 1.2*t1+2.0*ball_r:
        target_y = 0.5*table_size[1]

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
