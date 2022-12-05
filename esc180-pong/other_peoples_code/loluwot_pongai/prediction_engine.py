from math import ceil
from .utils import *

def predict(ball_pos, ball_vel, table_size, paddle_x):
    # print(ball_pos, ball_vel, table_size, paddle_x)
    if ball_vel[0] == 0 or ball_vel[1] == 0:
        print('ZERO VEL',ball_vel)
        # input()
        return None
    t = int((paddle_x - ball_pos[0])/ball_vel[0])
    dy = ball_vel[1]*t
    y = ball_pos[1] + dy
    # print(y)
    init_modulus = table_size[1] - 15
    # print(init_modulus)
    n_bounces = abs(y - (y % init_modulus))//(init_modulus)# + 1 if y < 0 else 0
    # dy = ball_vel[1]*t
    # y = ball_pos[1] + dy
    bounce_times = []
    goal = sign(ball_vel[1])*init_modulus*(1 if ball_vel[1] > 0 else 0)
    bounce_times.append(abs((goal - ball_pos[1])/ball_vel[1]))
    # print(bounce_times)
    y = abs(((y + init_modulus) % (init_modulus*2)) - init_modulus)
    # print('Time', t, 'vel', ball_vel[0], 'pos', ball_pos[0], 'pred', y+15/2)
    
    return y+15/2, t, n_bounces, bounce_times