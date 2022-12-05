import time, math, sys
#initial goal
from prediction_engine import predict
from utils import *
best_loc = 140 - 36
goal = best_loc

#global variables---------
default_goal = 140
MAX_VELS = 1
THRESHOLD = 1.2
max_angle = 45*math.pi/180
count = 0
idx = 0

#changeables--------
vels = []
continue_aiming = True
prediction = None
last_pos = None
ball_vel = None
ticks_prediction = 0
keep_predicting = False
last_collided = None
tot_bounces = None
predicted_vel = None
prediction_time = None
tot_prediction_time = None
region = None
target = None
temp_counter = 10
# ts = None

#debug metrics--------
accuracy = 0
accuracy_count = 0
error_count = 0

def pong_ai(paddle_frect, other_paddle_frect, ball_frect, table_size, ts):
    try:
        global last_pos, goal, ball_vel, prediction, ticks_prediction, keep_predicting, last_collided, predicted_vel, prediction_time, count, region, tot_prediction_time, tot_bounces
        global idx, target, temp_counter, accuracy, accuracy_count, error_count
        global vels, continue_aiming
        
        center_self = center(paddle_frect)
        center_other = center(other_paddle_frect)
        ball_center = center(ball_frect)
        init_time = time.time()
        if ball_center[0] == table_size[0]/2 and ball_center[1] == table_size[1]/2:
            ball_vel = None
            prediction = None
            last_pos = None
            ticks_prediction = 10
            keep_predicting = True
            last_collided = None
            predicted_vel = None
            prediction_time = None
            tot_bounces = None
            count = 0
            tot_prediction_time = None
            region = 0
            idx = 0
            target = None
            temp_counter = 10


        if last_pos:
            vel = [ball_frect.pos[i] - last_pos[i] for i in range(2)]
            ball_vel = vel[::]
            vels.append(ball_vel)
            vels = vels[max(0, len(vels) - 2):]
        last_pos = ball_frect.pos

        paddle_x1 = padx(paddle_frect.pos[0], table_size)
        paddle_x2 = padx(other_paddle_frect.pos[0], table_size)
        # print('ACTUAL VELOCITY:', ball_vel, predicted_vel)
        facing = paddle_frect.pos[0] < table_size[0]/2
        facing_sign = (2*facing - 1)
        if abs(ball_frect.pos[0] - paddle_x1) < (0 if ball_vel is None else mag(ball_vel)*THRESHOLD) and last_collided != 0:
            # print('COLLISION')
            # accuracy = max(abs(prediction - ball_center[1]), accuracy)
            keep_predicting = False
            ticks_prediction = math.inf
            goal = default_goal
            last_collided = 0
            continue_aiming = False
            temp_counter = 10

        if abs(ball_frect.pos[0] - paddle_x2) < (0 if ball_vel is None else mag(ball_vel)*THRESHOLD) and last_collided != 1:
            ticks_prediction = 2
            # print('START')
            # print(last_collided)
            keep_predicting = True
            continue_aiming = True
            last_collided = 1
            prediction = None

        if ticks_prediction > 0:
            ticks_prediction -= 1

        elif len(vels) == 2 and vels[-1] == vels[0] and sign(vels[0][0]) == -facing_sign:
            print('TS', ts)
            # last_pred = prediction
            # last_pred_time = prediction_time
            prediction, prediction_time, n_bounces, bounce_time = predict(ball_frect.pos, ball_vel, table_size, padx(paddle_frect.pos[0], table_size))


            prediction_sign = 1 - 2*(n_bounces % 2)
            predicted_vel = [-ball_vel[0], prediction_sign*ball_vel[1]]
            #targeting section
            max_dist = (prediction_time)/paddle_frect.size[1]
            leeway = min((abs(prediction - center_self[1]) - paddle_frect.size[1]/2 - 15/2)/paddle_frect.size[1], 1)
            leeway = (max_dist - leeway)
            dv = sign(prediction - center_self[1])
            bounds = None
            if leeway >= 0:
                if leeway <= 1:
                    bound = dv*0.5
                    bounds = [bound*max_angle*facing_sign, (bound - dv*leeway)*max_angle*facing_sign]
                else:
                    bound = (prediction - center_self[1])/paddle_frect.size[1]
                    bounds = [max(-0.5, bound - max_dist)*max_angle*facing_sign, min(0.5, bound + max_dist)*max_angle*facing_sign]
            else:
                # print('HOPELESS?')
                goal = prediction
                ticks_prediction = 2
                return
            bounds.sort()
            
            # region = 0
            if continue_aiming:
                modulus = table_size[1] - 15
                # idx = center_other[1] > table_size[1]/2
                current = [-ball_vel[0], prediction_sign*ball_vel[1]]
                dx = padx(other_paddle_frect.pos[0], table_size) - padx(paddle_frect.pos[0], table_size)
                for bounce in [1, -1]:
                    for iii in [0, 1, -1]:
                        target = [dx, ((0 if ts is None else ts*table_size[1])*bounce + iii*modulus*2) - prediction]
                        target_region = ang(current, target)/2*facing_sign
                        if bounds[0] <= target_region <= bounds[1]:
                            region = (target_region/max_angle)*paddle_frect.size[1]
                            break
                    else:
                        continue
                    break
                if region != 0:
                    # print('AIMING UP' if idx == 1 else 'AIMING DOWN')
                    pass
            goal = prediction - region
            ticks_prediction = math.inf if len(bounce_time) < 1 else bounce_time[0]
            continue_aiming = False
            # keep_predicting = False
            # count += 1

        # print('TOOK:', time.time() - init_time)
        if center_self[1] < goal:
            return 'down'
        elif center_self[1] > goal:
            return 'up'
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        import traceback
        traceback.print_exc()
        # print('ERROR1')
        error_count += 1
        # print('exception', e)
        # input()
        return
    # pass