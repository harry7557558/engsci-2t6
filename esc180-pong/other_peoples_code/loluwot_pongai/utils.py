import math
def ang(vec1, vec2):
    # return (vec1[0]*vec2[0] + vec1[1]*vec2[1])/math.sqrt((vec1[0]**2 + vec1[1]**2)*(vec2[0]**2 + vec2[1]**2))
    return math.atan2(vec1[0]*vec2[1] - vec1[1]*vec2[0], vec1[0]*vec2[0] + vec1[1]*vec2[1])
def center(frect):
    return [frect.pos[i] + frect.size[i]/2 for i in range(2)]

def padx(posx, table_size):
    paddle_x = posx
    if paddle_x < table_size[0]/2:
        paddle_x += 10
        # print(paddle_x, 1)
    else:
        paddle_x -= 15
        # print(paddle_x, 2) 
    return paddle_x

def sign(v):
    return 0 if v == 0 else abs(v)//v

def mag(v):
    return math.sqrt(v[0]**2 + v[1]**2)
