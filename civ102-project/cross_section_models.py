# Defines common cross section geometry.
# All function return (points, glues) = (list[list[point]], list[[point, point]])

import math


def trapezoid(wt, wb, h):
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
    assert max(wt, wb, h) > 0
    body = [
        (0.5*wt, h),
        (0.5*wb, 0),
        (-0.5*wb, 0),
        (-0.5*wt, h)
    ]
    top = [
        (-0.5*wt-10, h),
        (0.5*wt+10, h)
    ]
    strips = [
        [(-0.5*wt-10, h), (-0.5*wt, h)],
        [(-0.5*wt, h), (0.5*wt, h)],
        [(0.5*wt, h), (0.5*wt+10, h)]
    ]
    return [body, top] + strips, []


def trapezoid_glue_edge_1(wt, wb, h):
    assert max(wt, wb, h) > 0
    assert wt > 20
    body = [
        (0.5*wt, h),
        (0.5*wb, 0),
        (-0.5*wb, 0),
        (-0.5*wt, h)
    ]
    top = [
        (-0.5*wt-10, h),
        (0.5*wt+10, h)
    ]
    strips = [
        [(-0.5*wt-10, h), (-0.5*wt, h)],
        [(-0.5*wt, h), (-0.5*wt+10, h)],
        [(0.5*wt-10, h), (0.5*wt, h)],
        [(0.5*wt, h), (0.5*wt+10, h)]
    ]
    return [body, top] + strips, []


def trapezoid_rect_support(wt, wb, h):
    assert wt > wb
    start = (wb/2+(wt-wb)/(2*h)*(0.8*h), 0.8*h)
    points = [
        start,
        (0.5*wt, h),
        (0.5*wt, 0),
        (-0.5*wt, 0),
        (-0.5*wt, h),
        (-start[0], start[1])
    ]
    return [points], []


def trapezoid_rect_support_diaphram(wt, wb, h):
    assert wt > wb
    parts = [
        [(0.5*wt, h), (0.5*wt, 0), (0.5*wb, 0), (0.5*wt, h)],
        [(-0.5*wt, h), (-0.5*wt, 0), (-0.5*wb, 0), (-0.5*wt, h)]
    ]
    return parts, []
    

def trapezoid_edge_strenghen(wt, wb, h, et, es):
    theta = math.atan((wt-wb)/(2*h))
    ex = 0.5*wt - es*math.sin(theta)
    ey = h - es*math.cos(theta)
    parts = [
        [(0.5*wt-et, h), (0.5*wt, h), (ex, ey)],
        [(-(0.5*wt-et), h), (-0.5*wt, h), (-ex, ey)]
    ]
    return parts, []


def trapezoid_wall_strenghen(wt, wb, h):
    parts = [
        [(0.5*wb, 0), (0.5*wt, h)],
        [(-0.5*wb, 0), (0.5*wt, h)]
    ]
    return parts, []
