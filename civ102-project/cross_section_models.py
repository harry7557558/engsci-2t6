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
