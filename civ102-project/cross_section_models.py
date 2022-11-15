# Defines common cross section geometry.
# All function return (points, glues) = (list[list[point]], list[[point, point]])


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
    glues = [
        (-0.5*wt, h),
        (0.5*wt, h)
    ]
    return [points], [glues]


def trapezoid_internal_member_bottom(wt, wb, h):
    assert max(wt, wb, h) > 0
    points = [[
        (-0.5*wt, h),
        (0, h),
        (-0.5*wb, 0),
        (0.5*wb, 0),
        (0, h),
        (0.5*wt, h)
    ]]
    glues = [
        [(-0.5*wt, h), (0, h)],
        [(0, h), (0.5*wt, h)],
        [(-0.5*wb, 0), (0.5*wb, 0)]
    ]
    return points, glues


def trapezoid_internal_member_side(wt, wb, h):
    assert max(wt, wb, h) > 0
    points = [[
        (-0.5*wt, h),
        (0, h),
        (-0.5*wb, 0),
        (-0.5*wt, h)
    ], [
        (0.5*wt, h),
        (0, h),
        (0.5*wb, 0),
        (0.5*wt, h)
    ]]
    glues = [
        [(-0.5*wt, h), (0, h)],
        [(-0.5*wb, 0), (-0.5*wt, h)],
        [(0.5*wt, h), (0, h)],
        [(0.5*wb, 0), (0.5*wt, h)]
    ]
    return points, glues

