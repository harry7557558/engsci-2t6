import bridge_analysis as ba
import cross_section_models as csm


# total 9 parameters:
# @wt, @wb, @h
# length of each side beam, @sl
# length of the central beam @cl
# strenghen middle edges @mel, @met, @mes, length, top width, side width
# @d1 - location of diaphram or endpoint of wall, depending on the following variable
ANTISHEAR = ['None', 'diaphram', 'strenghen'][1]  # don't change this

# Trapezoid body: folded
# Trapezoid top: folded, or glue the edges (10mm padding)
GLUE_EDGE = True
# Diaphrams: glue the edges


def calc_cross_section(wt, wb, h, sl, cl, mel, met, mes, d1):
    c = 1250/2
    # geometry constraints
    if not min(wt, wb, h, cl, mel, met, mes) > 0:
        return None
    if wt < (90 if GLUE_EDGE else 100):  # top must hold the train
        return None
    if h > 200:  # hard constraint
        return None
    if mel > c or d1 > c:  # these can't be higher than half of the bridge length
        return None
    if cl > 2*c:  # must not longer than bridge
        return None
    if 2*sl+cl < 2*c+50*2:  # frame must be longer than bridge, with glue joints
        return None
    if wt < wb:  # force bottom large trapezoid
        return None
    if False:
        if (wt-wb)/2 > 75/2:  # must not fall when the train is at the edge
            return None
    if d1 < 100:  # diaphrams must be in order
        return None
    # beam cross sections
    if GLUE_EDGE:
        tm = csm.trapezoid_glue_edge_2(wt, wb, h)
        #ts = csm.trapezoid_glue_edge_1(wt, wb, h)
        ts = csm.trapezoid_glue_edge_2(wt, wb, h)
    else:
        tm = csm.trapezoid(wt, wb, h)
        ts = csm.trapezoid(wt, wb, h)
    sp = csm.trapezoid_rect_support(wt, wb, h)
    tes = csm.trapezoid_edge_strenghen(wt, wb, h, met, mes)
    tws = csm.trapezoid_wall_strenghen(wt, wb, h)
    res = [
        ba.BridgeCrossSection('side_beam_1', *ts, 0, sl),
        ba.BridgeCrossSection('side_beam_2', *ts, 2*c-sl, 2*c),
        ba.BridgeCrossSection('central_beam', *tm, c-0.5*cl, c+0.5*cl),
        ba.BridgeCrossSection('mid_strenghen', *tes, c-0.5*mel, c+0.5*mel),
        ba.BridgeCrossSection('support_1', *sp, 10, 70),
        ba.BridgeCrossSection('support_2', *sp, 2*c-70, 2*c-10),
    ]
    if ANTISHEAR[0] == 's':
        res += [
            ba.BridgeCrossSection('antishear_1', *tws, 60, d1),
            ba.BridgeCrossSection('antishear_2', *tws, 2*c-d1, 2*c-60),
        ]
    return res


def calc_diaphrams(wt, wb, h, sl, cl, mel, met, mes, d1):
    if wt < wb:  # force bottom large trapezoid
        return None
    if d1 < 100:  # diaphrams must be in order
        return None
    c = 1250 / 2
    x1 = 50
    x2 = 2*c - 50
    # each wheel 20mm*10mm = 200mm² (measured)
    # pressure on surface (400N/12)/(200mm²) = 0.1667 MPa
    # diaphram buckling load 4π²E / 12(1-μ²) * (1.27mm/100mm)² = 2.21 MPa
    # single layer of matboard is enough for a diaphram
    cs = csm.trapezoid_nowrap(wt, wb, h)
    sp = csm.trapezoid_rect_support_diaphram(wt, wb, h)
    res = [
        ba.BridgeCrossSection('de:support_1_1', *sp, 10, 30),
        ba.BridgeCrossSection('de:support_1_2', *sp, 50, 70),
        ba.BridgeCrossSection('d:support_1', *cs, 30, 50),
        ba.BridgeCrossSection('de:support_2_1', *sp, 2*c-30, 2*c-10),
        ba.BridgeCrossSection('de:support_2_2', *sp, 2*c-70, 2*c-50),
        ba.BridgeCrossSection('d:support_2', *cs, 2*c-50, 2*c-30),
    ]
    if ANTISHEAR[0] == 'd':
        res += [
            ba.BridgeCrossSection('d:d1_1', *cs, d1-10, d1+10),
            ba.BridgeCrossSection('d:d1_2', *cs, 2*c-d1-10, 2*c-d1+10),
        ]
    d2 = d1 + (c-d1) * 0.7
    res += [
        ba.BridgeCrossSection('d:d2_1', *cs, d2-10, d2+10),
        ba.BridgeCrossSection('d:d2_2', *cs, 2*c-d2-10, 2*c-d2+10),
    ]
    return res


if __name__ == "__main__":

    if GLUE_EDGE:
        initial_params = [91, 30, 75, 180, 1000, 550, 20, 20, 300]
        initial_params = [92.70233971611614, 30.57634152078104, 76.5843209011683, 186.6558974498439, 1012.005537378484, 518.1478113346183, 23.544923632735916, 22.805179996140616, 395.51768346505884]
    else:
        initial_params = [101, 40, 80, 400, 600, 600, 20, 20, 300]
        initial_params = [102.684522085972, 53.71937179728894, 82.55439750082016, 381.2256149789538, 611.4630960762545, 560.6199741168588, 19.65846215217679, 17.025439068871144, 297.20140203822444]

    bridge = ba.Bridge(
        calc_cross_section, calc_diaphrams,
        [[75, 150], [20, 100], [20, 100],
         [200, 630], [200, 1000],
         [0, 800], [0, 40], [0, 40],
         [100, 400]
        ],
        ['wt', 'wb', 'h', 'sl', 'cl', 'mel', 'met', 'mes', 'd1'],
        initial_params
    )
    #bridge.optimize_params()
    bridge.analyze(bridge.params, plot=True)
    bridge.plot_3d(zoom=1.5)
