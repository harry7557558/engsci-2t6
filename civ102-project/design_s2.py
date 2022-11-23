import bridge_analysis as ba
import cross_section_models as csm


def mix(a, b, t):
    return a + (b-a) * t


# total 9 parameters:
# @wt, @wb, @h
# length of each side beam, @sl
# length of the central beam @cl
# strenghen middle edges @mel, @met, @mes, length, top width, side width

# Trapezoid body: folded
# Trapezoid top: folded, or glue the edges (10mm padding)
GLUE_EDGE = True
# Diaphragms: glue the edges

C = __import__('beam_analysis').LENGTH/2
E11, E12, E21, E22 = 10, 30, 50, 70
EC = 0.5*(E12+E21)
D1 = mix(EC, 2*C-EC, 0.16)
D2 = mix(EC, 2*C-EC, 0.38)

# six diaphragms
# two at the ends, four uniformly spaced


def calc_cross_section(wt, wb, h, sl, cl, mel, met, mes):
    # geometry constraints
    if not min(wt, wb, h, cl, mel, met, mes) > 0:
        return None
    if wt < (90 if GLUE_EDGE else 100):  # top must hold the train
        return None
    if h > 200:  # hard constraint
        return None
    if mel > C:  # these can't be higher than half of the bridge length
        return None
    if cl > 2*C:  # must not longer than bridge
        return None
    if 2*sl+cl < 2*C+20*2:  # frame must be longer than bridge, with glue joints
        """Failure of glue joint due to tension
            400N / 2MPa = 200mm²
            200mm²/100mm = 2mm is theoretically enough
            Intuitively it doesn't make sense so put 20mm.
        """
        return None
    if wt < wb:  # force bottom large trapezoid
        return None
    if False:
        if (wt-wb)/2 > 75/2:  # must not fall when the train is at the edge
            return None
    # beam cross sections
    offset = 1.5
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
        ba.BridgeCrossSection('side_beam_2', *ts, 2*C-sl, 2*C),
        ba.BridgeCrossSection('central_beam', *tm, C-0.5*cl, C+0.5*cl, offset),
        ba.BridgeCrossSection('mid_strenghen', *tes, C-0.5*mel, C+0.5*mel),
        ba.BridgeCrossSection('support_1', *sp, E11, E22, -offset),
        ba.BridgeCrossSection('support_2', *sp, 2*C-E22, 2*C-E11, -offset),
    ]
    """Buckling of flange
        Failure stress 0.425π²E / 12(1-μ²) * (1.27mm/10mm)² = 23.5 MPa
        Far higher than the board's compressive strength. Not a big issue.
    """
    return res


def calc_diaphragms(wt, wb, h, sl, cl, mel, met, mes):
    if wt < wb:  # force bottom large trapezoid
        return None
    x1 = 50
    x2 = 2*C - 50
    """Buckling
        each wheel 20mm*10mm = 200mm² (measured)
        pressure on surface (400N/12)/(200mm²) = 0.1667 MPa
        all area 1.27mm * (100mm * n + 960mm * 2) = 2440mm² + 127mm² * n
        max pressure (400N)/(2440mm²) = 0.1640 MPa
        thin plate buckling load 4π²E / 12(1-μ²) * (1.27mm/100mm)² = 2.21 MPa
        shear buckling load 5π²E / 12(1-μ²) * ((1.27mm/100mm)² + (1.27mm/100mm)²) = 5.53 MPa
        single layer of matboard is enough for resisting buckling
    """
    offset = 1.0
    cs = csm.trapezoid_nowrap(wt, wb, h)
    sp = csm.trapezoid_rect_support_diaphragm(wt, wb, h)
    res = [
        ba.BridgeCrossSection('de:support_1_1', *sp, E11, E12, -offset),
        ba.BridgeCrossSection('de:support_1_2', *sp, E21, E22, -offset),
        ba.BridgeCrossSection('d:support_1', *cs, E12, E21, -0.5*offset),
        ba.BridgeCrossSection('de:support_2_1', *sp, 2*C-E12, 2*C-E11, -offset),
        ba.BridgeCrossSection('de:support_2_2', *sp, 2*C-E22, 2*C-E21, -offset),
        ba.BridgeCrossSection('d:support_2', *cs, 2*C-E21, 2*C-E12, -0.5*offset),
        ba.BridgeCrossSection('d:d1_1', *cs, D1-10, D1+10, -0.5*offset),
        ba.BridgeCrossSection('d:d1_2', *cs, 2*C-D1-10, 2*C-D1+10, -0.5*offset),
        ba.BridgeCrossSection('d:d2_1', *cs, D2-10, D2+10, -0.5*offset),
        ba.BridgeCrossSection('d:d2_2', *cs, 2*C-D2-10, 2*C-D2+10, -0.5*offset),
    ]
    return res


if __name__ == "__main__":

    if GLUE_EDGE:  # FoS = 1.227
        initial_params = [91, 30, 75, 180, 990, 550, 20, 15]
        initial_params = [93.30687690925592, 31.47898775249033, 80.49833020341086, 174.38359819519445, 1015.7345202782762, 573.0502448715256, 24.433320367758363, 15.486851655988339]
    else:  # FoS = 1.211
        initial_params = [101, 40, 80, 400, 600, 600, 20, 20]
        initial_params = [100.54145679296947, 32.464704401740775, 89.07456971232742, 255.6174743240572, 846.535369050526, 425.4754622128162, 10.05418326813907, 29.165368340775764]

    bridge = ba.Bridge(
        calc_cross_section, calc_diaphragms,
        [[75, 150], [20, 100], [20, 100],
         [200, 630], [200, 1000],
         [0, 800], [0, 40], [0, 40],
        ],
        ['wt', 'wb', 'h', 'sl', 'cl', 'mel', 'met', 'mes'],
        initial_params
    )
    bridge.assert_ccw()

    #bridge.optimize_params()
    bridge.analyze(bridge.params, plot=True)
    bridge.plot_3d(zoom=1.5)
