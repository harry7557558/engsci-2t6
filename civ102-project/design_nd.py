import bridge_analysis as ba
import cross_section_models as csm
from export_bridge import export_bridge


def mix(a, b, t):
    return a + (b-a) * t


# total 9 parameters:
# @wt, @wb, @h
# length of each side beam, @sl
# length of the central beam @cl
# strengthen middle edges @mel, @met, @mes, length, top width, side width

# Trapezoid body: folded
# Trapezoid top: glue the edges (10mm padding)
# Diaphragms: glue the edges

C = __import__('beam_analysis').LENGTH/2
E11, E12, E21, E22 = 0, 20, 40, 60
EC = 0.5*(E12+E21)

# 7+2 diaphragms


def calc_cross_section(wt, wb, h, sl, cl, mel, met, mes):
    # geometry constraints
    if not min(wt, wb, h, cl, mel, met, mes) > 0:
        return None
    if wt < 90:  # top must hold the train
        return None
    if wb < 20 or h > 200:  # hard constraints
        return None
    if mel > C:  # these can't be higher than half of the bridge length
        return None
    if 2*met > wt:  # no overlapping?
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
    # beam cross sections
    offset = 1.5
    tm = csm.trapezoid_glue_edge_2(wt, wb, h)
    ts = csm.trapezoid_glue_edge_2(wt, wb, h)
    sp = csm.trapezoid_rect_support(wt, wb, h)
    tes = csm.trapezoid_edge_strengthen(wt, wb, h, met, mes)
    tws = csm.trapezoid_wall_strengthen(wt, wb, h)
    res = [
        ba.BridgeCrossSection('side_beam_1', *ts, 0, sl, [offset]+[0]*4),
        ba.BridgeCrossSection('side_beam_2', *ts, 2*C-sl, 2*C, [offset]+[0]*4),
        ba.BridgeCrossSection('central_beam', *tm, C-0.5*cl, C+0.5*cl, [2*offset]+[offset]*4),
        ba.BridgeCrossSection('mid_strengthen', *tes, C-0.5*mel, C+0.5*mel),
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
    D1 = mix(EC, 2*C-EC, 0.14)
    D2 = mix(EC, 2*C-EC, 0.3)
    D3 = mix(EC, 2*C-EC, 0.4)
    D4 = mix(EC, 2*C-EC, 0.5)
    D1 = sl+10
    D2 = C-0.5*mel-10
    D3 = 0.5*(D2+D4)
    if D2-D1 < 20:
        return None
    offset = 1.0
    cs = csm.trapezoid_nowrap(wt, wb, h)
    sp = csm.trapezoid_rect_support_diaphragm(wt, wb, h)
    res = [
        ba.BridgeCrossSection('de:support_1%a', *cs, E12, E21, 0),
        ba.BridgeCrossSection('de:support_1_1%a', *sp, E11, E12, -0.5*offset),
        ba.BridgeCrossSection('de:support_1_2%c', *sp, E21, E22, -0.5*offset),
        ba.BridgeCrossSection('de:support_2%b', *cs, 2*C-E21, 2*C-E12, 0),
        ba.BridgeCrossSection('de:support_2_1%b', *sp, 2*C-E12, 2*C-E11, -0.5*offset),
        ba.BridgeCrossSection('de:support_2_2%d', *sp, 2*C-E22, 2*C-E21, -0.5*offset),
        ba.BridgeCrossSection('d:d1_1%c', *cs, D1-10, D1+10, 0.5*offset),
        ba.BridgeCrossSection('d:d1_2%d', *cs, 2*C-D1-10, 2*C-D1+10, 0.5*offset),
        ba.BridgeCrossSection('d:d2_1%%e', *cs, D2-10, D2+10, 0.5*offset),
        ba.BridgeCrossSection('d:d2_2%%e', *cs, 2*C-D2-10, 2*C-D2+10, 0.5*offset),
        ba.BridgeCrossSection('d:d3_1%%f', *cs, D3-10, D3+10, 0.5*offset),
        ba.BridgeCrossSection('d:d3_2%%f', *cs, 2*C-D3-10, 2*C-D3+10, 0.5*offset),
        ba.BridgeCrossSection('d:d4', *cs, D4-10, D4+10, 0.5*offset),
    ]
    return res


if __name__ == "__main__":

    # FoS = 2.40
    initial_params = [91, 30, 65, 180, 990, 500, 20, 15]
    initial_params = [91.61580029591246, 46.463653533343106, 81.17993434406554, 219.58374917543213, 927.3505703015933, 402.58218749702394, 20.25507075560746, 8.39290511377031]

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
    #bridge.analyze(bridge.params, plot=True)
    #bridge.plot_3d(zoom=1.5)

    description = """
<p>
The bridge design consists of three segments symmetrical about the middle span.
The middle one is thicker than the two at the sides due to glue joints.
Glue joints are closer to the end for a lower bending moment and curvature.
</p>
<p>
The body of the bridge is an inverted trapezoid.
There are two layers of matboard at the top, with flanges extending beyond the top of the trapezoid
to increase the height of centroidal axis and therefore decrease <i>y</i> in the equation <i>My/I</i>
and increase compressive failure load.
Components are designed to strengthen the middle span and the supports of the bridge.
</p>
<p>
The bottom and sides of the trapezoid are folded.
Inspired by the winning group last year,
we decide to glue the edges between paddings instead of using folded tabs
for the top of the trapezoid and diaphragms.
These parts are mainly subjected to compression and are less likely to fail due to tension and shear.
Intuitively, folding tabs involves deforming the matboard, making it more likely to buckle.
</p>
<p>
This bridge design has 7 diaphragms in the middle span.
Since the analysis shows shear buckling is not the main cause of the bridge's failure,
we consider there is no need for extra diaphragms at the ends.
The cross section of the middle of the bridge will likely shrink vertically due to bending moment
and therefore have a decreased second moment of area.
Diaphragms are placed in the middle to resist such deformation.
</p>
<p>
Analysis based on the CIV102 course shows the bridge will first fail due to compression in the midspan.
However, intuition tells us the glue joint between beam segments may be the weakest part.
The bridge may also crush at the ends due to reaction force.
</p>
""".strip()

    export_bridge(
        bridge,
        "design_nd",
        [100, 300, 600],
        description
    )
