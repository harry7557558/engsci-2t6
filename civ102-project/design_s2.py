import bridge_analysis as ba
import cross_section_models as csm

#ba.csa.SIGMA_T = ba.csa.SHEAR_C


# total 7 parameters:
# @wt, @wb, @h
# length of each side beam, @sl
# length of the central beam @cl
# length of each side member, @sml
# length of the central member, @cml


# whether have internal member throughout the beam
# when set to True: FoS~2, fails due to bending
# when set to False: FoS~3, highly sus it will fail due to cement shear
MEMBER_THROUGHOUT = False


trapezoid = csm.trapezoid
central_im = csm.trapezoid_internal_member_bottom
side_im = csm.trapezoid_internal_member_bottom


def calc_cross_section(wt, wb, h, sl, cl, sml, cml):
    c = 1260/2
    # geometry constraints
    if not min(wt, wb, h, cl, cml, sml) > 0:
        return None
    if wt < 75:  # top must hold train
        return None
    if cl > 2*c or cml > 2*c or sml > c:  # must not longer than bridge
        return None
    if 2*sl+cl < 2*c:  # frame must be longer than bridge
        return None
    if MEMBER_THROUGHOUT:
        if 2*sml+cml < 2*c:
            return None
    # beam cross sections
    return [
        ba.BridgeCrossSection('side_beam_1', *trapezoid(wt, wb, h), 0, sl),
        ba.BridgeCrossSection('side_beam_2', *trapezoid(wt, wb, h), 2*c-sl, 2*c),
        ba.BridgeCrossSection('central_beam', *trapezoid(wt, wb, h), c-0.5*cl, c+0.5*cl),
        ba.BridgeCrossSection('side_member_1', *side_im(wt, wb, h), 0, sml),
        ba.BridgeCrossSection('side_member_2', *side_im(wt, wb, h), 2*c-sml, 2*c),
        ba.BridgeCrossSection('central_member', *central_im(wt, wb, h), c-0.5*cml, c+0.5*cml)
    ]


if __name__ == "__main__":

    bridge = ba.Bridge(calc_cross_section,
                    [[75, 150], [20, 100], [20, 100],
                     [200, 630], [200, 1000], [100, 630], [200, 1000]],
                    ['wt', 'wb', 'h', 'sl', 'cl', 'sml', 'cml'],
                    [80, 40, 50, 400, 600, 200, 900])
