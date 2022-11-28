import numpy as np
import cross_section_analysis as csa


# these parameters strictly follows the project instruction
# does not consider bridge self-weight and deflection
csa.REACTION_MAX = 267.33
csa.SHEAR_MAX = 257.00
csa.BM_MAX = 69438


# assumption:
# matboard thickness is small compared to other dimensions
# treat parts as lines with line density

# by comparing the program's output with human calculation,
# this assumption affects failure load by ±5%


def parts(t2):
    parts = [
        [
            (50, 75+t2),
            (-50, 75+t2)
        ],
        [
            (-(40-5-2*t2), 75-t2),
            (-(40-t2), 75-t2),
            (-(40-t2), t2),
            (40-t2, t2),
            (40-t2, 75-t2),
            (40-5-2*t2, 75-t2)
        ]
    ]
    return [[np.array(p) for p in part] for part in parts]


res = csa.analyze_cross_section(
    parts(0), parts(0.5*csa.LINDEN_M),
    plot=True, return_full=True)
print(400*min(res), 'N')
