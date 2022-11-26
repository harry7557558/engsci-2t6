from bridge_analysis import *


def format_float(x, s=3):
    """Slide-rule precision"""
    if abs(x) < 1e-4:
        return "0"
    sigfig = s+1 if np.log10(abs(x))%1.0 < np.log10(2) else s
    tz = ""
    while x >= 10**sigfig:
        x = round(x/10)
        tz += "0"
    return f"{{:.{sigfig}g}}".format(x) + tz


def wrap_h2(text):
    h, p, m = 0, 1, 65521
    for c in text:  # rolling hash
        h = (h + ord(c)) * p % m
        p = p * 31 % m
    h = hex(h)[2:]
    return f'<h2 id="{h}"><a href="#{h}">{text}</a></h2>'


def wrap_svg(content, w, h):
    if isinstance(content, list):
        content = '\n'.join(content)
    return f"""
<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">
    {content}
</svg>"""


def wrap_table(contents):
    nrow = len(contents)
    ncol = max([len(row) for row in contents])
    trs = []
    for row in contents:
        tds = []
        for ele in row:
            tds.append("<td>"+ele+"</td>")
        trs.append("<tr>"+'\n'.join(tds)+"</tr>")
    return "<table>" + '\n'.join(trs) + "</table>"


def wrap_html(children):
    content = '\n'.join(children)
    return f"""<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <title>CIV102 Bridge Design</title>
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="robots" content="none" />
    <style>
        body{{ padding: 0px 10px; }}
        h2 a{{ color: black; }}
        img{{ max-width: 100%; max-height: 600px; display: block; }}
        td{{ padding: 10px; }}
    </style>
</head>
<body>
    {content}
</body>
</html>"""


def export_cross_section_svg(cs: BridgeCrossSection):
    t = csa.LINDEN_M  # thickness of the matboard
    children = []
    minx, maxx, miny, maxy = np.inf, -np.inf, np.inf, -np.inf
    for part in cs.parts_offset:
        points = []
        for p in part:
            minx, maxx = min(minx, p[0]), max(maxx, p[0])
            miny, maxy = min(miny, p[1]), max(maxy, p[1])
            points.append(','.join(map(str, p)))
        points = ' '.join(points)
        children.append(f'<polyline points="{points}" '
                        f'stroke-width="{t}" stroke="black" fill="none" />')
    pad, th = 10, 16
    minx, maxx, miny, maxy = minx-pad, maxx+pad, miny-pad, maxy+pad
    transform = ' '.join(map(str,
        [1, 0, 0, -1, -minx, maxy+th]))
    g = f'<g transform="matrix({transform})">' + '\n'.join(children) + '</g>'
    label = cs.label[:(cs.label+'%').find('%')]
    text = f'<text x="{0.5*(maxx-minx)}" y="{th}" text-anchor="middle">{label}</text>'
    return wrap_svg([g, text], maxx-minx, maxy-miny+th)


def export_cross_section_text(cs: BridgeCrossSection, show_length=True):
    lines = []
    if show_length:
        lines.append("length: " + format_float(cs.x1-cs.x0))
    else:
        lines.append("location: " + format_float(0.5*(cs.x0+cs.x1)) + " from one end")
    for part in cs.parts_offset:
        points = []
        for p in part:
            s = ', '.join([format_float(p[0]), format_float(p[1])])
            points.append(f"({s})")
        points = ', '.join(points) + ';'
        lines.append(points)
    return "<div>" + "<br/>\n".join(lines) + "</div>"


def export_bridge(bridge: Bridge,
                  filename: str,
                  manual_locations: list[float],
                  description: str):
    """filename should have no extension"""
    html_elements = [
        f"<h1>CIV102 Bridge Design <code>`{filename}`</code></h1>",
        description,
        "<hr/>"
    ]

    # figures
    if True:
        imgsrc = filename+"_fos.png"
        bridge.analyze(bridge.params, plot=True, show=False)
        plt.savefig(imgsrc)
        html_elements += [
            wrap_h2("FoS plots"),
            """<p><b>Top figure</b>:
the first four FoS are the actual FoS without dividing by factors (I asked Raymond).
Ignore flexural shear FoS when writing the report
(and possibly remove that plot in the future)
because I don't think my formula is correct.</p>""",
            """<p><b>Bottom figure</b>:
How the matboard will be cut. Gray is unused region.
Light yellow is support. Light blue is diaphragm.
A light pink rectangle contains multiple diaphragms.
A more detailed figure appears at the bottom of this page.</p>""",
            f'<img src="{imgsrc}" />',
            '<hr/>'
        ]
        imgsrc = filename+"_3d.png"
        bridge.plot_3d(zoom=1.5, show=False)
        plt.savefig(imgsrc)
        html_elements += [
            wrap_h2("3D bridge plot"),
            "<p>Gray is bridge body. Black is diaphragm. "
            "Red highlights the edge of the platform when the bridge is tested.</p>",
            f'<img src="{imgsrc}" />',
            '<hr/>'
        ]

    # cross sections
    html_elements += [
        wrap_h2("Cross-section components"),
        "<p>All lengths and coordinates are in mm. One pixel on this page is 1&nbsp;mm. "
        "Zoom in in your browser if you think the figures are too small.</p>",
        "<p>A cross section is represented as a list of point coordinates in order. "
        "This format makes it easier to analyze computationally.</p>",
    ]
    cross_sections = bridge.calc_cross_section(*bridge.params)
    if cross_sections is None:
        print("Cross section constraint violation.")
        cross_sections = []
    for cs in cross_sections:
        svg = export_cross_section_svg(cs)+"<br/>\n"
        text = export_cross_section_text(cs)
        html_elements.append(wrap_table([[svg, text]]))
    html_elements.append("<hr/>")

    # diaphragms
    html_elements += [
        wrap_h2("Diaphragm components"),
        "<p>All locations and coordinates are in mm. One pixel on this page is 1&nbsp;mm.</p>",
        "<p>A diaphragm a polygon represented as a list of point coordinates in order. "
        "The last point in the point list must be the same as the first point to enforce a closed polygon.</p>",
        "<p>In bridge construction, two 1cm strips at each side of a diaphragm and the thin edges of the diaphragm are glued.</p>",
    ]
    diaphragms = bridge.calc_diaphragms(*bridge.params)
    if diaphragms is None:
        print("Diaphragm constraint violation.")
        diaphragms = []
    diaphragms.sort(key=lambda d: d.x0)
    for d in diaphragms:
        svg = export_cross_section_svg(d)+"<br/>\n"
        text = export_cross_section_text(d, False)
        html_elements.append(wrap_table([[svg, text]]))
    html_elements.append("<hr/>")

    # manual calculation locations
    html_elements += [
        wrap_h2("Cross-sections to manually calculate"),
        "<p>Manually analyze these cross sections for the report.</p>",
        "<p>Note that lines may overlap in the diagram. "
        "The coordinates are more trustworthy. "
        "Zoom in in your browser if you think the figures are too small. ",
        "Glue joints can be identified by inspecting visually.</p>",
        "<p>Failure loads for <i>some</i> failure modes calculated from my computer program are given. "
        "The initial weight of the train is 400N. "
        "Since I considered additional factors, I think it is acceptable that a number calculated based on the course is different from the program's result "
        "(and I expect some hand results to be larger). "
        "Let me know if you get a number very different from the program's number so I can check for potential mistakes in my code.</p>",
    ]
    for x in manual_locations:
        # get parts
        parts, parts_offset = [], []
        for cs in cross_sections:
            if cs.x0 <= x <= cs.x1:
                parts += cs.parts
                parts_offset += cs.parts_offset
        assert len(parts) != 0

        # shear buckling `a`
        csa.LENGTH = 0  # for shear buckling
        for dc1, dc2 in zip(diaphragms[:-1], diaphragms[1:]):
            x1 = 0.5*(dc1.x0+dc1.x1)
            x2 = 0.5*(dc2.x0+dc2.x1)
            if x1 <= x <= x2:
                csa.LENGTH = x2 - x1
                break
        assert csa.LENGTH != 0

        # key points
        xs = []
        for cs in cross_sections:
            xs += [cs.x0, cs.x1]
        #for d in diaphragms:
        #    xs += [d.x0, 0.5*(d.x0+d.x1), d.x1]
        xs = sorted(set(xs))
        for x1, x2 in zip(xs[:-1], xs[1:]):
            if x1 <= x <= x2:
                break
        csa.BM_MAX = max(get_max_bend(x1, x2), 1)
        csa.SHEAR_MAX = max(get_max_shear(x1, x2), 1)

        # properties
        fos_bend, fos_bend_buckle, fos_shear, fos_shear_buckle = \
                    csa.analyze_cross_section(
                parts, parts_offset, return_full=True)
        cs = BridgeCrossSection(f"x={x}", parts_offset, [], x1, x2)
        svg = export_cross_section_svg(cs)
        text = export_cross_section_text(cs)
        W0 = 400
        fos = '<div>' + '<br/>\n'.join([
            "Location: (" + format_float(x1) + "&nbsp;mm, " + format_float(x2) + "&nbsp;mm)",
            "Max shear: " + format_float(csa.SHEAR_MAX) + "&nbsp;N",
            "Max bending: " + format_float(1e-3*csa.BM_MAX) + "&nbsp;N⋅m"
            "<hr/>"
            "Flexural: " + format_float(W0*fos_bend) + "&nbsp;N",
            "Buckling: " + format_float(W0*fos_bend_buckle) + "&nbsp;N",
            "Shear: " + format_float(W0*fos_shear) + "&nbsp;N",
            "Shear buckling: " + format_float(W0*fos_shear_buckle) + "&nbsp;N"
        ]) + '</div>'
        html_elements.append(wrap_table([[svg, text, fos]]))
    html_elements.append("<br/><hr/>")

    # how the matboard will be cut
    html_elements += [
        wrap_h2("How the matboard will be cut"),
        f"""<p>
The matboard is divided into rectangles.
Bottom left is (0, 0), top right is ({MATBOARD_W}, {MATBOARD_H}).
Light yellow is support. Light blue is diaphragm.
White is beam. Gray is unused region.
All units are in mm. One pixel in this diagram is 1&nbsp;mm.
</p><p>
On a desktop device, you can mouse hover a rectangle to read the numbers.
The second row contains the distance of the vertices of the rectangle from the edge of the matboard.
The third row contains the width and height of the rectangle on its own coordinate system.
Note that rectangles may be rotated.
A sequence of point coordinates in the rectangle's coordinate system
representing the cut or fold is followed (if cut or fold exists.)
</p><p>
When marking and cutting the matboard,
label each rectangle before cutting it off so we don't mess it up.
The bridge components are packed very compactly.
Make sure you make no mistake when marking and cutting the board -
there is no room for a mistake.
We have extra materials left, and we may consider using them to
strenghten the endpoints of the bridge that support the reaction force.
</p>""",
    ]
    html_elements += [
        '<div id="display" style="position:fixed;display:none;background-color:rgba(250,230,240,0.95);border:1px solid gray;padding:10px;max-width:500px;"></div>'
        """<script>
var mouseX = -1, mouseY = -1;
document.addEventListener('mousemove', function(event) {
    mouseX = event.clientX, mouseY = event.clientY;
});
function displayText(text) {
    let display = document.getElementById("display");
    if (text == '') { display.style.display = 'none'; return; }
    display.style.display = "block";
    if (mouseX < 0.5*window.innerWidth) {
        display.style.left = (mouseX+10)+'px';
        display.style.removeProperty('right');
    }
    else {
        display.style.right = (window.innerWidth-mouseX)+'px';
        display.style.removeProperty('left');
    }
    if (mouseY < 0.5*window.innerHeight) {
        display.style.top = (mouseY+10)+'px';
        display.style.removeProperty('bottom');
    }
    else {
        display.style.bottom = (window.innerHeight-mouseY)+'px';
        display.style.removeProperty('top');
    }
    display.innerHTML = text;
}
</script>"""
    ]
    svg_rects = []
    svg_marks = []
    svg_labels = []
    def add_label(xm, ym, x, y, rotate, label):
        nonlocal svg_labels
        svg_labels += [
            f'<text x="{x}" y="{y+4}" text-anchor="middle" style="pointer-events:none;"'
            f' transform="scale(1,-1) rotate({rotate},{xm},{ym})">{label}</text>'
        ]
    def add_rect(x, y, w, h, label, caption):
        nonlocal svg_rects
        fill = plot_rect_color('' if is_diaphragm_label(label) else label)
        svg_rects += [  
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}"'
            f' stroke="black" stoke-width="1px" fill="{fill}"\n'
            f' onmousemove=\'displayText("{caption}");\'/>'
        ]
        xm, ym = x+0.5*w, -(y+0.5*h)
        if '\n' in label:
            labels = []
            for l in label.strip().split('\n'):
                if l not in labels:
                    labels.append(l)
            rotate = -90 if w<8*max([len(l) for l in labels]) and h>w else 0
            for i in range(len(labels)):
                add_label(xm, ym, xm, ym+16*(i-0.5*len(labels)+0.5),
                          rotate, labels[i])
        else:
            rotate = -90 if w<8*len(label) and h>w else 0
            add_label(xm, ym, xm, ym, rotate, label)
    def add_mark(points, label):
        nonlocal svg_marks
        fill = 'none'
        if is_diaphragm_label(label):
            fill = plot_rect_color(label)
        s = []
        for p in points:
            s.append(','.join(map(str, p)))
        s = ' '.join(s)
        svg_marks += [
            f'<polyline points="{s}" style="pointer-events:none;"'
            f' stroke="gray" stroke-width="0.5" stroke-dasharray="2" fill="{fill}" />'
        ]
    add_rect(0, 0, MATBOARD_W, MATBOARD_H, '', '')
    for d in diaphragms:
        c = 0.5*(d.x0+d.x1)
        cross_sections += [
            BridgeCrossSection(d.label[:(d.label+'%').find('%')]+'=',
                               d.parts, d.glues, d.x0, d.x1, offset=-csa.LINDEN_M),
        ]
    rects, labels, marks = bridge.generate_rects(
        cross_sections, diaphragms,
        require_labels=True, require_marks=True)
    comps = list(zip(rects, labels, marks))
    packer = newPacker()
    for i in range(len(comps)):
        packer.add_rect(*comps[i][0], str(i))
    transform = ' '.join(map(str,
        [1, 0, 0, -1, 1, MATBOARD_H+1]))
    packer.add_bin(MATBOARD_W, MATBOARD_H)
    packer.pack()
    abin = packer[0]
    if len(abin) < len(rects):
        html_elements.append(
            "<h1 style='color:#b00'>Error: cannot pack all parts into the matboard."
            " The following diagram is incomplete.</h1>")
    for prect in abin:
        i = int(prect.rid)
        rect, label, marks = comps[i]
        # generate caption
        caption = [
            '; '.join(set(label.split('\n'))),
            "(left, bottom, right, top) = ({:.1f}, {:.1f}, {:.1f}, {:.1f})".format(
                prect.x, prect.y,
                MATBOARD_W-(prect.x+prect.width),
                MATBOARD_H-(prect.y+prect.height)),
            "(w, h) = ({:.1f}, {:.1f})".format(rect[0], rect[1]),
            "<hr/>"
        ]
        for mark in marks:
            caption.append(', '.join(
                    ["({:.1f}, {:.1f})".format(p[0], p[1]) for p in mark])+';')
        # add rect
        add_rect(prect.x, prect.y, prect.width, prect.height,
                 label, '<br/>'.join(caption).replace('<hr/><br/>','<hr/>'))
        p0 = np.array((prect.x, prect.y))
        A = np.array([[1,0,0],[0,1,0]] if rect[0]==prect.width
                     else [[0,-1,rect[1]],[1,0,0]])
        if '\n' not in label:
            label = '\n'.join([label]*len(marks))
        for mark, label in zip(marks, label.split('\n')):
            ps = [p0+A[:,0:2].dot(p)+A[:,2] for p in mark]
            add_mark(ps, label)
    svg = [f'<g transform="matrix({transform})">'] + \
          svg_rects + svg_marks + svg_labels + ["</g>"]
    svg = wrap_svg(svg, MATBOARD_W+2, MATBOARD_H+2)
    svg = f"""<div style="overflow-x:scroll" onmouseout="displayText('')">{svg}</div>"""
    html_elements.append(svg)

    # end
    html_elements += ["<br/>"]*5
    html = wrap_html(html_elements)
    open(filename+".html", 'wb').write(bytearray(html, 'utf-8'))
