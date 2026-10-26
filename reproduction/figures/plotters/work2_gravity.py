"""Plot all fixed-bank gravity outcomes and two outcome-aligned posterior heatmaps."""

import csv
import math
import pymupdf as fitz

f = fitz


def render(input_dir, output_dir):
    with (input_dir / "gravity_summary.csv").open(newline="") as stream:
        summary = list(csv.DictReader(stream))
    with (input_dir / "gravity_series.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    INK = (39 / 255, 71 / 255, 83 / 255)
    TEAL = (41 / 255, 157 / 255, 143 / 255)

    def text(p, x, y, s, size=8.5, bold=False, color=INK, align="left"):
        font = "hebo" if bold else "helv"
        w = f.get_text_length(s, fontname=font, fontsize=size)
        if align == "center":
            x -= w / 2
        if align == "right":
            x -= w
        p.insert_text((x, y), s, fontname=font, fontsize=size, color=color)

    def line(p, a, b, color=INK, width=0.7):
        p.draw_line(a, b, color=color, width=width)

    def new(h, w=468):
        d = f.open()
        p = d.new_page(width=w, height=h)
        return d, p

    # Fixed-bank sweep: same seven saved outcomes, fixed-size native labels.
    if sorted(float(r["actual_gravity"]) for r in summary) != [
        1.0,
        1.5,
        2.0,
        2.5,
        3.0,
        4.0,
        5.0,
    ]:
        raise ValueError("Expected the seven retained gravity levels, once each")
    if any(not 0 <= float(r["return_value"]) <= 6500 for r in summary):
        raise ValueError(
            "A return exceeds the retained publication axes; adjust the layout explicitly"
        )
    rs = summary
    d, p = new(174)
    text(p, 47, 15, "Total return / 25 s", 9, True)
    xx = lambda g: 55 + (g - 1) / 4 * 387
    yy = lambda v: 149 - v / 6500 * 118
    for v in [0, 2000, 4000, 6000]:
        line(p, (47, yy(v)), (456, yy(v)), (0.83, 0.83, 0.83), 0.4)
        text(p, 40, yy(v) + 3, str(v), 8.2, align="right")
    pts = [(xx(float(r["actual_gravity"])), yy(float(r["return_value"]))) for r in rs]
    for a, b in zip(pts, pts[1:]):
        line(p, a, b, INK, 1.2)
    for i, r in enumerate(rs):
        g = float(r["actual_gravity"])
        v = float(r["return_value"])
        x, y = pts[i]
        p.draw_circle((x, y), 2.4, color=INK, fill=INK)
        text(p, x, y - 8, f"{v:,.0f}", 8.3, align="center")
        text(p, x, 165, f"{g:g}x", 8.3, align="center")
    top = d
    INK = (39 / 255, 71 / 255, 83 / 255)
    CREAM = (246 / 255, 242 / 255, 232 / 255)
    TEAL = (41 / 255, 157 / 255, 143 / 255)
    FONT = {"regular": fitz.Font("helv"), "bold": fitz.Font("hebo")}

    def text(p, x, y, value, size=8.2, bold=False, align="left"):
        kind = "bold" if bold else "regular"
        name = "GravityStoryBold" if bold else "GravityStoryRegular"
        p.insert_font(fontname=name, fontbuffer=FONT[kind].buffer)
        w = FONT[kind].text_length(value, fontsize=size)
        if align == "center":
            x -= w / 2
        if align == "right":
            x -= w
        p.insert_text((x, y), value, fontname=name, fontsize=size, color=INK)

    codes = [f"{i:03b}" for i in range(8)]
    selected = {
        g: sorted(
            [r for r in rows if float(r["actual_gravity"]) == g],
            key=lambda r: int(r["transition_index"]),
        )
        for g in [2.0, 5.0]
    }
    for g, rr in selected.items():
        if len(rr) != 500 or [int(r["transition_index"]) for r in rr] != list(
            range(500)
        ):
            raise ValueError(
                "Expected 500 ordered transitions at each displayed gravity"
            )
        saved_return = float(
            next(r for r in summary if float(r["actual_gravity"]) == g)["return_value"]
        )
        if not math.isclose(
            sum(float(r["reward"]) for r in rr), saved_return, abs_tol=1e-8
        ):
            raise ValueError(
                "Trace rewards do not match the corresponding full-episode return"
            )
        for j, r in enumerate(rr):
            probs = [float(r["posterior_" + c]) for c in codes]
            if not (
                all(math.isfinite(v) and 0 <= v <= 1 + 1e-6 for v in probs)
                and math.isclose(sum(probs), 1, abs_tol=1e-6)
            ):
                raise ValueError(
                    "Posterior rows must be finite probability distributions"
                )
            if not math.isclose(
                float(r["outcome_time_s"]), (j + 1) * 0.05, abs_tol=1e-10
            ):
                raise ValueError(
                    "Posterior rows must align with the 0.05-second outcome boundaries"
                )
    src = top
    doc = fitz.open()
    p = doc.new_page(width=468, height=390)
    top_rect = fitz.Rect(0, 0, 468, 174)
    p.show_pdf_page(top_rect, src, 0, clip=top_rect)
    text(p, 251, 184, "Actual gravity", 8.5, align="center")
    plot_x0, plot_x1 = 47.0, 425.0
    panel_specs = [
        (2.0, 202.0, 270.0, 196.0, "Actual 2x gravity (interpolation)"),
        (5.0, 292.0, 360.0, 285.0, "Actual 5x gravity (extrapolation)"),
    ]
    cells = []

    def color(prob):
        value = min(max(prob, 0), 1)
        lo, hi, alpha = (
            (CREAM, TEAL, value * 2) if value <= 0.5 else (TEAL, INK, (value - 0.5) * 2)
        )
        return tuple(a + (b - a) * alpha for a, b in zip(lo, hi))

    def xx(t):
        return plot_x0 + t / 25 * (plot_x1 - plot_x0)

    for g, y0, y1, title_y, title in panel_specs:
        text(p, plot_x0, title_y, title, 8.8, True)
        row_h = (y1 - y0) / 8
        shape = p.new_shape()
        for i, c in enumerate(codes):
            cy0 = y0 + i * row_h
            cy1 = cy0 + row_h
            text(p, 40, cy0 + row_h / 2 + 2.8, c, 8.2, align="right")
            for r in selected[g]:
                time = float(r["outcome_time_s"])
                prob = float(r["posterior_" + c])
                # Observed samples are centered at their outcome times. Restrict the
                # final cell to the actual 25 s episode boundary; no initial prior.
                lo = max(0, time - 0.025)
                hi = min(25, time + 0.025)
                rect = fitz.Rect(xx(lo), cy0, xx(hi), cy1)
                shape.draw_rect(rect)
                shape.finish(fill=color(prob), color=None, width=0)
                cells.append(
                    {
                        "g": g,
                        "condition": c,
                        "index": int(r["transition_index"]),
                        "outcome_time_s": time,
                        "probability": prob,
                        "rect": list(rect),
                    }
                )
        shape.commit()
        p.draw_rect(
            fitz.Rect(plot_x0, y0, plot_x1, y1), color=(0.55, 0.55, 0.55), width=0.4
        )
        for i in range(1, 8):
            p.draw_line(
                (plot_x0, y0 + i * row_h),
                (plot_x1, y0 + i * row_h),
                color=(0.9, 0.9, 0.9),
                width=0.2,
            )
        for t in [0, 5, 10, 15, 20, 25]:
            p.draw_line((xx(t), y1), (xx(t), y1 + 3), color=INK, width=0.4)

    # One common posterior scale, unchanged by the actual gravity level.
    text(p, 442, 196, "Weight", 8.2, align="center")

    # A PDF axial shading gives a continuous colorbar without seams between strips.
    def rgb_pdf(value):
        return " ".join(f"{v:.10f}" for v in value)

    function_refs = []
    for lo, hi in [(CREAM, TEAL), (TEAL, INK)]:
        ref = doc.get_new_xref()
        doc.update_object(
            ref,
            f"<< /FunctionType 2 /Domain [0 1] /C0 [{rgb_pdf(lo)}] /C1 [{rgb_pdf(hi)}] /N 1 >>",
        )
        function_refs.append(ref)
    joined = doc.get_new_xref()
    doc.update_object(
        joined,
        f"<< /FunctionType 3 /Domain [0 1] /Functions [{function_refs[0]} 0 R {function_refs[1]} 0 R] /Bounds [0.5] /Encode [0 1 0 1] >>",
    )
    shade = doc.get_new_xref()
    doc.update_object(
        shade,
        f"<< /ShadingType 2 /ColorSpace /DeviceRGB /Coords [438 30 438 188] /Function {joined} 0 R /Extend [true true] >>",
    )
    resource = int(doc.xref_get_key(p.xref, "Resources")[1].split()[0])
    doc.xref_set_key(resource, "Shading", f"<< /BeliefScale {shade} 0 R >>")
    stream = p.get_contents()[-1]
    doc.update_stream(
        stream, doc.xref_stream(stream) + b"\nq 438 30 8 158 re W n /BeliefScale sh Q\n"
    )
    p.draw_rect(fitz.Rect(438, 202, 446, 360), color=(0.55, 0.55, 0.55), width=0.4)
    for value in [0, 0.5, 1]:
        yy = 360 - value * 158
        p.draw_line((446, yy), (449, yy), color=INK, width=0.4)
        text(p, 451, yy + 2.8, f"{value:g}", 8.2)
    for t in [0, 5, 10, 15, 20, 25]:
        text(p, xx(t), 374, str(t), 8.2, align="center")
    text(p, (plot_x0 + plot_x1) / 2, 387, "Outcome time (s)", 8.5, align="center")

    doc.set_metadata({"title": "Gravity sweep return and evolving posterior beliefs"})
    doc.save(output_dir / "gravity_summary.pdf", garbage=4, deflate=True)
    p.get_pixmap(matrix=fitz.Matrix(2, 2)).save(output_dir / "gravity_summary.png")
