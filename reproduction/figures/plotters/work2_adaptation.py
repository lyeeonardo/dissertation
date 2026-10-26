"""Plot the two paired target resets across the three independent fitting arms."""

import csv
import pymupdf as f


def render(input_dir, output_dir):
    with (input_dir / "adaptation.csv").open(newline="") as stream:
        rows = [
            r
            for r in csv.DictReader(stream)
            if r["case_key"] in ["target_a", "target_b"]
        ]
    METHODS = [
        ("genaif", "GenAIF", "#274753", -12, None),
        ("oracle", "True-condition diffusion", "#BF7628", 12, "[1 2] 0"),
        ("history", "History diffusion", "#299D8F", -4, "[4 2] 0"),
        ("dadp", "DADP", "#8AB07C", 4, "[6 1.5 1 1.5] 0"),
    ]
    ARMS = [
        ("zero", "Before"),
        ("replay_only", "Replay only"),
        ("expert_1000", "Target + replay"),
    ]
    CASES = [("target_a", "triangle", -5.5), ("target_b", "square", 5.5)]
    FONT = {"regular": f.Font("helv"), "bold": f.Font("hebo")}
    rgb = lambda h: tuple(int(h[i : i + 2], 16) / 255 for i in [1, 3, 5])
    INK = rgb("#274753")
    NEUTRAL = (0.5, 0.5, 0.47)
    GRID = (0.88, 0.88, 0.86)
    points = []
    means = []

    def check(name, ok, **extra):
        if not ok:
            raise ValueError(f"{name}: {extra}")

    def text(p, x, y, value, size=8.5, bold=False, align="left", color=INK):
        kind = "bold" if bold else "regular"
        name = "SingleUpdateBold" if bold else "SingleUpdateRegular"
        p.insert_font(fontname=name, fontbuffer=FONT[kind].buffer)
        width = FONT[kind].text_length(value, fontsize=size)
        if align == "center":
            x -= width / 2
        if align == "right":
            x -= width
        p.insert_text((x, y), value, fontname=name, fontsize=size, color=color)
        check(
            "label_geometry",
            x >= 0 and x + width <= 468 and y - size >= 0 and y <= 280,
            text=value,
        )

    def marker(p, x, y, shape, color, fill=None, size=2.3):
        if shape == "circle":
            p.draw_circle((x, y), size, color=color, fill=fill, width=1)
        elif shape == "square":
            p.draw_rect(
                f.Rect(x - size, y - size, x + size, y + size),
                color=color,
                fill=fill,
                width=0.9,
            )
        else:
            sh = p.new_shape()
            sh.draw_polyline(
                [
                    (x, y - size),
                    (x + size, y + size),
                    (x - size, y + size),
                    (x, y - size),
                ]
            )
            sh.finish(color=color, fill=fill, width=0.9, closePath=True)
            sh.commit()

    def row(method, arm, case):
        rr = [
            r
            for r in rows
            if (r["method"], r["arm"], r["case_key"]) == (method, arm, case)
        ]
        check("unique_source_point", len(rr) == 1, method=method, arm=arm, case=case)
        return rr[0]

    check("exact_target_population", len(rows) == 24)
    check(
        "all_four_controllers",
        set(r["method"] for r in rows) == {m[0] for m in METHODS},
    )
    check("all_three_arms", set(r["arm"] for r in rows) == {a[0] for a in ARMS})
    for r in rows:
        check(
            "full_25_second_5x_episode_in_bank_1_5",
            float(r["actual_gravity"]) == 5
            and float(r["bank_high"]) == 5
            and int(r["native_steps"]) == 500
            and int(r["assigned_steps"]) == 500,
            method=r["method"],
            arm=r["arm"],
            case=r["case_key"],
        )
    for case, _, _ in CASES:
        check(
            "paired_reset_and_sampling_seeds",
            len(
                {
                    (r["reset_seed"], r["sampling_seed"])
                    for r in rows
                    if r["case_key"] == case
                }
            )
            == 1,
            case=case,
        )

    doc = f.open()
    p = doc.new_page(width=468, height=280)
    text(p, 13, 14, "25-second return", 9.5, True)
    marker(p, 182, 10.5, "circle", INK, INK, 3)
    text(p, 191, 14, "Mean")
    marker(p, 245, 10.5, "triangle", NEUTRAL, (1, 1, 1), 2.5)
    text(p, 254, 14, "Reset A")
    marker(p, 322, 10.5, "square", NEUTRAL, (1, 1, 1), 2.3)
    text(p, 331, 14, "Reset B")

    YMIN, YMAX = -200.0, 500.0
    left, right, top, bottom = 43, 319, 39, 242
    xs = [78, 181, 284]
    yy = lambda value: bottom - (value - YMIN) / (YMAX - YMIN) * (bottom - top)
    for tick in [-200, -100, 0, 100, 200, 300, 400, 500]:
        y = yy(tick)
        p.draw_line(
            (left, y),
            (right, y),
            color=NEUTRAL if tick == 0 else GRID,
            width=0.65 if tick == 0 else 0.35,
        )
        text(p, left - 6, y + 2.9, str(tick), align="right")
    p.draw_line((left, top), (left, bottom), color=(0.6, 0.6, 0.58), width=0.45)
    p.draw_line((left, bottom), (right, bottom), color=(0.6, 0.6, 0.58), width=0.45)
    for x, (_, label) in zip(xs, ARMS):
        p.draw_line((x, bottom), (x, bottom + 3), color=INK, width=0.45)
        text(p, x, 262, label, 9, align="center")

    for method, label, hexcolor, offset, dash in METHODS:
        color = rgb(hexcolor)
        for i, (arm, arm_label) in enumerate(ARMS):
            selected = [row(method, arm, case) for case, _, _ in CASES]
            vals = [float(r["return_value"]) for r in selected]
            mean = sum(vals) / 2
            means.append(
                dict(
                    method=method,
                    arm=arm,
                    mean_return=mean,
                    x=xs[i] + offset,
                    y=yy(mean),
                    label=f"{mean:.2f}",
                )
            )
            for record, (case, shape, reset_offset), value in zip(
                selected, CASES, vals
            ):
                check(
                    "all_return_points_within_range",
                    YMIN <= value <= YMAX,
                    method=method,
                    arm=arm,
                    case=case,
                )
                points.append(
                    dict(
                        method=method,
                        arm=arm,
                        case_key=case,
                        return_value=record["return_value"],
                        x=xs[i] + offset + reset_offset,
                        y=yy(value),
                        marker=shape,
                        reset_seed=record["reset_seed"],
                        sampling_seed=record["sampling_seed"],
                    )
                )
        mm = [m for m in means if m["method"] == method]
        for a, b in zip(mm, mm[1:]):
            p.draw_line(
                (a["x"], a["y"]), (b["x"], b["y"]), color=color, width=1.65, dashes=dash
            )
        for point in [v for v in points if v["method"] == method]:
            marker(p, point["x"], point["y"], point["marker"], color, (1, 1, 1), 2.25)
        for m in mm:
            marker(p, m["x"], m["y"], "circle", color, color, 2.9)
        end = mm[-1]
        p.draw_line(
            (end["x"] + 7, end["y"]),
            (334, end["y"]),
            color=color,
            width=0.7,
            dashes=dash,
        )
        text(p, 340, end["y"] - 2, label, 8.5, True, color=color)
        text(p, 340, end["y"] + 10, f"{end['mean_return']:.2f}", 8.5, color=color)

    doc.set_metadata({"title": "Small-data model updating at 5x gravity"})
    doc.save(output_dir / "small_data_update.pdf", garbage=4, deflate=True)
    p.get_pixmap(matrix=f.Matrix(2.2, 2.2)).save(output_dir / "small_data_update.png")
