"""Check 2D first-order mesh quality (Quad/Tria) from an Abaqus .inp file.

Reads *NODE / *ELEMENT data (plain part-level, or *Part + *Assembly/*Instance
with translation/rotation), classifies every first-order 2D element as a
Quad (4-node) or Tria (3-node) irrespective of its Abaqus element-type label
(S4R, S4, CPS4, CPE4, CAX4, M3D4, S3R, S3, CPS3, ... all treated the same),
and evaluates four shape-quality metrics against user-set thresholds:

    1. Aspect ratio       (fail if  >= --ar-max,      default 5)
    2. Skew                (fail if  >  --skew-max,    default 45 deg)
    3. Corner angles       (fail if any corner angle outside
                             [--quad-angle-min, --quad-angle-max] for quads,
                             default 45-135 deg, or
                             [--tria-angle-min, --tria-angle-max] for trias,
                             default 30-120 deg)
    4. Small element length (fail if the shortest edge of the element is
                              <  --min-length, default 0.9 mm)

Second-order elements (elements with midside nodes, i.e. node counts other
than 3 or 4) and 3D solid element types are skipped and reported separately.

Metric definitions (documented here since these are not single universal
standards -- adjust the formulas below if your review process uses a
different convention):

  Quad aspect ratio : ratio of the two bimedian lengths (lines joining
                       midpoints of opposite edges). 1.0 for a square.
  Quad skew         : |90 deg - angle between the two bimedians|.

  Tria aspect ratio : circumradius / (2 x inradius). 1.0 for an
                       equilateral triangle (this is the standard
                       "radius ratio" shape metric).
  Tria skew         : max(theta_max - 60, 60 - theta_min) over the
                       triangle's 3 corner angles (deviation from the
                       60 deg equilateral angle, in degrees).

  Small element length : the shortest edge length of the element
                          (min over 4 edges for Quad, 3 edges for Tria).

Usage:
    python mesh_quality_check.py plate_with_hole_mesh.inp
    python mesh_quality_check.py model.inp --out-dir results --ar-max 4 --min-length 1.0
"""

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# .inp parsing
# ---------------------------------------------------------------------------

# Abaqus element-type prefixes that are never 2D planar elements, even when
# their node count happens to be 3 or 4 (e.g. C3D4 is a 4-node tet).
NON_2D_TYPE_PREFIXES = ("C3D", "DC3D", "AC3D", "DCC3D", "COH3D", "SC", "C3D4")


def _split_data_line(line: str) -> list:
    return [tok.strip() for tok in line.strip().rstrip(",").split(",")]


def _parse_keyword_line(line: str) -> tuple:
    parts = [p.strip() for p in line.lstrip("*").split(",")]
    name = parts[0].upper()
    params = {}
    for p in parts[1:]:
        if "=" in p:
            k, v = p.split("=", 1)
            params[k.strip().upper()] = v.strip()
        elif p:
            params[p.strip().upper()] = True
    return name, params


def _rotate_point(p: np.ndarray, a: np.ndarray, b: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate point p about the axis a->b by angle_deg degrees (Rodrigues' formula)."""
    k = b - a
    norm = np.linalg.norm(k)
    if norm < 1e-12:
        return p
    k = k / norm
    v = p - a
    theta = np.radians(angle_deg)
    v_rot = (
        v * np.cos(theta)
        + np.cross(k, v) * np.sin(theta)
        + k * np.dot(k, v) * (1 - np.cos(theta))
    )
    return a + v_rot


class RawElement:
    __slots__ = ("elem_id", "etype", "node_ids")

    def __init__(self, elem_id, etype, node_ids):
        self.elem_id = elem_id
        self.etype = etype
        self.node_ids = node_ids


def _parse_part_body(lines, start, end):
    """Parse *Node / *Element lines within [start, end) into local dicts."""
    nodes = {}
    elements = []
    mode = None
    cur_etype = None
    pending = None  # (elem_id, [node tokens so far]) awaiting continuation
    i = start
    while i < end:
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        if stripped.startswith("**"):
            i += 1
            continue
        if stripped.startswith("*"):
            name, params = _parse_keyword_line(stripped)
            if name == "NODE":
                mode = "NODE"
            elif name == "ELEMENT":
                mode = "ELEMENT"
                cur_etype = params.get("TYPE", "")
                pending = None
            else:
                mode = None
            i += 1
            continue

        if mode == "NODE":
            toks = _split_data_line(stripped)
            try:
                nid = toks[0]
                coords = [float(x) for x in toks[1:4]]
                while len(coords) < 3:
                    coords.append(0.0)
                nodes[nid] = np.array(coords, dtype=float)
            except (ValueError, IndexError):
                pass
        elif mode == "ELEMENT":
            toks = _split_data_line(stripped)
            ends_with_comma = stripped.rstrip().endswith(",")
            if pending is None:
                elem_id, node_toks = toks[0], toks[1:]
                pending = [elem_id, list(node_toks)]
            else:
                pending[1].extend(toks)
            if ends_with_comma:
                i += 1
                continue
            elements.append(RawElement(pending[0], cur_etype, pending[1]))
            pending = None
        i += 1
    return nodes, elements


def parse_inp(path: str):
    """Parse an Abaqus .inp file (plain, or *Part/*Assembly/*Instance based).

    Returns (elements, skipped_counts) where elements is a list of dicts:
        {"id": str, "type": str, "shape": "QUAD"|"TRI", "coords": np.ndarray(n,3)}
    and skipped_counts is a dict describing elements that were parsed but
    excluded (second-order / non-2D / degenerate node refs).
    """
    with open(path, "r", errors="replace") as f:
        lines = f.readlines()

    # Find *Part ... *End Part blocks
    part_blocks = {}
    part_start = None
    part_name = None
    top_level_ranges = []  # ranges NOT inside any *Part or *Assembly block
    block_start = 0
    in_special = False

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("*") or stripped.startswith("**"):
            continue
        name, params = _parse_keyword_line(stripped)
        if name == "PART":
            if not in_special:
                top_level_ranges.append((block_start, i))
            part_name = params.get("NAME", f"PART-{len(part_blocks)}").upper()
            part_start = i
            in_special = True
        elif name == "END PART":
            part_blocks[part_name] = (part_start, i)
            in_special = False
            block_start = i + 1
        elif name == "ASSEMBLY":
            if not in_special:
                top_level_ranges.append((block_start, i))
            in_special = True
            assembly_start = i
        elif name == "END ASSEMBLY":
            in_special = False
            block_start = i + 1

    if not in_special:
        top_level_ranges.append((block_start, len(lines)))

    all_nodes = {}   # key -> coords
    all_elements = []  # list of (key_prefix, RawElement)

    # 1) plain top-level *Node/*Element (outside any *Part/*Assembly)
    for (s, e) in top_level_ranges:
        n, el = _parse_part_body(lines, s, e)
        for nid, c in n.items():
            all_nodes[("", nid)] = c
        for elem in el:
            all_elements.append(("", elem))

    # 2) *Part definitions, instantiated directly under their own name
    #    (covers files where *Part bodies are used without *Assembly/*Instance)
    part_nodes = {}
    part_elems = {}
    for pname, (s, e) in part_blocks.items():
        n, el = _parse_part_body(lines, s, e)
        part_nodes[pname] = n
        part_elems[pname] = el

    # 3) *Assembly / *Instance blocks: resolve transforms and instantiate parts
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith("*") and not stripped.startswith("**"):
            name, params = _parse_keyword_line(stripped)
            if name == "INSTANCE":
                inst_name = params.get("NAME", f"INST-{i}").upper()
                ref_part = params.get("PART", "").upper()
                j = i + 1
                translation = None
                rotation = None
                data_lines = []
                while j < len(lines) and not lines[j].strip().upper().startswith("*END INSTANCE"):
                    dl = lines[j].strip()
                    if dl and not dl.startswith("*"):
                        data_lines.append(dl)
                    j += 1
                if len(data_lines) >= 1:
                    try:
                        translation = np.array([float(x) for x in _split_data_line(data_lines[0])[:3]])
                    except ValueError:
                        translation = None
                if len(data_lines) >= 2:
                    try:
                        vals = [float(x) for x in _split_data_line(data_lines[1])]
                        rotation = (np.array(vals[0:3]), np.array(vals[3:6]), vals[6])
                    except (ValueError, IndexError):
                        rotation = None

                src_nodes = part_nodes.get(ref_part, {})
                for nid, c in src_nodes.items():
                    p = c.copy()
                    if rotation is not None:
                        p = _rotate_point(p, rotation[0], rotation[1], rotation[2])
                    if translation is not None:
                        p = p + translation
                    all_nodes[(inst_name, nid)] = p
                for elem in part_elems.get(ref_part, []):
                    all_elements.append((inst_name, elem))
                i = j + 1
                continue
        i += 1

    # Build final element list
    elements = []
    skipped = defaultdict(int)
    for prefix, raw in all_elements:
        n_nodes = len(raw.node_ids)
        etype_up = (raw.etype or "").upper()
        if n_nodes not in (3, 4):
            skipped["second_order_or_unsupported_node_count"] += 1
            continue
        if any(etype_up.startswith(p) for p in NON_2D_TYPE_PREFIXES):
            skipped["non_2d_element_type"] += 1
            continue
        try:
            coords = np.array([all_nodes[(prefix, nid)] for nid in raw.node_ids])
        except KeyError:
            skipped["missing_node_reference"] += 1
            continue
        shape = "QUAD" if n_nodes == 4 else "TRI"
        elements.append({
            "id": f"{prefix}.{raw.elem_id}" if prefix else raw.elem_id,
            "type": raw.etype,
            "shape": shape,
            "coords": coords,
        })

    return elements, dict(skipped)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _angle_at(p_prev: np.ndarray, p_curr: np.ndarray, p_next: np.ndarray) -> float:
    v1 = p_prev - p_curr
    v2 = p_next - p_curr
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_a))


def _edge_lengths(coords: np.ndarray) -> list:
    m = len(coords)
    return [float(np.linalg.norm(coords[(i + 1) % m] - coords[i])) for i in range(m)]


def quad_metrics(coords: np.ndarray) -> dict:
    n0, n1, n2, n3 = coords
    angles = [
        _angle_at(n3, n0, n1),
        _angle_at(n0, n1, n2),
        _angle_at(n1, n2, n3),
        _angle_at(n2, n3, n0),
    ]

    mA = (n0 + n1) / 2.0
    mB = (n2 + n3) / 2.0
    mC = (n1 + n2) / 2.0
    mD = (n3 + n0) / 2.0
    bimedian1 = mB - mA
    bimedian2 = mD - mC
    len1 = np.linalg.norm(bimedian1)
    len2 = np.linalg.norm(bimedian2)
    aspect_ratio = max(len1, len2) / min(len1, len2) if min(len1, len2) > 1e-12 else float("inf")

    if len1 > 1e-12 and len2 > 1e-12:
        cos_a = np.clip(np.dot(bimedian1, bimedian2) / (len1 * len2), -1.0, 1.0)
        angle_between = np.degrees(np.arccos(cos_a))
        skew = abs(90.0 - angle_between)
    else:
        skew = 90.0

    edges = _edge_lengths(coords)

    return {
        "aspect_ratio": aspect_ratio,
        "skew": skew,
        "angles": angles,
        "min_angle": min(angles),
        "max_angle": max(angles),
        "min_edge_length": min(edges),
    }


def tria_metrics(coords: np.ndarray) -> dict:
    n0, n1, n2 = coords
    angles = [
        _angle_at(n2, n0, n1),
        _angle_at(n0, n1, n2),
        _angle_at(n1, n2, n0),
    ]

    a = np.linalg.norm(n1 - n2)
    b = np.linalg.norm(n2 - n0)
    c = np.linalg.norm(n0 - n1)
    s = (a + b + c) / 2.0
    area = np.linalg.norm(np.cross(n1 - n0, n2 - n0)) / 2.0

    if area > 1e-12:
        circumradius = (a * b * c) / (4.0 * area)
        inradius = area / s
        aspect_ratio = circumradius / (2.0 * inradius) if inradius > 1e-12 else float("inf")
    else:
        aspect_ratio = float("inf")

    skew = max(max(angles) - 60.0, 60.0 - min(angles))

    edges = _edge_lengths(coords)

    return {
        "aspect_ratio": aspect_ratio,
        "skew": skew,
        "angles": angles,
        "min_angle": min(angles),
        "max_angle": max(angles),
        "min_edge_length": min(edges),
        "area": area,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_elements(elements: list, thresholds: dict) -> pd.DataFrame:
    rows = []
    for el in elements:
        shape = el["shape"]
        if shape == "QUAD":
            m = quad_metrics(el["coords"])
            amin, amax = thresholds["quad_angle_min"], thresholds["quad_angle_max"]
        else:
            m = tria_metrics(el["coords"])
            amin, amax = thresholds["tria_angle_min"], thresholds["tria_angle_max"]

        ar_fail = m["aspect_ratio"] >= thresholds["ar_max"]
        skew_fail = m["skew"] > thresholds["skew_max"]
        angle_fail = m["min_angle"] < amin or m["max_angle"] > amax
        small_length_fail = m["min_edge_length"] < thresholds["min_length"]

        failed_checks = []
        if ar_fail:
            failed_checks.append("Aspect ratio")
        if skew_fail:
            failed_checks.append("Skew")
        if angle_fail:
            failed_checks.append("Angles")
        if small_length_fail:
            failed_checks.append("Small element length")

        rows.append({
            "element_id": el["id"],
            "type": el["type"],
            "shape": shape,
            "aspect_ratio": m["aspect_ratio"],
            "skew": m["skew"],
            "min_angle": m["min_angle"],
            "max_angle": m["max_angle"],
            "min_edge_length": m["min_edge_length"],
            "ar_fail": ar_fail,
            "skew_fail": skew_fail,
            "angle_fail": angle_fail,
            "small_length_fail": small_length_fail,
            "any_fail": bool(failed_checks),
            "failed_checks": ", ".join(failed_checks),
        })
    return pd.DataFrame(rows)


def build_deviation_table(df: pd.DataFrame) -> pd.DataFrame:
    quads = df[df["shape"] == "QUAD"]
    trias = df[df["shape"] == "TRI"]

    metric_fail_cols = {
        "Aspect ratio": "ar_fail",
        "Skew": "skew_fail",
        "Angles": "angle_fail",
        "Small element length": "small_length_fail",
    }

    data = {"Quads": [], "Trias": [], "Total": []}
    index = []
    for label, col in metric_fail_cols.items():
        index.append(label)
        data["Quads"].append(int(quads[col].sum()) if len(quads) else 0)
        data["Trias"].append(int(trias[col].sum()) if len(trias) else 0)
        data["Total"].append(int(df[col].sum()) if len(df) else 0)

    return pd.DataFrame(data, index=index)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_txt_report(path: str, inp_path: str, df: pd.DataFrame, deviation: pd.DataFrame,
                      skipped: dict, thresholds: dict) -> None:
    n_total = len(df)
    n_quad = int((df["shape"] == "QUAD").sum())
    n_tria = int((df["shape"] == "TRI").sum())
    pct_tria = (n_tria / n_total * 100.0) if n_total else 0.0

    lines = []
    lines.append("=" * 60)
    lines.append("2D FIRST-ORDER MESH QUALITY REPORT")
    lines.append("=" * 60)
    lines.append(f"Input file : {inp_path}")
    lines.append("")
    lines.append("Thresholds:")
    lines.append(f"  Aspect ratio          : < {thresholds['ar_max']}")
    lines.append(f"  Skew                  : <= {thresholds['skew_max']} deg")
    lines.append(f"  Quad corner angles    : {thresholds['quad_angle_min']} - {thresholds['quad_angle_max']} deg")
    lines.append(f"  Tria corner angles    : {thresholds['tria_angle_min']} - {thresholds['tria_angle_max']} deg")
    lines.append(f"  Small element length  : < {thresholds['min_length']} mm")
    lines.append("")
    lines.append(f"1. Total elements : {n_total}")
    lines.append(f"2. No of Quads    : {n_quad}")
    lines.append(f"3. No of Trias    : {n_tria}")
    lines.append("")
    lines.append("Deviation (elements failing threshold):")
    lines.append("")
    header = f"{'':<22}{'Quads':>10}{'Trias':>10}{'Total':>10}"
    lines.append(header)
    for label in deviation.index:
        row = deviation.loc[label]
        lines.append(f"{label:<22}{int(row['Quads']):>10}{int(row['Trias']):>10}{int(row['Total']):>10}")
    lines.append("")
    lines.append(f"Percent of Tria ratio : {pct_tria:.2f}%")

    sections = [
        ("Aspect ratio", "ar_fail", "aspect_ratio", "Aspect ratio", "{:>10.3f}", False),
        ("Skew", "skew_fail", "skew", "Skew (deg)", "{:>10.2f}", False),
        ("Angles", "angle_fail", None, "Min/Max angle (deg)", None, None),
        ("Small element length", "small_length_fail", "min_edge_length", "Min edge length (mm)", "{:>10.4f}", True),
    ]
    for title, fail_col, value_col, value_label, fmt, sort_ascending in sections:
        fails = df[df[fail_col]]
        if value_col:
            fails = fails.sort_values(value_col, ascending=sort_ascending)
        lines.append("")
        lines.append(f"Elements failing {title} : {len(fails)}")
        if len(fails):
            lines.append("")
            if title == "Angles":
                lines.append(f"{'Element ID':<16}{'Shape':<8}{'Type':<10}{'Min angle':>12}{'Max angle':>12}")
                for _, r in fails.iterrows():
                    lines.append(f"{str(r['element_id']):<16}{r['shape']:<8}{str(r['type']):<10}{r['min_angle']:>12.2f}{r['max_angle']:>12.2f}")
            else:
                lines.append(f"{'Element ID':<16}{'Shape':<8}{'Type':<10}{value_label:>22}")
                for _, r in fails.iterrows():
                    lines.append(f"{str(r['element_id']):<16}{r['shape']:<8}{str(r['type']):<10}{fmt.format(r[value_col]):>22}")

    if skipped:
        lines.append("")
        lines.append("Skipped (not evaluated):")
        for reason, count in skipped.items():
            lines.append(f"  {reason.replace('_', ' ')}: {count}")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_excel_report(path: str, inp_path: str, df: pd.DataFrame, deviation: pd.DataFrame,
                        skipped: dict, thresholds: dict) -> None:
    n_total = len(df)
    n_quad = int((df["shape"] == "QUAD").sum())
    n_tria = int((df["shape"] == "TRI").sum())
    pct_tria = (n_tria / n_total * 100.0) if n_total else 0.0

    summary_rows = [
        ("Input file", inp_path),
        ("Total elements", n_total),
        ("No of Quads", n_quad),
        ("No of Trias", n_tria),
        ("Percent of Tria ratio (%)", round(pct_tria, 2)),
        ("", ""),
        ("Aspect ratio threshold", f"< {thresholds['ar_max']}"),
        ("Skew threshold (deg)", f"<= {thresholds['skew_max']}"),
        ("Quad angle range (deg)", f"{thresholds['quad_angle_min']} - {thresholds['quad_angle_max']}"),
        ("Tria angle range (deg)", f"{thresholds['tria_angle_min']} - {thresholds['tria_angle_max']}"),
        ("Small element length threshold (mm)", f"< {thresholds['min_length']}"),
    ]
    for reason, count in skipped.items():
        summary_rows.append((f"Skipped: {reason.replace('_', ' ')}", count))
    summary_df = pd.DataFrame(summary_rows, columns=["Item", "Value"])

    deviation_out = deviation.reset_index().rename(columns={"index": "Deviation"})

    failing_out = df[df["any_fail"]].copy()
    failing_out = failing_out[[
        "element_id", "type", "shape", "failed_checks",
        "aspect_ratio", "skew", "min_angle", "max_angle", "min_edge_length",
    ]].rename(columns={
        "element_id": "Element ID",
        "type": "Element Type",
        "shape": "Shape",
        "failed_checks": "Failed Checks",
        "aspect_ratio": "Aspect Ratio",
        "skew": "Skew (deg)",
        "min_angle": "Min Angle (deg)",
        "max_angle": "Max Angle (deg)",
        "min_edge_length": "Min Edge Length (mm)",
    })

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False, startrow=0)
        deviation_out.to_excel(writer, sheet_name="Summary", index=False, startrow=len(summary_df) + 2)
        failing_out.to_excel(writer, sheet_name="Failing_Elements", index=False)

    from openpyxl import load_workbook
    from openpyxl.utils import get_column_letter

    wb = load_workbook(path)
    ws = wb["Summary"]
    ws.cell(row=len(summary_df) + 2, column=1, value="Deviation (elements failing threshold)")
    for col_idx in range(1, 5):
        ws.column_dimensions[get_column_letter(col_idx)].width = 26

    ws2 = wb["Failing_Elements"]
    for col_idx in range(1, 10):
        ws2.column_dimensions[get_column_letter(col_idx)].width = 18

    wb.save(path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inp_path", help="Path to the Abaqus .inp file")
    parser.add_argument("--out-dir", default=None, help="Directory for outputs (default: alongside the .inp file)")
    parser.add_argument("--ar-max", type=float, default=5.0, help="Aspect ratio fail threshold (default 5)")
    parser.add_argument("--skew-max", type=float, default=45.0, help="Skew fail threshold in degrees (default 45)")
    parser.add_argument("--quad-angle-min", type=float, default=45.0)
    parser.add_argument("--quad-angle-max", type=float, default=135.0)
    parser.add_argument("--tria-angle-min", type=float, default=30.0)
    parser.add_argument("--tria-angle-max", type=float, default=120.0)
    parser.add_argument("--min-length", type=float, default=0.9, help="Small element length fail threshold in mm (default 0.9)")
    args = parser.parse_args()

    if not os.path.isfile(args.inp_path):
        print(f"Error: file not found: {args.inp_path}", file=sys.stderr)
        sys.exit(1)

    thresholds = {
        "ar_max": args.ar_max,
        "skew_max": args.skew_max,
        "quad_angle_min": args.quad_angle_min,
        "quad_angle_max": args.quad_angle_max,
        "tria_angle_min": args.tria_angle_min,
        "tria_angle_max": args.tria_angle_max,
        "min_length": args.min_length,
    }

    print(f"Parsing {args.inp_path} ...")
    elements, skipped = parse_inp(args.inp_path)
    if not elements:
        print("Error: no first-order 2D Quad/Tria elements found in this file.", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(elements)} first-order 2D elements.")

    df = evaluate_elements(elements, thresholds)
    deviation = build_deviation_table(df)

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.inp_path))
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.inp_path))[0]
    txt_path = os.path.join(out_dir, f"{base}_mesh_quality.txt")
    xlsx_path = os.path.join(out_dir, f"{base}_mesh_quality.xlsx")

    write_txt_report(txt_path, args.inp_path, df, deviation, skipped, thresholds)
    write_excel_report(xlsx_path, args.inp_path, df, deviation, skipped, thresholds)

    with open(txt_path) as f:
        print("\n" + f.read())

    print(f"Saved: {txt_path}")
    print(f"Saved: {xlsx_path}")


if __name__ == "__main__":
    main()
