# Mesh Quality Review

A Python script that reviews the quality of a 2D first-order finite-element mesh
(Quad/Tria) exported from Abaqus as an `.inp` file, and produces a text and
Excel quality report.

It works irrespective of element type or analysis type, as long as the
elements are first-order 2D planar elements (e.g. `S4R`, `S4`, `CPS4`, `CPE4`,
`CAX4`, `M3D4`, `S3R`, `S3`, `CPS3`, ...). Second-order elements and 3D solid
elements are detected and skipped, with a count reported separately.

## Quality checks

| Metric | Fail condition | Notes |
|---|---|---|
| Aspect ratio | `>= 5.0` | Quad: ratio of bimedian lengths. Tria: circumradius / (2 x inradius). |
| Skew | `> 45 deg` | Quad: deviation of the bimedian intersection angle from 90 deg. Tria: deviation of corner angles from 60 deg. |
| Corner angles | outside `45-135 deg` (Quad) / `30-120 deg` (Tria) | Any corner angle outside range fails the element. |
| Small element length | `< 0.9 mm` | Shortest edge of the element. |

All thresholds are configurable via CLI flags (see below). The report also
states the percentage of Tria elements in the mesh.

## Usage

```bash
pip install -r requirements.txt
python mesh_quality_check.py <path-to-model.inp>
```

Optional flags:

```bash
python mesh_quality_check.py model.inp \
  --out-dir results \
  --ar-max 4 \
  --skew-max 45 \
  --quad-angle-min 45 --quad-angle-max 135 \
  --tria-angle-min 30 --tria-angle-max 120 \
  --min-length 1.0
```

This writes `<model>_mesh_quality.txt` and `<model>_mesh_quality.xlsx`
alongside the input file (or into `--out-dir` if given), and prints the text
report to the console.

The text report includes: total element / Quad / Tria counts, a deviation
table (elements failing each check, split by Quads/Trias/Total), the percent
Tria ratio, and a per-element listing of every failure. The Excel report adds
a `Summary` sheet and a `Failing_Elements` sheet.

## Sample data

This repo includes two example meshes and their generated reports, so you can
try the script immediately:

- [`plate_2holes_100x30_mesh.inp`](plate_2holes_100x30_mesh.inp) — a 100 x 30 mm
  plate with two 8 mm dia holes (see [`Rect_plate_inp.jpg`](Rect_plate_inp.jpg)
  for the sketch), with its generated
  [`.txt`](plate_2holes_100x30_mesh_mesh_quality.txt) and
  [`.xlsx`](plate_2holes_100x30_mesh_mesh_quality.xlsx) reports.
- [`plate_with_hole_mesh.inp`](plate_with_hole_mesh.inp) — a larger plate-with-hole
  mesh, with its generated
  [`.txt`](plate_with_hole_mesh_mesh_quality.txt) and
  [`.xlsx`](plate_with_hole_mesh_mesh_quality.xlsx) reports.

Reproduce either report with:

```bash
python mesh_quality_check.py plate_2holes_100x30_mesh.inp
python mesh_quality_check.py plate_with_hole_mesh.inp
```

## Installation on another system

See [`Installation_other_system.docx`](Installation_other_system.docx) for
step-by-step setup instructions on a fresh machine.

## Requirements

- Python 3.9+
- numpy, pandas, openpyxl (see `requirements.txt`)
