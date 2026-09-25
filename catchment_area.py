#!/usr/bin/env python
"""Vantaanjoki catchment area calculator.

Computes the upstream catchment area for any point inside the Vantaanjoki
river basin from a 2 m D8 flow-accumulation raster (values = upstream area
in m²), and reports the share of the whole Vantaanjoki catchment and of the
tributary the point drains through.

Coordinates are ETRS-TM35FIN (EPSG:3067) metres:

    python catchment_area.py -x 387822 -y 6697699

Coordinates copied from QGIS ("369002.59,6704730.30") can be passed as one
argument with --xy:

    python catchment_area.py --xy 369002.59,6704730.30

Without -x/-y the help and the whole-catchment summary are shown (the info
lists the main tributaries; many smaller streams are recognized too).

The query point is snapped to the highest-accumulation cell within a small
radius (default 10 m) — but only if that cell has meaningfully (>= 1.1x)
more accumulation than the exact query cell, so points already on a stream
do not drift downstream. The snapping distance is reported. Water features
are recognized from a companion raster: a query point inside one of the
131 lakes (Ranta10) returns the lake's name and the catchment area at the
lake's outlet; a point in a wide river channel (Ranta10 polygon rivers)
snaps to the channel's thalweg; a point on a mire (OSM wetlands) gets an
informational note.

Tributary identification: the flow path is traced downstream by following
increasing accumulation (no flow-direction raster needed); the first stored
tributary pour cell the path passes identifies the tributary. Pour cells were
derived once from SYKE Valuma-aluejako / vesistöaluejako data (© SYKE,
CC BY 4.0) and snapped to this raster; each lies 20-60 m from an official
SYKE purkupiste. The pour-cell, lake and peatland tables live in
vantaanjoki_data.py. All areas are raster-derived accumulation values.

With --raster the upstream catchment of the (snapped) query point is also
written as a GeoTIFF into the current directory, named after the pour cell
(catchment_x<X>_y<Y>.tiff; uint8, 1 = catchment, nodata 0, source grid and
CRS). The catchment is delineated from the accumulation raster by
inverting the same increasing-accumulation rule the tracing uses. The
raster matches the reported area to a fraction of a percent for
catchments larger than a few hectares; below that the reconstruction
blurs by a cell or two around the true extent (a note reports the
difference).

The analysis is importable: analyze() returns a Report that print_report()
renders. Unit tests: python -m unittest test_catchment_area
"""

import argparse
import contextlib
import math
import os
import shutil
import sys
import textwrap
from dataclasses import dataclass, field

import numpy as np
import rasterio
from rasterio import windows

from vantaanjoki_data import (
    LAKE_ID_MAX,
    LAKES,
    MAIN_TRIBUTARIES,
    PEAT_BASE,
    PEAT_ID_MAX,
    PEATLANDS,
    RIVER_CHANNEL_ID,
    TRIB_BY_NAME,
    TRIBUTARIES,
    WHOLE,
)

try:  # never crash on console encodings that lack a character
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except Exception:
    pass

# Flow-accumulation raster location. Edit this for your setup; a relative
# path is resolved against the directory this script is in.
RASTER_PATH = "04_accumulation/flow_accumulation_vantaanjoki.tif"

# Water-features raster on the same grid as the accumulation raster; the id
# encoding is defined in vantaanjoki_data.py. If the file is missing,
# lake/river/mire recognition is skipped with a note.
WATER_RASTER_PATH = "04_accumulation/water_features_vantaanjoki.tif"

CHANNEL_SNAP_RADIUS = 60.0  # m; search for the thalweg within the channel

# Snap only if the best cell in the search radius has more than SNAP_FACTOR
# times the accumulation of the exact query cell — otherwise the point would
# always drift ~one radius downstream along its own stream.
SNAP_FACTOR = 1.1

# Sanity check: warn if a cell within this radius (m) of the query point has
# more than SANITY_FACTOR times the catchment area of the cell actually used.
SANITY_RADIUS = 100.0
SANITY_FACTOR = 2.0

# Above this accumulation (m²) an equal-valued neighbour can only be float32
# quantization on a large channel; below it, strict +4 m² steps are exact.
PLATEAU_MIN = 1.6e7

# An accumulation jump of at least this factor along the traced path is a
# confluence into a larger river (drives the channel disclaimer wording).
CONFLUENCE_FACTOR = 2.0

OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


class CatchmentError(Exception):
    """A query that cannot be answered (bad point, missing/mismatched data)."""


@dataclass
class Report:
    """Result of one point query, rendered by print_report()."""

    x: float
    y: float
    area_m2: float
    is_lake: bool = False
    headline: str | None = None  # "In lake X" / "In Y channel" / mire note
    snap_line: str | None = None
    tributary: str | None = None  # innermost stored tributary, or None
    notes: list[str] = field(default_factory=list)
    pour_row: int | None = None  # effective pour cell: post-snap, lake outlet
    pour_col: int | None = None
    raster_line: str | None = None  # names the written catchment GeoTIFF


def resolve_raster(path):
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    return path


def compass(dx, dy):
    """8-point compass direction of (dx, dy), grid north up."""
    if dx == 0 and dy == 0:
        return ""
    deg = math.degrees(math.atan2(dx, dy)) % 360
    return ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][int((deg + 22.5) // 45) % 8]


def fmt(v, dec=0):
    return f"{v:,.{dec}f}".replace(",", " ")


def fmt_km2(v_m2):
    km2 = v_m2 / 1e6
    return fmt(km2, 2) if km2 >= 1 else f"{km2:.4g}"


def fmt_area(v_m2):
    ha = v_m2 / 1e4
    ha_s = fmt(ha, 1) if ha >= 1 else f"{ha:.4g}"
    return f"  {fmt(v_m2)} m²  |  {ha_s} ha  |  {fmt_km2(v_m2)} km²"


def pct(p):
    return f"{p:.2f} %" if p >= 0.1 else f"{p:.3g} %"


def wrap_width():
    """Report wrap width: the terminal, but never wider than 78 columns
    (fits a half-screen 1080p window) nor narrower than 40."""
    return max(40, min(shutil.get_terminal_size().columns, 78))


def wrap_text(text, width=None):
    return "\n".join(textwrap.wrap(
        text, width or wrap_width(), subsequent_indent="  ",
        break_long_words=False, break_on_hyphens=False,
    ))


def read_band(src, **kwargs):
    """2-D read of band 1 (window/boundless/fill_value pass through).

    rasterio 1.5.0 makes a single-int ``read(1, ...)`` two-dimensional by
    assigning ``out.shape`` in place, which NumPy >= 2.5 deprecates (fixed
    upstream in rasterio 1.5.1). A one-element index list skips that path;
    ``[0]`` is a view of the same buffer.
    """
    return src.read([1], **kwargs)[0]


class Grid:
    """Cached-window cell access so tracing does not do per-cell tile reads."""

    def __init__(self, src, size=2048):
        self.src = src
        self.size = size
        self.height = src.height
        self.width = src.width
        self.arr = None
        self.r0 = self.c0 = 0

    def value(self, row, col):
        # Off-raster probes must read as nodata; without this guard the
        # max(0, ...) window clamp below would let a negative index wrap
        # around to the far edge of the cached block.
        if not (0 <= row < self.height and 0 <= col < self.width):
            return float("nan")
        s = self.size
        if (
            self.arr is None
            or not (self.r0 <= row < self.r0 + self.arr.shape[0])
            or not (self.c0 <= col < self.c0 + self.arr.shape[1])
        ):
            self.r0 = max(0, row - s // 2)
            self.c0 = max(0, col - s // 2)
            self.arr = read_band(
                self.src,
                window=windows.Window(self.c0, self.r0, s, s),
                boundless=True,
                fill_value=np.nan,
            )
        return float(self.arr[row - self.r0, col - self.c0])


def plateau_scan(grid, row, col, acc, hit_cells, cap=50_000):
    """Walk the connected plateau of cells valued exactly acc.

    Returns (hit_name, exit_rc, exit_acc): the tributary name if a stored
    pour cell lies on the plateau (the scan stops there), else the
    neighbouring cell with the lowest strictly greater accumulation
    (exit_rc None if the bounded scan found neither).
    """
    plateau = {(row, col)}
    stack = [(row, col)]
    exit_rc, exit_acc = None, math.inf
    while stack:
        r, c = stack.pop()
        for dr, dc in OFFSETS:
            rc = (r + dr, c + dc)
            if rc in plateau:
                continue
            v = grid.value(*rc)
            if v == acc and len(plateau) < cap:
                if rc in hit_cells:
                    return hit_cells[rc], None, math.inf
                plateau.add(rc)
                stack.append(rc)
            elif acc < v < exit_acc:
                exit_acc, exit_rc = v, rc
    return None, exit_rc, exit_acc


def trace_tributary(grid, row, col, hit_cells, max_iters=1_000_000):
    """Trace downstream to the first stored tributary pour cell.

    Returns (name or None, crossed) where crossed is True if the path went
    through a >= CONFLUENCE_FACTOR accumulation jump (a confluence into a
    larger river) before reaching the pour cell.
    """
    acc = grid.value(row, col)
    crossed = False
    if math.isnan(acc):
        return None, crossed
    for _ in range(max_iters):
        if (row, col) in hit_cells:
            return hit_cells[(row, col)], crossed
        if acc >= PLATEAU_MIN:
            hit_name, nxt, next_acc = plateau_scan(grid, row, col, acc, hit_cells)
            if hit_name is not None:
                return hit_name, crossed
        else:
            nxt, next_acc = None, math.inf
            for dr, dc in OFFSETS:
                v = grid.value(row + dr, col + dc)
                if acc < v < next_acc:
                    next_acc, nxt = v, (row + dr, col + dc)
        if nxt is None:
            return None, crossed
        if next_acc >= CONFLUENCE_FACTOR * acc:
            crossed = True
        (row, col), acc = nxt, next_acc
    return None, crossed


def plateau_component(value, row, col, acc, cap=1_000_000):
    """Collect the connected plateau of cells valued exactly acc.

    Like plateau_scan, but returns the plateau itself: (cells, exit_rc,
    exit_acc, capped) where cells is the set of plateau cells, exit_rc the
    neighbouring cell with the lowest strictly greater accumulation (None
    if there is none) and capped tells whether the scan hit cap (cells and
    exit are then incomplete). value is a callable (row, col) -> float so
    the flood runs against Grid.value, i.e. the full raster.
    """
    cells = {(row, col)}
    stack = [(row, col)]
    exit_rc, exit_acc = None, math.inf
    capped = False
    while stack:
        r, c = stack.pop()
        for dr, dc in OFFSETS:
            rc = (r + dr, c + dc)
            if rc in cells:
                continue
            v = value(*rc)
            if v == acc:
                if len(cells) >= cap:
                    capped = True
                else:
                    cells.add(rc)
                    stack.append(rc)
            elif acc < v < exit_acc:
                exit_acc, exit_rc = v, rc
    return cells, exit_rc, exit_acc, capped


def receiver_field(src, r0, c0, h, w, strip=1024):
    """Flat receiver pointers for the h x w window at (r0, c0) of src.

    nxt[i] is the flat window index of cell i's receiver: the neighbour
    with the lowest strictly greater accumulation, ties resolved
    first-wins in OFFSETS order exactly like trace_tributary (the strict
    `nb < best` update keeps the earliest minimum). Cells with no
    receiver inside the window point at themselves: nodata cells, local
    maxima, and cells whose receiver lies outside the window (delineate's
    growth loop re-resolves those in a larger window). Also returns every
    plateau candidate as (row, col, acc) in window coordinates: cells
    >= PLATEAU_MIN with an exactly-equal 8-neighbour.
    """
    nxt = np.empty(h * w, dtype=np.int32)
    cands = []
    dr_of = np.array([o[0] for o in OFFSETS], dtype=np.int8)
    dc_of = np.array([o[1] for o in OFFSETS], dtype=np.int8)
    ci = np.arange(w, dtype=np.int32)[None, :]
    for s0 in range(0, h, strip):
        sh = min(strip, h - s0)
        a = read_band(
            src,
            window=windows.Window(c0 - 1, r0 + s0 - 1, w + 2, sh + 2),
            boundless=True,
            fill_value=np.nan,
        )
        core = a[1 : 1 + sh, 1 : 1 + w]
        best = np.full((sh, w), np.inf, dtype=np.float32)
        bidx = np.zeros((sh, w), dtype=np.int8)
        eq = np.zeros((sh, w), dtype=bool)
        with np.errstate(invalid="ignore"):
            for k, (dr, dc) in enumerate(OFFSETS):
                nb = a[1 + dr : 1 + dr + sh, 1 + dc : 1 + dc + w]
                m = (nb > core) & (nb < best)
                best[m] = nb[m]
                bidx[m] = k
                eq |= nb == core
            plat = eq & (core >= PLATEAU_MIN)
        ri = np.arange(s0, s0 + sh, dtype=np.int32)[:, None]
        tr = ri + dr_of[bidx]
        tc = ci + dc_of[bidx]
        ok = np.isfinite(best) & (tr >= 0) & (tr < h) & (tc >= 0) & (tc < w)
        nxt[s0 * w : (s0 + sh) * w] = np.where(ok, tr * w + tc, ri * w + ci).ravel()
        for r, c, v in zip(*np.nonzero(plat), core[plat]):
            cands.append((s0 + int(r), int(c), float(v)))
    return nxt, cands


def pointer_fixpoint(nxt, chunk=1 << 24, max_passes=34):
    """Resolve every pointer chain in nxt to its absorbing cell, in place.

    Pointer doubling (buf[i] = nxt[nxt[i]]) until a pass changes nothing:
    each pass doubles how far every cell looks ahead, so ceil(log2(longest
    path)) passes resolve all chains; self-pointers absorb. Chunked so the
    fancy-indexing temporaries stay bounded.
    """
    buf = np.empty_like(nxt)
    for _ in range(max_passes):
        changed = False
        for i0 in range(0, nxt.size, chunk):
            seg = nxt[nxt[i0 : i0 + chunk]]
            if not changed and not np.array_equal(seg, nxt[i0 : i0 + chunk]):
                changed = True
            buf[i0 : i0 + chunk] = seg
        nxt, buf = buf, nxt
        if not changed:
            break
    return nxt


def delineate(src, row, col, init_half=512):
    """Delineate the upstream catchment of the pour cell (row, col).

    Inverts the trace model: a cell is in the catchment when following
    its receivers (plateau units contracted to their exit cell) reaches
    the pour cell or the pour cell's plateau unit. Returns (mask, row0,
    col0): a bool array cropped to the catchment's bounding box plus its
    offsets in the source grid.

    Runs in a window that starts sized from the already-known catchment
    area (the pour cell's accumulation) and doubles towards any side the
    catchment touches. A member chain that leaves the window must
    re-enter it to reach the pour cell, and the re-entry cell resolves as
    a member on the window border, so border members are a complete
    growth signal even though out-of-window receivers self-absorb.
    """
    grid = Grid(src)
    pour_acc = grid.value(row, col)
    if math.isnan(pour_acc):
        raise CatchmentError("cannot delineate: the pour cell has no data")

    pour_unit = {(row, col)}
    if pour_acc >= PLATEAU_MIN:
        pour_unit = plateau_component(grid.value, row, col, pour_acc)[0]

    half = max(init_half, int(math.sqrt(pour_acc / 4.0) / 2))
    r0 = max(0, min(row - half, min(r for r, _ in pour_unit)))
    c0 = max(0, min(col - half, min(c for _, c in pour_unit)))
    r1 = min(src.height, max(row + half + 1, max(r for r, _ in pour_unit) + 1))
    c1 = min(src.width, max(col + half + 1, max(c for _, c in pour_unit) + 1))

    while True:
        h, w = r1 - r0, c1 - c0
        if h * w > 100_000_000:
            print(
                f"delineate: window {h}x{w} needs ~{h * w * 9 / 1e9:.1f} GB, "
                "this may take a few minutes",
                file=sys.stderr,
            )
        elif h * w > 20_000_000:
            print(f"delineate: window {h}x{w}", file=sys.stderr)
        nxt, cands = receiver_field(src, r0, c0, h, w)

        # contract plateau units to their exit cell; units are flooded on
        # the full raster, so the exit is exact even when the window cuts
        # the unit (out-of-window exits self-absorb like any other border).
        # Units at or above the pour's accumulation can never drain to it
        # (chains strictly increase), so they stay self-absorbed unflooded
        # — this is what keeps a bigger river crossing the window cheap.
        seen = set(pour_unit)
        for wr, wc, acc_c in cands:
            if acc_c >= pour_acc:
                continue
            rc = (r0 + wr, c0 + wc)
            if rc in seen:
                continue
            cells, exit_rc, _exit_acc, capped = plateau_component(
                grid.value, *rc, acc_c
            )
            seen |= cells
            tgt = None
            if (
                not capped
                and exit_rc is not None
                and r0 <= exit_rc[0] < r1
                and c0 <= exit_rc[1] < c1
            ):
                tgt = (exit_rc[0] - r0) * w + (exit_rc[1] - c0)
            for ur, uc in cells:
                if r0 <= ur < r1 and c0 <= uc < c1:
                    i = (ur - r0) * w + (uc - c0)
                    nxt[i] = i if tgt is None else tgt
        pour_flat = (row - r0) * w + (col - c0)
        for ur, uc in pour_unit:
            if r0 <= ur < r1 and c0 <= uc < c1:
                nxt[(ur - r0) * w + (uc - c0)] = pour_flat

        member = (pointer_fixpoint(nxt) == pour_flat).reshape(h, w)

        nr0 = max(0, r0 - h) if r0 > 0 and member[0, :].any() else r0
        nr1 = min(src.height, r1 + h) if r1 < src.height and member[-1, :].any() else r1
        nc0 = max(0, c0 - w) if c0 > 0 and member[:, 0].any() else c0
        nc1 = min(src.width, c1 + w) if c1 < src.width and member[:, -1].any() else c1
        if (nr0, nr1, nc0, nc1) == (r0, r1, c0, c1):
            break
        r0, r1, c0, c1 = nr0, nr1, nc0, nc1

    rows = np.flatnonzero(member.any(axis=1))
    cols = np.flatnonzero(member.any(axis=0))
    mask = member[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
    return mask, r0 + int(rows[0]), c0 + int(cols[0])


def catchment_filename(x, y):
    """Output name for the catchment raster of the pour cell at (x, y)."""
    return f"catchment_x{x:.0f}_y{y:.0f}.tiff"


def write_catchment_raster(path, src, mask, row0, col0, pour_xy, area_m2,
                           source_name):
    """Write mask as a uint8 GeoTIFF (1 = catchment, nodata 0) on src's
    grid windowed at (row0, col0). Overwrites path silently."""
    h, w = mask.shape
    profile = dict(
        driver="GTiff", height=h, width=w, count=1, dtype="uint8",
        crs=src.crs,
        transform=windows.transform(windows.Window(col0, row0, w, h), src.transform),
        nodata=0, compress="deflate", tiled=True,
        blockxsize=512, blockysize=512,
    )
    px, py = pour_xy
    try:
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(mask.astype("uint8"), 1)
            dst.set_band_description(
                1, f"catchment of pour cell x={px:.0f} y={py:.0f}"
            )
            dst.update_tags(
                pour_x=f"{px:.0f}", pour_y=f"{py:.0f}",
                area_m2=f"{area_m2:.0f}", member_cells=str(int(mask.sum())),
                source_raster=source_name, generator="catchment_area.py",
            )
    except OSError as e:
        raise CatchmentError(f"cannot write catchment raster {path}: {e}") from e


def max_accumulation_cell(src, x, y, radius_m, wsrc=None, feature_id=None,
                          chebyshev=False):
    """Cell with the highest accumulation within radius_m of (x, y).

    Distance is Euclidean unless chebyshev=True, which searches the full
    square window (used for the channel thalweg, where the search is already
    constrained to channel cells and must reach diagonally across a wide
    river). Ties are resolved toward the nearest cell. With wsrc/feature_id
    only cells carrying that water-feature id are considered. Returns None
    if no candidate cell has data.
    """
    row, col = src.index(x, y)
    n = max(1, int(math.ceil(radius_m / src.res[0])))
    win = windows.Window(col - n, row - n, 2 * n + 1, 2 * n + 1)
    acc = read_band(src, window=win, boundless=True, fill_value=np.nan)
    rr, cc = np.mgrid[-n : n + 1, -n : n + 1]
    dist = np.hypot(rr, cc) * src.res[0]
    ok = ~np.isnan(acc)
    if not chebyshev:
        ok &= dist <= radius_m
    if wsrc is not None:
        ids = read_band(wsrc, window=win, boundless=True, fill_value=0)
        ok &= ids == feature_id
    if not ok.any():
        return None
    # highest accumulation; ties resolved toward the nearest cell
    i = np.lexsort((dist.ravel(), -np.where(ok, acc, -np.inf).ravel()))[0]
    r, c = np.unravel_index(i, acc.shape)
    return row - n + int(r), col - n + int(c)


def snap(grid, x, y, row, col, radius_m, wsrc=None, feature_id=None,
         chebyshev=False):
    """Move (row, col) to the best nearby cell, with hysteresis.

    The move happens only if the best cell has at least SNAP_FACTOR times
    the accumulation of the current cell (or the current cell is nodata).
    Returns (row, col, snap_line) where snap_line describes the move for
    the report, or None if the point stayed put.
    """
    src = grid.src
    target = max_accumulation_cell(src, x, y, radius_m, wsrc, feature_id,
                                   chebyshev)
    if target is None or target == (row, col):
        return row, col, None
    acc_here = grid.value(row, col)
    acc_there = grid.value(*target)
    if not (math.isnan(acc_here) or acc_there >= SNAP_FACTOR * acc_here):
        return row, col, None
    row, col = target
    sx, sy = src.xy(row, col)
    dist = math.hypot(sx - x, sy - y)
    heading = compass(sx - x, sy - y)
    snap_line = (
        f"Snapped to:             x={sx:.0f}  y={sy:.0f}  "
        f"({dist:.1f} m {heading})"
    )
    return row, col, snap_line


def water_feature_id(wsrc, row, col):
    """Water-feature id at (row, col); 0 = no feature, outside, or no raster."""
    if wsrc is None or not (0 <= row < wsrc.height and 0 <= col < wsrc.width):
        return 0
    return int(read_band(wsrc, window=windows.Window(col, row, 1, 1))[0, 0])


def check_water_grid(src, wsrc, water_path):
    """The water raster must share the accumulation raster's grid exactly:
    row/col indices are used interchangeably between the two."""
    if (
        wsrc.transform != src.transform
        or wsrc.width != src.width
        or wsrc.height != src.height
        or wsrc.crs != src.crs
    ):
        raise CatchmentError(
            "water-features raster is not on the accumulation raster's grid: "
            f"{water_path}"
        )


def analyze(src, wsrc, x, y, snap_radius=10.0, no_snap=False):
    """Analyze one query point and return a Report.

    src is the open flow-accumulation raster; wsrc the open water-features
    raster or None (recognition skipped). Raises CatchmentError for points
    outside the raster or the catchment.
    """
    b = src.bounds
    if not (b.left <= x <= b.right and b.bottom <= y <= b.top):
        raise CatchmentError(
            "point is outside the Vantaanjoki raster "
            f"(x {b.left:.0f}..{b.right:.0f}, y {b.bottom:.0f}..{b.top:.0f}).\n"
            "Coordinates must be ETRS-TM35FIN (EPSG:3067) metres, "
            "e.g. -x 387822 -y 6697699"
        )

    row, col = src.index(x, y)
    grid = Grid(src)
    report = Report(x=x, y=y, area_m2=math.nan)

    feature_id = water_feature_id(wsrc, row, col)
    lake = LAKES.get(feature_id) if 0 < feature_id <= LAKE_ID_MAX else None
    in_channel = feature_id == RIVER_CHANNEL_ID
    if PEAT_BASE <= feature_id <= PEAT_ID_MAX:
        mire = PEATLANDS.get(feature_id)
        report.headline = (
            f"note: the point is on mire {mire}"
            if mire
            else "note: the point is on an unnamed mire"
        )

    if lake is not None:
        name, _tunnus, outlet_x, outlet_y, lake_area_m2 = lake
        report.is_lake = True
        report.area_m2 = lake_area_m2
        report.headline = f"In lake {name}" if name else "In unnamed lake"
        # continue from the lake outlet: its accumulation is the lake's
        # catchment, and the tributary trace starts there
        row, col = src.index(outlet_x, outlet_y)
    elif not no_snap:
        if in_channel:
            # snap to the thalweg: the max-accumulation channel cell nearby
            row, col, report.snap_line = snap(
                grid, x, y, row, col, CHANNEL_SNAP_RADIUS,
                wsrc=wsrc, feature_id=RIVER_CHANNEL_ID, chebyshev=True,
            )
        else:
            row, col, report.snap_line = snap(grid, x, y, row, col, snap_radius)

    report.pour_row, report.pour_col = row, col
    if not report.is_lake:
        report.area_m2 = grid.value(row, col)
    if math.isnan(report.area_m2):
        raise CatchmentError(
            "point is outside the Vantaanjoki catchment (no data). "
            "Coordinates must be ETRS-TM35FIN (EPSG:3067) metres."
        )

    # sanity check: is there a much larger stream near the query point?
    if lake is None and not in_channel:
        larger = max_accumulation_cell(src, x, y, SANITY_RADIUS)
        if larger is not None:
            larger_acc = grid.value(*larger)
            if larger_acc > SANITY_FACTOR * report.area_m2:
                lx, ly = src.xy(*larger)
                heading = compass(lx - x, ly - y)
                ratio = larger_acc / report.area_m2
                ratio_s = f"{ratio:.1f}" if ratio < 100 else fmt(ratio)
                report.notes.append(
                    f"note: a point within {SANITY_RADIUS:g} meters "
                    f"{heading} of the query point has a {ratio_s}x larger "
                    f"catchment area ({fmt_km2(larger_acc)} km², "
                    f"at x={lx:.0f} y={ly:.0f})."
                )

    hit_cells = {src.index(t["hit_x"], t["hit_y"]): t["name"] for t in TRIBUTARIES}
    report.tributary, crossed = trace_tributary(grid, row, col, hit_cells)
    if in_channel:
        # name the channel only if the path reaches the stored river
        # without crossing a confluence into a larger river
        if crossed:
            report.headline = "In a channel"
        else:
            report.headline = f"In {report.tributary or 'Vantaanjoki'} channel"
    return report


def print_report(report):
    print()
    if report.headline:
        print(report.headline)
        print()
    print(f"Query point (TM35FIN):  x={report.x:.0f}  y={report.y:.0f}")
    if report.snap_line:
        print(report.snap_line)
    if report.notes:
        print()
    for note in report.notes:
        print(wrap_text(note))
    print()
    print("Catchment area of the lake:" if report.is_lake else "Upstream catchment area:")
    print(fmt_area(report.area_m2))
    print()
    name = report.tributary
    while name is not None:
        trib = TRIB_BY_NAME[name]
        print(
            f"Share of {name} catchment ({fmt_km2(trib['area_m2'])} km²): "
            f"{pct(100.0 * report.area_m2 / trib['area_m2'])}"
        )
        name = trib["parent"]
    print(
        f"Share of Vantaanjoki catchment ({fmt_km2(WHOLE['area_m2'])} km²): "
        f"{pct(100.0 * report.area_m2 / WHOLE['area_m2'])}"
    )
    if report.raster_line:
        print()
        print(report.raster_line)
    print()


def print_info(parser):
    print()
    print(__doc__.strip().split("\n\n")[0])
    print(f"""
Whole Vantaanjoki catchment:
  {fmt(WHOLE['area_m2'])} m²  |  {fmt(WHOLE['area_m2'] / 1e4, 1)} ha  |  {fmt(WHOLE['area_m2'] / 1e6, 2)} km²
  outlet at x={WHOLE['outlet_x']:.0f} y={WHOLE['outlet_y']:.0f}

Tributary catchments:""")
    for t in TRIBUTARIES:
        if t["name"] not in MAIN_TRIBUTARIES:
            continue
        extra = f"  (drains into {t['parent']})" if t["parent"] else ""
        print(f"  {t['name']:<16} {t['area_m2'] / 1e6:7.2f} km²{extra}")
    print()
    print(parser.format_help(), end="")


def build_parser():
    ap = argparse.ArgumentParser(
        description="Vantaanjoki catchment area calculator (ETRS-TM35FIN / EPSG:3067).",
    )
    ap.add_argument("-x", type=float, help="easting in TM35FIN (m), e.g. 387822")
    ap.add_argument("-y", type=float, help="northing in TM35FIN (m), e.g. 6697699")
    ap.add_argument("--xy", metavar="X,Y",
                    help='both coordinates as one "easting,northing" argument '
                         "(the QGIS clipboard format), e.g. 369002.59,6704730.30")
    ap.add_argument("--snap-radius", type=float, default=10.0, metavar="M",
                    help="snap search radius in metres (default 10)")
    ap.add_argument("--no-snap", action="store_true",
                    help="use the exact query cell, no snapping")
    ap.add_argument("--raster", dest="write_raster", action="store_true",
                    help="write the catchment as a GeoTIFF "
                         "(catchment_x<X>_y<Y>.tiff) into the current directory")
    ap.add_argument("--acc-raster", default=RASTER_PATH, metavar="PATH",
                    help=f"path to the flow accumulation raster (default {RASTER_PATH})")
    ap.add_argument("--water-raster", default=WATER_RASTER_PATH, metavar="PATH",
                    help=f"path to the water-features raster (default {WATER_RASTER_PATH})")
    return ap


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.xy:
        try:
            xs, ys = args.xy.split(",")
            args.x, args.y = float(xs.strip()), float(ys.strip())
        except ValueError:
            parser.error(
                '--xy expects "easting,northing" in TM35FIN, '
                "e.g. --xy 369002.59,6704730.30"
            )

    if args.x is None and args.y is None:
        print_info(parser)
        return 0
    if args.x is None or args.y is None:
        parser.error("both -x and -y are required (TM35FIN metres)")

    raster = resolve_raster(args.acc_raster)
    if not os.path.isfile(raster):
        sys.exit(f"error: raster not found: {raster}")
    water_path = resolve_raster(args.water_raster)

    pre_notes = []
    try:
        with contextlib.ExitStack() as stack:
            src = stack.enter_context(rasterio.open(raster))
            wsrc = None
            if os.path.isfile(water_path):
                wsrc = stack.enter_context(rasterio.open(water_path))
                check_water_grid(src, wsrc, water_path)
            else:
                pre_notes.append(
                    f"note: water-features raster not found ({water_path}); "
                    "lake/river/mire recognition skipped"
                )
            report = analyze(
                src, wsrc, args.x, args.y,
                snap_radius=args.snap_radius, no_snap=args.no_snap,
            )
            if args.write_raster:
                mask, row0, col0 = delineate(src, report.pour_row, report.pour_col)
                n = int(mask.sum())
                cell_m2 = abs(src.res[0]) * abs(src.res[1])
                px, py = src.xy(report.pour_row, report.pour_col)
                name = catchment_filename(px, py)
                write_catchment_raster(name, src, mask, row0, col0, (px, py),
                                       report.area_m2, os.path.basename(raster))
                if abs(n * cell_m2 - report.area_m2) > 0.01 * max(
                    n * cell_m2, report.area_m2
                ):
                    report.notes.append(
                        f"note: the catchment raster covers {fmt(n)} cells = "
                        f"{fmt(n * cell_m2)} m², the reported area is "
                        f"{fmt(report.area_m2)} m². Flow is reconstructed "
                        "from accumulation alone: very small catchments blur "
                        "into neighbouring cells, and lake areas come from "
                        "the SYKE table."
                    )
                report.raster_line = f"Catchment raster:       {name}"
    except CatchmentError as e:
        sys.exit(wrap_text(f"error: {e}"))
    report.notes[:0] = pre_notes
    print_report(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
