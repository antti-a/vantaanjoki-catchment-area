"""Unit tests for catchment_area.py on small synthetic rasters.

Run inside the water conda env:

    python -m unittest -v test_catchment_area
"""

import contextlib
import io
import math
import os
import tempfile
import unittest
import warnings
from unittest import mock

import numpy as np
import rasterio
from rasterio import windows
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

import catchment_area as ca
import vantaanjoki_data as data

RES = 2.0


def synthetic(arr):
    """A float32 in-memory raster (EPSG:3067, 2 m cells, NaN nodata)."""
    arr = np.asarray(arr, dtype="float32")
    memfile = MemoryFile()
    with memfile.open(
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:3067",
        transform=from_origin(0.0, arr.shape[0] * RES, RES, RES),
        nodata=float("nan"),
    ) as ds:
        ds.write(arr, 1)
    return memfile


def channel_grid():
    """4 m² background with one west-to-east channel of rising accumulation."""
    arr = np.full((9, 20), 4.0, dtype="float32")
    arr[4, :] = 4.0 * (np.arange(20, dtype="float32") + 2)
    return arr


class GridTests(unittest.TestCase):
    def test_value_reads_cells(self):
        with synthetic(channel_grid()) as mf, mf.open() as src:
            grid = ca.Grid(src)
            self.assertEqual(grid.value(4, 10), 48.0)
            self.assertEqual(grid.value(0, 0), 4.0)

    def test_off_raster_is_nan_not_wrapped(self):
        with synthetic(np.full((4, 4), 8.0, dtype="float32")) as mf, mf.open() as src:
            grid = ca.Grid(src)
            for row, col in ((-1, 0), (0, -1), (-1, -1), (4, 0), (0, 4)):
                self.assertTrue(math.isnan(grid.value(row, col)), (row, col))


class ReadBandTests(unittest.TestCase):
    def test_reads_never_set_array_shape(self):
        # rasterio 1.5.0 turns read(1, ...) 2-D by assigning out.shape, which
        # NumPy >= 2.5 deprecates; every raster read must avoid that path.
        with synthetic(channel_grid()) as mf, mf.open() as src:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "error", message="Setting the shape on a NumPy array"
                )
                win = windows.Window(9, 3, 3, 3)
                self.assertEqual(ca.read_band(src, window=win).shape, (3, 3))
                self.assertEqual(
                    ca.read_band(src, window=win, boundless=True,
                                 fill_value=np.nan)[1, 1],
                    48.0,
                )
                self.assertEqual(ca.water_feature_id(src, 4, 10), 48)
                x, y = src.xy(6, 10)
                ca.analyze(src, None, x, y, snap_radius=10.0)
                ca.delineate(src, 4, 10)


class TraceTests(unittest.TestCase):
    def test_reaches_hit_cell_downstream(self):
        with synthetic(channel_grid()) as mf, mf.open() as src:
            grid = ca.Grid(src)
            name, crossed = ca.trace_tributary(grid, 4, 2, {(4, 15): "Testoja"})
            self.assertEqual(name, "Testoja")
            self.assertFalse(crossed)

    def test_confluence_jump_sets_crossed(self):
        arr = channel_grid()
        arr[4, 10:] = 400.0 + 4.0 * np.arange(10, dtype="float32")
        with synthetic(arr) as mf, mf.open() as src:
            grid = ca.Grid(src)
            name, crossed = ca.trace_tributary(grid, 4, 2, {(4, 15): "Testoja"})
            self.assertEqual(name, "Testoja")
            self.assertTrue(crossed)

    def test_plateau_hit_and_plateau_exit(self):
        arr = np.full((9, 20), 4.0, dtype="float32")
        arr[4, 2:10] = 2.0e7  # above PLATEAU_MIN: equal-valued channel cells
        arr[4, 10] = 2.05e7
        with synthetic(arr) as mf, mf.open() as src:
            grid = ca.Grid(src)
            name, _ = ca.trace_tributary(grid, 4, 3, {(4, 8): "Plateauoja"})
            self.assertEqual(name, "Plateauoja")
            name, _ = ca.trace_tributary(grid, 4, 3, {(4, 10): "Exitoja"})
            self.assertEqual(name, "Exitoja")

    def test_dead_end_returns_none(self):
        with synthetic(channel_grid()) as mf, mf.open() as src:
            grid = ca.Grid(src)
            name, _ = ca.trace_tributary(grid, 4, 2, {(0, 0): "Elsewhere"})
            self.assertIsNone(name)


class SnapTests(unittest.TestCase):
    def test_max_accumulation_cell_prefers_nearest_on_ties(self):
        arr = np.full((11, 11), 4.0, dtype="float32")
        arr[5, 7] = 100.0
        arr[5, 2] = 100.0
        with synthetic(arr) as mf, mf.open() as src:
            x, y = src.xy(5, 5)
            self.assertEqual(ca.max_accumulation_cell(src, x, y, 10.0), (5, 7))

    def test_snap_hysteresis_keeps_stream_points_put(self):
        arr = np.full((11, 11), 4.0, dtype="float32")
        arr[5, 5] = 95.0
        arr[5, 7] = 100.0  # bigger, but below SNAP_FACTOR * 95
        with synthetic(arr) as mf, mf.open() as src:
            grid = ca.Grid(src)
            x, y = src.xy(5, 5)
            row, col, snap_line = ca.snap(grid, x, y, 5, 5, 10.0)
            self.assertEqual((row, col), (5, 5))
            self.assertIsNone(snap_line)

    def test_snap_moves_when_clearly_better(self):
        arr = np.full((11, 11), 4.0, dtype="float32")
        arr[5, 5] = 50.0
        arr[5, 7] = 100.0
        with synthetic(arr) as mf, mf.open() as src:
            grid = ca.Grid(src)
            x, y = src.xy(5, 5)
            row, col, snap_line = ca.snap(grid, x, y, 5, 5, 10.0)
            self.assertEqual((row, col), (5, 7))
            self.assertIn("Snapped to:", snap_line)
            self.assertIn("4.0 m E", snap_line)


class AnalyzeTests(unittest.TestCase):
    def test_point_query_without_water_raster(self):
        with synthetic(channel_grid()) as mf, mf.open() as src:
            x, y = src.xy(4, 10)
            report = ca.analyze(src, None, x, y, no_snap=True)
            self.assertEqual(report.area_m2, 48.0)
            self.assertFalse(report.is_lake)
            self.assertIsNone(report.tributary)

    def test_snapping_moves_to_the_channel(self):
        with synthetic(channel_grid()) as mf, mf.open() as src:
            x, y = src.xy(6, 10)  # two cells south of the channel
            report = ca.analyze(src, None, x, y, snap_radius=10.0)
            self.assertEqual(report.area_m2, 64.0)  # best channel cell in reach
            self.assertIsNotNone(report.snap_line)

    def test_outside_raster_raises(self):
        with synthetic(channel_grid()) as mf, mf.open() as src:
            with self.assertRaises(ca.CatchmentError):
                ca.analyze(src, None, -100.0, -100.0)


def forward_members(src, pour):
    """Brute-force oracle: trace every finite cell forward exactly like
    trace_tributary and collect the cells whose path reaches pour (or the
    pour cell's plateau unit)."""
    grid = ca.Grid(src)
    pour_acc = grid.value(*pour)
    pour_unit = {pour}
    if pour_acc >= ca.PLATEAU_MIN:
        pour_unit = ca.plateau_component(grid.value, *pour, pour_acc)[0]
    members = set()
    for r in range(src.height):
        for c in range(src.width):
            if math.isnan(grid.value(r, c)):
                continue
            rr, cc = r, c
            acc = grid.value(rr, cc)
            while True:
                if (rr, cc) in pour_unit:
                    members.add((r, c))
                    break
                if acc > pour_acc:
                    break
                if acc >= ca.PLATEAU_MIN:
                    _cells, nxt, nxt_acc, _capped = ca.plateau_component(
                        grid.value, rr, cc, acc
                    )
                else:
                    nxt, nxt_acc = None, math.inf
                    for dr, dc in ca.OFFSETS:
                        v = grid.value(rr + dr, cc + dc)
                        if acc < v < nxt_acc:
                            nxt_acc, nxt = v, (rr + dr, cc + dc)
                if nxt is None:
                    break
                (rr, cc), acc = nxt, nxt_acc
    return members


def delineate_set(src, row, col, **kw):
    """delineate() result as a set of absolute (row, col) cells."""
    mask, r0, c0 = ca.delineate(src, row, col, **kw)
    return {(r0 + int(r), c0 + int(c)) for r, c in zip(*np.nonzero(mask))}


class DelineateTests(unittest.TestCase):
    def test_channel_catchment(self):
        with synthetic(channel_grid()) as mf, mf.open() as src:
            members = delineate_set(src, 4, 10)
            self.assertEqual(members, forward_members(src, (4, 10)))
            self.assertIn((4, 0), members)      # channel head drains through
            self.assertIn((3, 11), members)     # bank cell draining to the pour
            self.assertNotIn((4, 11), members)  # downstream of the pour

    def test_single_row_is_exactly_consistent(self):
        # acc 4, 8, 12, ... is a consistent accumulation: the catchment of
        # (0, k) is exactly the k+1 cells upstream, i.e. acc/4 cells
        arr = 4.0 * np.arange(1, 21, dtype="float32")[None, :]
        with synthetic(arr) as mf, mf.open() as src:
            for k in (0, 7, 19):
                members = delineate_set(src, 0, k)
                self.assertEqual(members, {(0, j) for j in range(k + 1)})
                self.assertEqual(len(members) * 4.0, float(arr[0, k]))

    def test_receiver_ties_first_offset_wins(self):
        # (5, 5) has two equal-lowest greater neighbours; the trace picks
        # the first in OFFSETS order (NW), so (5, 5) drains NW, not SE
        arr = np.full((11, 11), 4.0, dtype="float32")
        arr[4, 4] = 8.0
        arr[6, 6] = 8.0
        arr[3, 3] = 12.0
        arr[7, 7] = 12.0
        with synthetic(arr) as mf, mf.open() as src:
            self.assertIn((5, 5), delineate_set(src, 3, 3))
            self.assertNotIn((5, 5), delineate_set(src, 7, 7))
            for pour in ((3, 3), (7, 7)):
                self.assertEqual(delineate_set(src, *pour),
                                 forward_members(src, pour))

    def test_plateau_catchment(self):
        arr = np.full((9, 20), 4.0, dtype="float32")
        arr[4, 2:10] = 2.0e7  # above PLATEAU_MIN: equal-valued channel cells
        arr[4, 10] = 2.05e7
        with synthetic(arr) as mf, mf.open() as src:
            below = delineate_set(src, 4, 10)  # pour just below the plateau
            self.assertEqual(below, forward_members(src, (4, 10)))
            for c in range(2, 10):
                self.assertIn((4, c), below)
            on = delineate_set(src, 4, 5)      # pour on the plateau
            self.assertEqual(on, forward_members(src, (4, 5)))
            self.assertNotIn((4, 10), on)      # the plateau's exit cell

    def test_nan_edges(self):
        arr = channel_grid()
        arr[0, :] = np.nan
        arr[-1, :] = np.nan
        arr[:, 0] = np.nan
        arr[:, -1] = np.nan
        with synthetic(arr) as mf, mf.open() as src:
            members = delineate_set(src, 4, 10)
            self.assertEqual(members, forward_members(src, (4, 10)))
            for r, c in members:
                self.assertFalse(math.isnan(float(arr[r, c])), (r, c))

    def test_growth_from_tiny_window(self):
        # a 5x5 start window forces several asymmetric growth rounds
        with synthetic(channel_grid()) as mf, mf.open() as src:
            self.assertEqual(
                delineate_set(src, 4, 17, init_half=2),
                forward_members(src, (4, 17)),
            )


class WriteRasterTests(unittest.TestCase):
    def test_filename(self):
        self.assertEqual(
            ca.catchment_filename(387823.0, 6697699.0),
            "catchment_x387823_y6697699.tiff",
        )

    def test_write_and_reopen(self):
        with synthetic(channel_grid()) as mf, mf.open() as src:
            mask, r0, c0 = ca.delineate(src, 4, 10)
            n = int(mask.sum())
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "catchment_test.tiff")
                for _ in range(2):  # the second write overwrites silently
                    ca.write_catchment_raster(
                        path, src, mask, r0, c0, src.xy(4, 10), 48.0, "acc.tif"
                    )
                with rasterio.open(path) as dst:
                    self.assertEqual(dst.dtypes[0], "uint8")
                    self.assertEqual(dst.nodata, 0.0)
                    self.assertEqual(dst.crs.to_epsg(), 3067)
                    self.assertEqual(
                        dst.transform,
                        windows.transform(
                            windows.Window(c0, r0, mask.shape[1], mask.shape[0]),
                            src.transform,
                        ),
                    )
                    band = ca.read_band(dst)
                    self.assertEqual(int(band.sum()), n)
                    self.assertLessEqual(set(np.unique(band)), {0, 1})
                    self.assertIn("pour cell", dst.descriptions[0])
                    tags = dst.tags()
                    self.assertEqual(tags["pour_x"], "21")
                    self.assertEqual(tags["pour_y"], "9")
                    self.assertEqual(tags["area_m2"], "48")
                    self.assertEqual(tags["member_cells"], str(n))
                    self.assertEqual(tags["source_raster"], "acc.tif")


class ReportPourCellTests(unittest.TestCase):
    def test_no_snap_pour_is_query_cell(self):
        with synthetic(channel_grid()) as mf, mf.open() as src:
            x, y = src.xy(4, 10)
            report = ca.analyze(src, None, x, y, no_snap=True)
            self.assertEqual((report.pour_row, report.pour_col), (4, 10))

    def test_snapped_pour_is_snap_target(self):
        with synthetic(channel_grid()) as mf, mf.open() as src:
            x, y = src.xy(6, 10)
            report = ca.analyze(src, None, x, y, snap_radius=10.0)
            self.assertEqual(report.area_m2, 64.0)
            self.assertEqual((report.pour_row, report.pour_col), (4, 14))


class MainTests(unittest.TestCase):
    def test_raster_flag_writes_geotiff(self):
        arr = channel_grid()
        with tempfile.TemporaryDirectory() as tmp:
            acc_path = os.path.join(tmp, "acc.tif")
            with rasterio.open(
                acc_path, "w", driver="GTiff",
                height=arr.shape[0], width=arr.shape[1], count=1,
                dtype="float32", crs="EPSG:3067",
                transform=from_origin(0.0, arr.shape[0] * RES, RES, RES),
                nodata=float("nan"),
            ) as ds:
                ds.write(arr, 1)
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    rc = ca.main([
                        "-x", "21", "-y", "9", "--no-snap", "--raster",
                        "--acc-raster", acc_path,
                        "--water-raster", os.path.join(tmp, "missing.tif"),
                    ])
                self.assertEqual(rc, 0)
                name = ca.catchment_filename(21.0, 9.0)
                self.assertTrue(os.path.isfile(os.path.join(tmp, name)))
                self.assertIn("Catchment raster:", buf.getvalue())
                self.assertIn(name, buf.getvalue())
            finally:
                os.chdir(cwd)


class WrapTests(unittest.TestCase):
    SANITY_NOTE = (
        "note: a point within 100 meters SE of the query point has a 81.9x "
        "larger catchment area (0.01901 km², at x=387895 y=6697631)."
    )
    WATER_NOTE = (
        "note: water-features raster not found (C:\\Users\\Somebody\\documents"
        "\\dev\\catchment_area\\04_accumulation\\water_features_vantaanjoki.tif); "
        "lake/river/mire recognition skipped"
    )

    def test_wrap_text_wraps_to_width(self):
        wrapped = ca.wrap_text(self.SANITY_NOTE, width=78).split("\n")
        self.assertGreater(len(wrapped), 1)
        for line in wrapped:
            self.assertLessEqual(len(line), 78, line)
        for cont in wrapped[1:]:
            self.assertTrue(cont.startswith("  "), cont)

    def test_long_path_token_stays_intact(self):
        # break_long_words=False keeps paths copy-pasteable: a single token
        # longer than the width overflows alone instead of being split
        for line in ca.wrap_text(self.WATER_NOTE, width=78).split("\n"):
            if len(line) > 78:
                self.assertNotIn(" ", line.strip(), line)

    def test_wrap_width_bounds(self):
        for cols, want in ((200, 78), (60, 60), (20, 40)):
            with mock.patch.object(
                ca.shutil, "get_terminal_size",
                return_value=os.terminal_size((cols, 50)),
            ):
                self.assertEqual(ca.wrap_width(), want)

    def test_print_report_fits_78_columns(self):
        report = ca.Report(x=387822.0, y=6697699.0, area_m2=232.0)
        report.snap_line = "Snapped to:             x=387813  y=6697699  (9.0 m W)"
        report.notes = [self.SANITY_NOTE]
        report.raster_line = "Catchment raster:       catchment_x387813_y6697699.tiff"
        buf = io.StringIO()
        with mock.patch.object(
            ca.shutil, "get_terminal_size",
            return_value=os.terminal_size((200, 50)),
        ), contextlib.redirect_stdout(buf):
            ca.print_report(report)
        for line in buf.getvalue().splitlines():
            self.assertLessEqual(len(line), 78, line)

    def test_error_message_wraps(self):
        with synthetic(channel_grid()) as mf, mf.open() as src:
            with self.assertRaises(ca.CatchmentError) as cm:
                ca.analyze(src, None, -100.0, -100.0)
        for line in ca.wrap_text(f"error: {cm.exception}", width=78).split("\n"):
            self.assertLessEqual(len(line), 78, line)

    def test_snap_line_is_short(self):
        arr = np.full((11, 11), 4.0, dtype="float32")
        arr[5, 5] = 50.0
        arr[5, 7] = 100.0
        with synthetic(arr) as mf, mf.open() as src:
            grid = ca.Grid(src)
            x, y = src.xy(5, 5)
            _row, _col, snap_line = ca.snap(grid, x, y, 5, 5, 10.0)
            self.assertLessEqual(len(snap_line), 78)
            self.assertIn("4.0 m E", snap_line)


class FormatTests(unittest.TestCase):
    def test_compass(self):
        self.assertEqual(ca.compass(0, 1), "N")
        self.assertEqual(ca.compass(1, 1), "NE")
        self.assertEqual(ca.compass(1, 0), "E")
        self.assertEqual(ca.compass(0, -1), "S")
        self.assertEqual(ca.compass(-1, 0), "W")
        self.assertEqual(ca.compass(0, 0), "")

    def test_number_formatting(self):
        self.assertEqual(ca.fmt(1234567.0), "1 234 567")
        self.assertEqual(ca.fmt_km2(2_500_000.0), "2.50")
        self.assertEqual(ca.fmt_km2(500_000.0), "0.5")
        self.assertEqual(ca.pct(12.5), "12.50 %")
        self.assertEqual(ca.pct(0.05), "0.05 %")
        self.assertEqual(
            ca.fmt_area(1_000_000.0), "  1 000 000 m²  |  100.0 ha  |  1.00 km²"
        )


class DataTests(unittest.TestCase):
    def test_tributary_parents_resolve(self):
        for t in data.TRIBUTARIES:
            if t["parent"] is not None:
                self.assertIn(t["parent"], data.TRIB_BY_NAME, t["name"])

    def test_feature_ids_in_their_ranges(self):
        for lake_id in data.LAKES:
            self.assertTrue(0 < lake_id <= data.LAKE_ID_MAX, lake_id)
        for peat_id in data.PEATLANDS:
            self.assertTrue(data.PEAT_BASE <= peat_id <= data.PEAT_ID_MAX, peat_id)

    def test_main_tributaries_exist(self):
        self.assertLessEqual(set(data.MAIN_TRIBUTARIES), set(data.TRIB_BY_NAME))


if __name__ == "__main__":
    unittest.main()
