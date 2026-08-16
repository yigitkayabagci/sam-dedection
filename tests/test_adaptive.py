"""The adaptive window's geometry and, mostly, its hysteresis.

The window is a segment boundary: every time it moves, the memory bank has to
be re-anchored because SAM 2 stores features in input coordinates. So the
property that matters is not that the window *can* follow a target -- it is
that it **refuses to move for small changes**. A window that chased every frame
would be correct geometrically and useless in practice.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.trackers.adaptive import (  # noqa: E402
    AdaptiveConfig,
    AdaptiveWindow,
    Reacquisition,
    box_side,
    ladder_side,
    tile_grid,
)

FRAME = (640, 512)


def box(cx, cy, side):
    return np.array([cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2])


class TestBoxSide(unittest.TestCase):
    def test_takes_the_longer_side(self):
        self.assertEqual(box_side(np.array([0.0, 0.0, 10.0, 40.0])), 40.0)

    def test_absent_box_is_nan(self):
        self.assertTrue(np.isnan(box_side(np.full(4, np.nan))))


class TestLadderSide(unittest.TestCase):
    def test_rungs_are_geometric_from_the_minimum(self):
        cfg = AdaptiveConfig(min_window=128, scale_step=2.0, max_window=1024)
        self.assertEqual(ladder_side(100, cfg, FRAME), 128)
        self.assertEqual(ladder_side(129, cfg, FRAME), 256)
        self.assertEqual(ladder_side(256, cfg, FRAME), 256)
        self.assertEqual(ladder_side(300, cfg, FRAME), 512)

    def test_never_exceeds_the_shorter_frame_side(self):
        cfg = AdaptiveConfig(min_window=128, max_window=4096)
        self.assertEqual(ladder_side(10_000, cfg, FRAME), min(FRAME))

    def test_a_nan_target_falls_back_to_the_floor(self):
        self.assertEqual(ladder_side(float("nan"), AdaptiveConfig(), FRAME), 128)

    def test_the_floor_is_clamped_to_the_frame(self):
        cfg = AdaptiveConfig(min_window=128)
        self.assertEqual(ladder_side(10, cfg, (64, 64)), 64)


class TestCoordinateMapping(unittest.TestCase):
    def _window(self):
        window = AdaptiveWindow(FRAME, AdaptiveConfig(size=512))
        window.side, window.origin = 128, (64, 96)
        return window

    def test_scale_is_the_magnification(self):
        self.assertAlmostEqual(self._window().scale, 4.0)

    def test_roundtrip(self):
        window = self._window()
        original = np.array([100.0, 120.0, 140.0, 160.0])
        np.testing.assert_allclose(window.to_frame(window.to_window(original)),
                                   original, atol=1e-9)

    def test_a_tiny_target_arrives_at_the_model_much_larger(self):
        # The whole point: a 10 px drone in a 128 px window is a 40 px one at
        # 512, which a stride-16 backbone can actually work with.
        window = self._window()
        mapped = window.to_window(box(128, 160, 10))
        self.assertAlmostEqual(box_side(mapped), 40.0)

    def test_contains(self):
        window = self._window()
        self.assertTrue(window.contains(box(128, 160, 20)))
        self.assertFalse(window.contains(box(300, 160, 20)))


class TestHysteresis(unittest.TestCase):
    def _settled(self, target=20.0, cfg=None):
        cfg = cfg or AdaptiveConfig(target_fraction=0.08, scale_step=1.5, margin=0.2)
        window = AdaptiveWindow(FRAME, cfg)
        window.move_to(box(320, 256, target))
        return window

    def test_a_settled_window_does_not_move_for_the_same_target(self):
        window = self._settled()
        self.assertIsNone(window.needs_move(box(320, 256, 20)))

    def test_small_drift_does_not_move_it(self):
        window = self._settled()
        for offset in range(0, 20, 2):
            self.assertIsNone(window.needs_move(box(320 + offset, 256, 20)),
                              f"moved after only {offset} px of drift")

    def test_small_size_changes_do_not_resize_it(self):
        window = self._settled()
        for side in (18.0, 20.0, 24.0, 28.0):
            self.assertIsNone(window.needs_move(box(320, 256, side)),
                              f"resized for a {side} px target")

    def test_it_moves_when_the_target_nears_an_edge(self):
        window = self._settled()
        x0, _, side = window.rect
        self.assertEqual(window.needs_move(box(x0 + side - 5, 256, 20)), "edge")

    def test_it_resizes_when_the_target_grows_past_a_rung(self):
        window = self._settled(target=20.0)
        self.assertEqual(window.needs_move(box(320, 256, 90.0)), "size")

    def test_it_holds_position_when_there_is_no_target(self):
        window = self._settled()
        before = window.rect
        self.assertIsNone(window.needs_move(np.full(4, np.nan)))
        self.assertEqual(window.rect, before)

    def test_a_steadily_moving_target_moves_the_window_rarely(self):
        # The number that decides whether this is usable: each move costs a
        # memory re-anchor, so a few per hundred frames is fine and one per
        # frame is not.
        window = self._settled()
        for step in range(200):
            target = box(60 + step * 2.5, 256, 20)
            if window.needs_move(target):
                window.move_to(target)
        self.assertLess(window.moves, 20, f"{window.moves} moves over 200 frames")
        self.assertGreater(window.moves, 1, "the window never followed the target")

    def test_a_target_with_room_around_it_ends_up_centred(self):
        window = AdaptiveWindow(FRAME, AdaptiveConfig(target_fraction=0.08))
        target = box(320, 256, 24)
        window.move_to(target)

        self.assertTrue(window.contains(target))
        x0, y0, side = window.rect
        self.assertAlmostEqual(x0 + side / 2, 320, delta=window.config.grid)
        self.assertAlmostEqual(y0 + side / 2, 256, delta=window.config.grid)

    def test_near_an_edge_containment_wins_over_centring(self):
        # A 432 px window cannot be centred on a target 24 px from the corner
        # of a 640x512 frame. Staying inside the frame is what matters; the
        # target being off-centre is not a failure.
        window = AdaptiveWindow(FRAME, AdaptiveConfig(target_fraction=0.08))
        for target in (box(12, 12, 24), box(628, 500, 24)):
            x0, y0, side = window.move_to(target)
            self.assertTrue(window.contains(target))
            self.assertGreaterEqual(x0, 0)
            self.assertGreaterEqual(y0, 0)
            self.assertLessEqual(x0 + side, FRAME[0])
            self.assertLessEqual(y0 + side, FRAME[1])

    def test_the_window_size_tracks_the_requested_fraction(self):
        window = AdaptiveWindow(FRAME, AdaptiveConfig(target_fraction=0.1,
                                                       min_window=32, scale_step=1.25))
        window.move_to(box(320, 256, 40))
        fraction = 40 / window.side
        self.assertGreater(fraction, 0.1 / 1.25)
        self.assertLess(fraction, 0.1 * 1.25)


class TestTileGrid(unittest.TestCase):
    def test_tiles_cover_the_frame(self):
        tiles = tile_grid(FRAME, 256, overlap=0.2)
        covered = np.zeros(FRAME[::-1], dtype=bool)
        for x, y, side in tiles:
            covered[y:y + side, x:x + side] = True
        self.assertTrue(covered.all())

    def test_tiles_stay_inside_the_frame(self):
        for x, y, side in tile_grid(FRAME, 300, overlap=0.3):
            self.assertGreaterEqual(x, 0)
            self.assertGreaterEqual(y, 0)
            self.assertLessEqual(x + side, FRAME[0])
            self.assertLessEqual(y + side, FRAME[1])

    def test_tiles_overlap(self):
        # A target on a boundary is split between two tiles and convincing in
        # neither; overlap is the whole reason to tile rather than to grid.
        tiles = tile_grid((512, 512), 256, overlap=0.25)
        xs = sorted({x for x, _, _ in tiles})
        self.assertGreater(len(xs), 2)
        self.assertLess(xs[1] - xs[0], 256)

    def test_a_tile_as_large_as_the_frame_gives_one_tile(self):
        self.assertEqual(tile_grid((512, 512), 512), [(0, 0, 512)])

    def test_an_oversized_tile_is_clamped_to_the_shorter_side(self):
        # Clamped to 512 by the frame's height -- which then still needs two
        # tiles to cover its 640 px width.
        tiles = tile_grid(FRAME, 4096)
        self.assertTrue(all(side == 512 for _, _, side in tiles))
        self.assertEqual(max(x + side for x, _, side in tiles), FRAME[0])


class TestReacquisition(unittest.TestCase):
    def test_a_brief_loss_does_not_trigger_a_sweep(self):
        # Two frames of occlusion usually resolve themselves; a sweep costs one
        # model pass per tile and must stay rare.
        state = Reacquisition(patience=30)
        for _ in range(5):
            state.observe(present=False)
        self.assertFalse(state.should_sweep())

    def test_a_sustained_loss_triggers_one(self):
        state = Reacquisition(patience=10, cooldown=0)
        for _ in range(10):
            state.observe(present=False)
        self.assertTrue(state.should_sweep())

    def test_finding_the_target_resets_the_counter(self):
        state = Reacquisition(patience=5, cooldown=0)
        for _ in range(4):
            state.observe(present=False)
        state.observe(present=True)
        self.assertFalse(state.should_sweep())

    def test_the_cooldown_stops_it_sweeping_every_frame(self):
        state = Reacquisition(patience=3, cooldown=20)
        for _ in range(3):
            state.observe(present=False)
        self.assertTrue(state.should_sweep())
        state.swept()
        for _ in range(3):
            state.observe(present=False)
        self.assertFalse(state.should_sweep())


if __name__ == "__main__":
    unittest.main()
