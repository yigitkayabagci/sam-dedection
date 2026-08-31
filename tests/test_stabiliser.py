"""The classical guards, on synthetic sequences that reproduce three failures.

Each test builds the case it is about rather than asserting on a recording: a
camera that slides, a mask that balloons into terrain, a box that jumps to a
distractor. The point of doing it this way is that the expected answer is known
by construction -- the frames were made by translating a known pattern by a
known amount -- so a failure here is the guard being wrong, never the ground
truth being arguable.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import cv2
except ImportError:                                      # pragma: no cover
    cv2 = None

from src.trackers.stabiliser import (  # noqa: E402
    LOST,
    REACQUIRED,
    SUSPECT,
    TRACKING,
    Decision,
    GuardConfig,
    Stabiliser,
    area_of,
    box_of,
    centre_of,
    estimate_shift,
    local_contrast,
    match_template,
)

SIZE = (240, 320)          # (height, width)


def ground(seed: int = 0) -> np.ndarray:
    """Textured terrain: what optical flow needs and a field of sky is not."""
    rng = np.random.default_rng(seed)
    field = rng.integers(40, 120, SIZE, dtype=np.uint8)
    if cv2 is not None:
        field = cv2.GaussianBlur(field, (5, 5), 0)
    return field


def with_target(frame: np.ndarray, centre, size=(14, 10), level: int = 205
                ) -> tuple[np.ndarray, np.ndarray]:
    """`frame` with a target drawn on it, and that target's box.

    The target carries an internal pattern rather than being a solid block, for
    the same reason a real one does: normalised cross-correlation divides by the
    patch's own standard deviation, so a *uniform* target correlates equally
    with everything and cannot be re-acquired at all. A saturated thermal target
    is that degenerate case, which `match_template` refuses outright.
    """
    out = frame.copy()
    cx, cy = int(centre[0]), int(centre[1])
    half_w, half_h = size[0] // 2, size[1] // 2
    x0, y0 = max(cx - half_w, 0), max(cy - half_h, 0)
    x1, y1 = min(cx + half_w, out.shape[1]), min(cy + half_h, out.shape[0])
    patch = np.full((y1 - y0, x1 - x0), level, dtype=np.int32)
    patch[::3, :] -= 45          # a windscreen, a hot bonnet: structure to match
    patch[:, ::4] -= 25
    out[y0:y1, x0:x1] = np.clip(patch, 0, 255).astype(np.uint8)
    return out, np.array([x0, y0, x1, y1], dtype=np.float64)


def faint_target(frame: np.ndarray, centre, size=(14, 10), lift: float = 3.0
                 ) -> tuple[np.ndarray, np.ndarray]:
    """A target drawn at its own background's level -- faint by construction.

    Picking a grey value and hoping it comes out faint is how a test starts
    passing for the wrong reason: the pattern inside the target moves its mean,
    so "level 88" was a *high* contrast case here and "level 100" a low one.
    Reading the local mean and sitting `lift` levels above it makes the
    signal-to-clutter ratio a property of the construction rather than a
    coincidence of the constants.
    """
    cx, cy = int(centre[0]), int(centre[1])
    half_w, half_h = size[0] // 2, size[1] // 2
    x0, y0 = max(cx - half_w, 0), max(cy - half_h, 0)
    x1, y1 = min(cx + half_w, frame.shape[1]), min(cy + half_h, frame.shape[0])
    local = float(frame[max(y0 - 12, 0):y1 + 12, max(x0 - 12, 0):x1 + 12].mean())
    out = frame.copy()
    patch = np.full((y1 - y0, x1 - x0), local + lift)
    patch[::3, :] -= 2.0
    out[y0:y1, x0:x1] = np.clip(patch, 0, 255).astype(np.uint8)
    return out, np.array([x0, y0, x1, y1], dtype=np.float64)


def slide(frame: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """The whole scene translated -- a drone's yaw, as the sensor sees it."""
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(frame, matrix, (frame.shape[1], frame.shape[0]),
                          borderMode=cv2.BORDER_REFLECT)


@unittest.skipIf(cv2 is None, "OpenCV is needed for the flow and the matcher")
class EgoMotionTest(unittest.TestCase):
    def test_the_background_shift_is_recovered(self):
        first = ground()
        for dx, dy in ((7, -4), (-3, 2), (0, 0), (11, 9)):
            with self.subTest(shift=(dx, dy)):
                measured = estimate_shift(first, slide(first, dx, dy))
                self.assertAlmostEqual(measured[0], dx, delta=0.75)
                self.assertAlmostEqual(measured[1], dy, delta=0.75)

    def test_the_target_does_not_get_to_vote_on_the_camera(self):
        """The one thing in the frame that is not background is the target, and
        a grid that samples it reports the target's motion as the camera's --
        which would let a real jump pass as camera shake."""
        first, box = with_target(ground(), (160, 120), size=(90, 70))
        # The scene is still; only the target moves, a long way.
        second, _ = with_target(ground(), (240, 120), size=(90, 70))
        excluded = estimate_shift(first, second, exclude=box)
        included = estimate_shift(first, second)
        self.assertLess(abs(excluded[0]), 1.0)
        self.assertLessEqual(abs(excluded[0]), abs(included[0]) + 1e-6)

    def test_a_frame_with_no_texture_is_not_guessed_at(self):
        flat = np.full(SIZE, 90, dtype=np.uint8)
        self.assertEqual(estimate_shift(flat, flat), (0.0, 0.0))

    def test_two_frames_of_different_sizes_are_refused(self):
        with self.assertRaises(ValueError):
            estimate_shift(ground(), np.zeros((10, 10), np.uint8))


@unittest.skipIf(cv2 is None, "OpenCV is needed for the flow and the matcher")
class PlausibilityTest(unittest.TestCase):
    """The balloon and the jump, which no promptable segmenter refuses itself."""

    def setUp(self):
        self.guard = Stabiliser()
        self.frame, self.box = with_target(ground(), (160, 120))
        self.guard.update(self.frame, self.box)
        # A few honest frames, so the medians mean something.
        for step in range(1, 5):
            frame, box = with_target(ground(), (160 + 2 * step, 120))
            self.assertEqual(self.guard.update(frame, box).state, TRACKING)

    def test_a_box_that_balloons_is_refused_and_the_prediction_is_held(self):
        frame, _ = with_target(ground(), (168, 120))
        balloon = np.array([40, 40, 300, 220], dtype=np.float64)
        decision = self.guard.update(frame, balloon)
        self.assertEqual(decision.state, SUSPECT)
        self.assertIn("frame", decision.reason + " covers")
        self.assertFalse(decision.accepted)
        # What comes back is where the target should be, not where the tracker
        # said it was.
        self.assertLess(float(np.linalg.norm(centre_of(decision.box)
                                             - np.array([170.0, 120.0]))), 12.0)

    def test_a_box_that_jumps_to_a_distractor_is_refused(self):
        frame, _ = with_target(ground(), (168, 120))
        jumped = np.array([20, 200, 34, 210], dtype=np.float64)
        decision = self.guard.update(frame, jumped)
        self.assertEqual(decision.state, SUSPECT)
        self.assertIn("jumped", decision.reason)

    def test_an_honest_change_is_not_refused(self):
        """A target that grows as the drone descends must pass. A gate that
        fires on this is worse than no gate: it locks the tracker onto its own
        prediction."""
        for step, size in enumerate(((15, 11), (16, 12), (17, 13)), start=5):
            frame, box = with_target(ground(), (160 + 2 * step, 120), size=size)
            with self.subTest(size=size):
                self.assertEqual(self.guard.update(frame, box).state, TRACKING)

    def test_three_refused_frames_in_a_row_are_a_lost_track(self):
        for index in range(3):
            frame, _ = with_target(ground(), (170 + 2 * index, 120))
            state = self.guard.update(frame, np.array([40, 40, 300, 220])).state
        self.assertIn(state, (LOST, REACQUIRED))

    def test_one_refused_frame_is_not(self):
        """Blur, a bird, one bad decode. Hysteresis is what tells those from a
        loss, and calling a loss on the first of them is how a tracker throws
        away a track it still has."""
        frame, _ = with_target(ground(), (170, 120))
        self.assertEqual(self.guard.update(frame, np.array([40, 40, 300, 220])).state,
                         SUSPECT)
        frame, box = with_target(ground(), (172, 120))
        self.assertEqual(self.guard.update(frame, box).state, REACQUIRED)
        frame, box = with_target(ground(), (174, 120))
        self.assertEqual(self.guard.update(frame, box).state, TRACKING)


@unittest.skipIf(cv2 is None, "OpenCV is needed for the flow and the matcher")
class MovingCameraTest(unittest.TestCase):
    """The case this was written for: the camera slides *and* the target moves."""

    def test_a_sliding_camera_does_not_look_like_a_jump(self):
        guard = Stabiliser()
        scene = ground()
        frame, box = with_target(scene, (160, 120))
        guard.update(frame, box)
        for step in range(1, 6):
            # The whole world moves 6 px a frame; the target sits still in it.
            moved = slide(scene, -6 * step, 0)
            frame, box = with_target(moved, (160 - 6 * step, 120))
            decision = guard.update(frame, box)
            with self.subTest(step=step):
                self.assertEqual(decision.state, TRACKING)
                self.assertAlmostEqual(decision.shift[0], -6.0, delta=1.5)

    def test_the_search_region_travels_with_the_camera(self):
        """A target lost while the camera pans is not where it was in image
        coordinates, and a search centred there finds terrain."""
        guard = Stabiliser(GuardConfig(suspect_frames=1, match_threshold=0.3))
        scene = ground()
        frame, box = with_target(scene, (160, 120))
        guard.update(frame, box)
        for step in range(1, 4):
            moved = slide(scene, -8 * step, 0)
            frame, box = with_target(moved, (160 - 8 * step, 120))
            guard.update(frame, box)
        # The tracker now claims nothing for one frame while the pan continues.
        moved = slide(scene, -8 * 4, 0)
        frame, _ = with_target(moved, (160 - 8 * 4, 120))
        decision = guard.update(frame, None)
        self.assertIn(decision.state, (LOST, REACQUIRED))
        self.assertLess(abs(centre_of(decision.box)[0] - (160 - 32)), 14.0)


@unittest.skipIf(cv2 is None, "OpenCV is needed for the flow and the matcher")
class ReacquisitionTest(unittest.TestCase):
    def test_a_lost_target_is_found_again_by_its_own_appearance(self):
        guard = Stabiliser(GuardConfig(suspect_frames=2, match_threshold=0.35))
        scene = ground(1)
        frame, box = with_target(scene, (160, 120))
        guard.update(frame, box)
        for step in range(1, 4):
            frame, box = with_target(scene, (160 + 3 * step, 120))
            guard.update(frame, box)

        # The tracker balloons for three frames while the target keeps moving.
        states = []
        for step in range(4, 7):
            frame, _ = with_target(scene, (160 + 3 * step, 120))
            states.append(guard.update(frame, np.array([10, 10, 310, 230])).state)
        # Refused, refused, then searched -- and once the guard is driving it
        # searches every frame rather than dead-reckoning between attempts.
        self.assertEqual(states, [SUSPECT, REACQUIRED, REACQUIRED])
        # And it landed on the target, not on terrain that correlated.
        self.assertLess(abs(centre_of(guard.box)[0] - (160 + 3 * 6)), 8.0)

    def test_matching_refuses_a_region_too_small_to_hold_the_template(self):
        frame = ground()
        template = frame[:40, :40]
        box, score = match_template(frame, template, [0, 0, 20, 20])
        self.assertIsNone(box)
        self.assertTrue(np.isnan(score))

    def test_a_track_lost_for_too_long_stops_searching(self):
        """The appearance kept from before the loss goes stale, and matching it
        forever is how a tracker locks onto a bush that happens to correlate."""
        guard = Stabiliser(GuardConfig(suspect_frames=1, give_up_after=3,
                                       match_threshold=0.99))
        scene = ground(2)
        frame, box = with_target(scene, (160, 120))
        guard.update(frame, box)
        for _ in range(6):
            decision = guard.update(ground(3), None)
        self.assertEqual(decision.state, LOST)
        self.assertTrue(np.isnan(decision.match))


@unittest.skipIf(cv2 is None, "OpenCV is needed for the flow and the matcher")
class ContrastTest(unittest.TestCase):
    def test_local_contrast_agrees_with_the_training_side_measure(self):
        """One definition of contrast across training, evaluation and the
        tracker -- `image_loop.instance_contrast` is the same ratio on a mask."""
        strong, box = with_target(ground(), (160, 120), level=235)
        weak, _ = with_target(ground(), (160, 120), level=95)
        self.assertGreater(local_contrast(strong, box), 3.0)
        self.assertLess(local_contrast(weak, box), 3.0)

    def test_a_linear_stretch_leaves_the_ratio_exactly_where_it_was(self):
        """Why no enhancement ships here. Any affine change of the pixels
        multiplies the target's signal and the ground's clutter by the same
        factor, and the model works on the ratio."""
        frame, box = faint_target(ground(), (160, 120))
        stretched = np.clip(frame.astype(np.float32) * 1.8 - 40,
                            0, 255).astype(np.uint8)
        self.assertAlmostEqual(local_contrast(stretched, box),
                               local_contrast(frame, box), delta=0.2)

    def test_local_equalisation_does_not_raise_it_either(self):
        """CLAHE is the usual suggestion and it is measurably not an answer: it
        amplifies the clutter immediately around the target at least as much as
        the target itself, so the ratio comes out no better -- here, worse."""
        frame, box = faint_target(ground(), (160, 120))
        equalised = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(frame)
        self.assertLessEqual(local_contrast(equalised, box),
                             local_contrast(frame, box) + 0.1)

    def test_a_faint_target_tightens_the_gates_instead(self):
        guard = Stabiliser()
        strong, box = with_target(ground(), (160, 120), level=235)
        weak, faint_box = faint_target(ground(), (160, 120))
        self.assertGreater(local_contrast(strong, box), guard.config.caution_below)
        self.assertLess(local_contrast(weak, faint_box), guard.config.caution_below)
        self.assertEqual(guard._gates(strong, box).max_jump, guard.config.max_jump)
        self.assertLess(guard._gates(weak, faint_box).max_jump,
                        guard.config.max_jump)


@unittest.skipIf(cv2 is None, "OpenCV is needed for the flow and the matcher")
class TrackerIntegrationTest(unittest.TestCase):
    """What the tracker does with a verdict: drop the mask, keep the reason.

    Built without a model on purpose -- `EdgeTAMTracker.__init__` loads nothing,
    so the judging path can be exercised on real frames with no GPU, no
    checkpoint and no EdgeTAM checkout.
    """

    def tracker(self, frames_dir, **guard):
        from src.trackers.edgetam_tracker import EdgeTAMTracker
        from src.trackers.stabiliser import FrameMotion

        built = EdgeTAMTracker(model_cfg="configs/edgetam.yaml",
                               checkpoint="none.pt",
                               guard={"suspect_frames": 3, **guard})
        built._motion = FrameMotion(frames_dir, reduce=1)
        return built

    def write(self, directory: Path, frames):
        for index, frame in enumerate(frames):
            cv2.imwrite(str(directory / f"{index:05d}.png"), frame)

    def mask_of(self, box, shape=SIZE) -> np.ndarray:
        mask = np.zeros(shape, dtype=bool)
        x0, y0, x1, y1 = (int(v) for v in box)
        mask[y0:y1, x0:x1] = True
        return mask

    def test_an_honest_mask_passes_through_untouched(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            frames, boxes = [], []
            for step in range(4):
                frame, box = with_target(ground(), (160 + 3 * step, 120))
                frames.append(frame)
                boxes.append(box)
            self.write(directory, frames)
            tracker = self.tracker(directory)
            for index, box in enumerate(boxes):
                mask = self.mask_of(box)
                out = tracker._judge(index, {1: mask})
                self.assertTrue((out[1] == mask).all())
                self.assertEqual(tracker.verdicts[index][1].state, TRACKING)

    def test_a_ballooned_mask_comes_back_empty_with_its_reason(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            frames, boxes = [], []
            for step in range(5):
                frame, box = with_target(ground(), (160 + 3 * step, 120))
                frames.append(frame)
                boxes.append(box)
            self.write(directory, frames)
            tracker = self.tracker(directory)
            for index in range(4):
                tracker._judge(index, {1: self.mask_of(boxes[index])})

            balloon = self.mask_of([20, 20, 300, 220])
            out = tracker._judge(4, {1: balloon})
            self.assertFalse(out[1].any())
            self.assertEqual(out[1].shape, balloon.shape)
            verdict = tracker.verdicts[4][1]
            self.assertFalse(verdict.accepted)
            self.assertTrue(verdict.reason)

    def test_each_object_is_judged_on_its_own_track(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            frames = [with_target(ground(), (160 + 3 * step, 120))[0]
                      for step in range(3)]
            self.write(directory, frames)
            tracker = self.tracker(directory)
            for index in range(3):
                out = tracker._judge(index, {
                    1: self.mask_of([150 + 3 * index, 115, 170 + 3 * index, 125]),
                    2: self.mask_of([40, 40, 60, 52]),
                })
                self.assertEqual(sorted(out), [1, 2])
            self.assertEqual(sorted(tracker._guards), [1, 2])
            self.assertIsNot(tracker._guards[1], tracker._guards[2])

    def test_a_reset_forgets_every_track_and_every_verdict(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            self.write(directory, [with_target(ground(), (160, 120))[0]])
            tracker = self.tracker(directory)
            tracker._judge(0, {1: self.mask_of([150, 115, 170, 125])})
            tracker.reset()
            self.assertEqual(tracker._guards, {})
            self.assertEqual(tracker.verdicts, {})

    def test_no_guard_block_means_no_judging_at_all(self):
        from src.trackers.edgetam_tracker import EdgeTAMTracker

        for block in (None, {}, {"enabled": False}):
            with self.subTest(guard=block):
                built = EdgeTAMTracker(model_cfg="c", checkpoint="k", guard=block)
                self.assertIsNone(built.guard_config)


class GeometryTest(unittest.TestCase):
    """No OpenCV: the parts a caller can rely on with nothing installed."""

    def test_the_box_of_an_empty_mask_is_nothing_rather_than_a_point(self):
        self.assertIsNone(box_of(np.zeros((8, 8), bool)))

    def test_the_box_of_a_mask_is_tight_around_it(self):
        mask = np.zeros((10, 10), bool)
        mask[3:6, 2:7] = True
        np.testing.assert_array_equal(box_of(mask), [2, 3, 7, 6])
        self.assertEqual(area_of(box_of(mask)), 15)

    def test_a_decision_says_whether_the_box_was_the_tracker_s_own(self):
        self.assertTrue(Decision(None, TRACKING).accepted)
        self.assertTrue(Decision(None, REACQUIRED).accepted)
        self.assertFalse(Decision(None, SUSPECT).accepted)
        self.assertFalse(Decision(None, LOST).accepted)

    def test_a_reset_guard_holds_nothing_of_the_last_video(self):
        guard = Stabiliser()
        guard.box = np.array([1.0, 2.0, 3.0, 4.0])
        guard.missing = 5
        guard.reset()
        self.assertIsNone(guard.box)
        self.assertEqual((guard.missing, guard.state), (0, TRACKING))


class ShippedConfigTest(unittest.TestCase):
    """The guard has to be reachable from a config file, or it does not exist.

    `tools/track_adaptive.py` and `tools/eval_antiuav.py` both read a YAML and
    splat it into `build_tracker(name, **backend)`, so every key in a shipped
    config is a keyword argument to a tracker constructor. That makes an
    unknown key a `TypeError` at start-up rather than something ignored -- and
    it made the two failures these tests exist to stop: `stabiliser.py` was 700
    lines that no shipped config turned on, and the TensorRT tracker accepted
    `ego_motion` while dropping `guard` from its signature, so a config with
    both worked on PyTorch and crashed on the engine build.
    """

    CONFIGS = ("edgetam_768_samurai.yaml", "edgetam_512_thermal_guard.yaml")

    def configs(self):
        import yaml

        for name in self.CONFIGS:
            path = ROOT / "configs" / name
            self.assertTrue(path.is_file(), f"{name} is not in configs/")
            yield name, yaml.safe_load(path.read_text())

    def test_every_guard_block_is_a_guard_config(self):
        import inspect

        known = set(inspect.signature(GuardConfig).parameters)
        for name, body in self.configs():
            with self.subTest(config=name):
                guard = body.get("guard")
                self.assertIsInstance(guard, dict, f"{name} ships no guard block")
                fields = {k: v for k, v in guard.items() if k != "enabled"}
                self.assertEqual(set(fields) - known, set())
                GuardConfig(**fields)

    def test_the_thermal_config_runs_the_whole_classical_stack(self):
        """`guard` measures the jump after the background's motion is removed,
        so a guard without `ego_motion` is measuring a drone's yaw as the
        target moving. The three belong together and the config that exists to
        be pointed at thermal footage has to carry all three."""
        body = dict(self.configs()).get("edgetam_512_thermal_guard.yaml")
        for block in ("samurai", "ego_motion", "guard"):
            self.assertIn(block, body)
        self.assertEqual(body["image_size"], 512)

    def test_a_standalone_config_says_what_the_policy_overlay_says(self):
        """The same thresholds now live in two places and must not drift.

        `configs/policies/*.yaml` are overlays `tools/run_records.py` merges
        onto whichever backend YAML a (weights, size) pair chose -- the right
        shape for an ablation across engines. The two configs here are whole
        files, for pointing `cli.py --config` at directly. Both are wanted, and
        a `max_jump` changed in one of them and not the other would make two
        runs that read as the same configuration behave differently.
        """
        import yaml

        overlay = yaml.safe_load(
            (ROOT / "configs" / "policies" / "guard.yaml").read_text())
        for name, body in self.configs():
            for block in ("samurai", "ego_motion", "guard"):
                with self.subTest(config=name, block=block):
                    self.assertEqual(body[block], overlay[block])

    def test_both_trackers_accept_every_key_a_shipped_config_holds(self):
        import inspect

        from src.trackers.edgetam_tracker import EdgeTAMTracker
        from src.trackers.edgetam_trt_tracker import EdgeTAMTRTTracker

        for name, body in self.configs():
            for tracker in (EdgeTAMTracker, EdgeTAMTRTTracker):
                with self.subTest(config=name, tracker=tracker.__name__):
                    taken = set(inspect.signature(tracker).parameters)
                    self.assertEqual(set(body) - taken, set())



if __name__ == "__main__":
    unittest.main()
