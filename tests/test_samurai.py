"""Motion-aware mask selection and the memory gate.

Numpy and torch only. The two claims worth pinning are:

* **a permissive configuration is exactly stock SAM 2** -- so this can be
  switched off, and so a regression shows up as a behaviour change rather than
  as a slightly different number;
* **a poisoned frame never enters the memory bank** -- the failure this exists
  to fix, expressed as a test rather than as an anecdote.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.trackers.samurai import (  # noqa: E402
    FrameRecord,
    KalmanBoxTracker,
    MotionAwareMemory,
    Samurai,
    SamuraiConfig,
    _filtered_memory,
    boxes_from_masks,
    mask_boxes,
    xyah_to_xyxy,
    xyxy_to_xyah,
)

SIZE = 64
PERMISSIVE = SamuraiConfig(kf_weight=0.0, stable_frames=0, memory_iou=-1.0,
                           memory_obj_score=float("-inf"), memory_kf_score=-1.0)


def mask_at(box, size=SIZE, logit=8.0):
    """Mask logits that are positive exactly inside `box`."""
    out = np.full((size, size), -logit, dtype=np.float32)
    x0, y0, x1, y1 = (int(v) for v in box)
    out[y0:y1, x0:x1] = logit
    return out


def candidates(boxes):
    """Mask logits for each box, as the SAM head would emit them."""
    return np.stack([mask_at(b) for b in boxes])


def boxes_of(boxes):
    """The same candidates, reduced to boxes the way `rescore` does on device."""
    return mask_boxes(torch.from_numpy(candidates(boxes))[None])[0].numpy()


class TestBoxConversions(unittest.TestCase):
    def test_roundtrip(self):
        box = np.array([10.0, 20.0, 30.0, 60.0])
        np.testing.assert_allclose(xyah_to_xyxy(xyxy_to_xyah(box)), box, atol=1e-6)

    def test_aspect_is_width_over_height(self):
        state = xyxy_to_xyah(np.array([0.0, 0.0, 20.0, 40.0]))
        self.assertAlmostEqual(state[2], 0.5)
        self.assertAlmostEqual(state[3], 40.0)


class TestKalmanBoxTracker(unittest.TestCase):
    def test_it_learns_a_constant_velocity(self):
        kf = KalmanBoxTracker()
        kf.initiate(np.array([0.0, 0.0, 20.0, 20.0]))
        for step in range(1, 15):
            kf.predict()
            kf.update(np.array([step * 5.0, 0.0, step * 5.0 + 20, 20.0]))

        predicted = kf.predict()
        # After 14 frames at 5 px/frame the next centre should be near 15*5+10.
        self.assertAlmostEqual(float((predicted[0] + predicted[2]) / 2), 85.0, delta=3.0)

    def test_prediction_before_initiation_is_none(self):
        self.assertIsNone(KalmanBoxTracker().predict())
        self.assertFalse(KalmanBoxTracker().ready)

    def test_update_on_a_fresh_filter_initiates_it(self):
        kf = KalmanBoxTracker()
        kf.update(np.array([1.0, 2.0, 3.0, 4.0]))
        self.assertTrue(kf.ready)

    def test_reset_clears_the_track(self):
        kf = KalmanBoxTracker()
        kf.initiate(np.array([0.0, 0.0, 10.0, 10.0]))
        kf.reset()
        self.assertFalse(kf.ready)

    def test_it_stays_finite_on_a_degenerate_box(self):
        kf = KalmanBoxTracker()
        kf.initiate(np.array([5.0, 5.0, 5.0, 5.0]))  # zero area
        for _ in range(5):
            kf.predict()
            kf.update(np.array([5.0, 5.0, 5.0, 5.0]))
        self.assertTrue(np.isfinite(kf.mean).all())


class TestBoxesFromMasks(unittest.TestCase):
    def test_positive_logits_become_a_box(self):
        boxes = boxes_from_masks(candidates([(10, 10, 20, 30)]))
        np.testing.assert_allclose(boxes[0], [10, 10, 20, 30])

    def test_an_all_negative_mask_is_nan(self):
        empty = np.full((1, SIZE, SIZE), -5.0, dtype=np.float32)
        self.assertTrue(np.isnan(boxes_from_masks(empty)).all())

    def test_the_on_device_extractor_agrees_with_the_numpy_reference(self):
        # `rescore` uses the torch path so it moves 3 boxes across the bus
        # instead of 3 masks. It has to be the same answer.
        masks = candidates([(10, 10, 20, 30), (0, 0, 1, 1), (40, 5, 63, 60)])
        np.testing.assert_allclose(
            mask_boxes(torch.from_numpy(masks)[None])[0].numpy(),
            boxes_from_masks(masks),
        )

    def test_the_on_device_extractor_marks_an_empty_candidate_nan(self):
        masks = np.stack([mask_at((10, 10, 20, 20)),
                          np.full((SIZE, SIZE), -5.0, dtype=np.float32)])
        boxes = mask_boxes(torch.from_numpy(masks)[None])[0].numpy()

        np.testing.assert_allclose(boxes[0], [10, 10, 20, 20])
        self.assertTrue(np.isnan(boxes[1]).all())


class TestSelection(unittest.TestCase):
    def _warm(self, memory, box=(10, 10, 20, 20), frames=20, step=2):
        """Track a steadily moving target until the filter is trusted."""
        x0, y0, x1, y1 = box
        for i in range(frames):
            memory.select(boxes_of([(x0 + i * step, y0, x1 + i * step, y1)]),
                          np.array([0.9]))
        return memory

    def test_appearance_alone_decides_during_warm_up(self):
        memory = MotionAwareMemory(SamuraiConfig(stable_frames=15))
        selection = memory.select(boxes_of([(0, 0, 10, 10), (40, 40, 50, 50)]),
                                  np.array([0.4, 0.8]))
        self.assertEqual(selection.index, 1)

    def test_motion_breaks_a_tie_once_the_filter_is_trusted(self):
        memory = self._warm(MotionAwareMemory(SamuraiConfig(stable_frames=8)))
        # Two candidates the IoU head cannot separate: one continues the track,
        # one has jumped across the frame.
        on_track = (10 + 20 * 2, 10, 20 + 20 * 2, 20)
        jumped = (0, 45, 10, 55)
        selection = memory.select(boxes_of([jumped, on_track]), np.array([0.7, 0.7]))
        self.assertEqual(selection.index, 1)
        self.assertGreater(selection.kf_score, 0.0)

    def test_a_small_kf_weight_does_not_overrule_a_clear_appearance_win(self):
        # Appearance is usually right; the motion term is there to break ties
        # and to veto, not to take over.
        memory = self._warm(MotionAwareMemory(SamuraiConfig(stable_frames=8,
                                                            kf_weight=0.15)))
        on_track = (10 + 20 * 2, 10, 20 + 20 * 2, 20)
        elsewhere = (0, 45, 10, 55)
        selection = memory.select(boxes_of([elsewhere, on_track]),
                                  np.array([0.99, 0.30]))
        self.assertEqual(selection.index, 0)

    def test_kf_weight_zero_reproduces_the_stock_argmax(self):
        memory = self._warm(MotionAwareMemory(SamuraiConfig(kf_weight=0.0,
                                                            stable_frames=0)))
        ious = np.array([0.2, 0.9, 0.5])
        selection = memory.select(boxes_of([(0, 0, 5, 5), (30, 30, 40, 40),
                                              (50, 50, 60, 60)]), ious)
        self.assertEqual(selection.index, int(np.argmax(ious)))
        np.testing.assert_allclose(selection.weighted, ious)

    def test_warm_up_restarts_when_the_tracker_and_the_filter_disagree(self):
        # A filter fed its own disagreements would converge on the wrong track
        # and then be trusted, which is worse than not using one.
        memory = MotionAwareMemory(SamuraiConfig(stable_frames=15, stable_iou=0.3))
        memory.select(boxes_of([(10, 10, 20, 20)]), np.array([0.9]))
        memory.select(boxes_of([(12, 10, 22, 20)]), np.array([0.9]))
        self.assertEqual(memory.stable, 2)
        memory.select(boxes_of([(14, 10, 24, 20)]), np.array([0.1]))
        self.assertEqual(memory.stable, 0)

    def test_an_empty_candidate_does_not_poison_the_filter(self):
        memory = self._warm(MotionAwareMemory(SamuraiConfig(stable_frames=5)))
        before = memory.kalman.mean.copy()
        memory.select(np.full((1, 4), np.nan), np.array([0.1]))
        self.assertTrue(np.isfinite(memory.kalman.mean).all())
        self.assertFalse(np.allclose(before, 0))


class TestMemoryGate(unittest.TestCase):
    def test_a_good_frame_is_kept(self):
        memory = MotionAwareMemory()
        memory.record(5, iou=0.8, object_score=2.0, kf_score=0.7)
        self.assertTrue(memory.keep(5))

    def test_a_frame_where_the_object_score_went_negative_is_refused(self):
        # This is the failure: a negative object score writes `no_obj_ptr` into
        # the bank, and the next seven frames read it back.
        memory = MotionAwareMemory()
        memory.record(5, iou=0.8, object_score=-0.4, kf_score=0.7)
        self.assertFalse(memory.keep(5))

    def test_each_threshold_can_reject_on_its_own(self):
        cases = {
            "iou": dict(iou=0.1, object_score=2.0, kf_score=0.7),
            "object_score": dict(iou=0.8, object_score=-1.0, kf_score=0.7),
            "kf_score": dict(iou=0.8, object_score=2.0, kf_score=-0.1),
        }
        for name, kwargs in cases.items():
            memory = MotionAwareMemory()
            memory.record(1, **kwargs)
            self.assertFalse(memory.keep(1), f"{name} did not reject")

    def test_an_unrecorded_frame_is_kept(self):
        # A prompted frame, or one tracked before this was installed, has not
        # failed anything. Keeping it is both the safe and the honest default.
        self.assertTrue(MotionAwareMemory().keep(99))

    def test_the_newest_acceptable_frames_are_returned_oldest_first(self):
        samurai = Samurai()
        for frame in range(10):
            samurai.state(0).record(frame, iou=0.8, object_score=1.0, kf_score=0.7)
        samurai.state(0).record(8, iou=0.8, object_score=-1.0, kf_score=0.7)  # poisoned
        self.assertEqual(samurai.acceptable(list(range(10)), 3), [6, 7, 9])

    def test_a_permissive_config_rejects_nothing(self):
        memory = MotionAwareMemory(PERMISSIVE)
        memory.record(1, iou=0.0, object_score=-100.0, kf_score=0.0)
        self.assertTrue(memory.keep(1))
        self.assertTrue(PERMISSIVE.permissive)


class TestFilteredMemory(unittest.TestCase):
    def _dict(self, frames):
        return {"cond_frame_outputs": {0: "prompt"},
                "non_cond_frame_outputs": {f: f"out{f}" for f in frames}}

    def _samurai(self, rejected=()):
        samurai = Samurai()
        for frame in range(20):
            bad = frame in rejected
            samurai.state(0).record(frame, iou=0.1 if bad else 0.8,
                                    object_score=1.0, kf_score=0.7)
        return samurai

    def test_nothing_rejected_leaves_the_dict_untouched(self):
        output_dict = self._dict(range(10))
        self.assertIs(_filtered_memory(output_dict, self._samurai(), 10, 7, 1),
                      output_dict)

    def test_a_rejected_frame_is_replaced_by_an_older_good_one(self):
        # Backfilling is the point: after a bad patch the bank stays full of
        # frames that were actually good, instead of running short.
        filtered = _filtered_memory(self._dict(range(12)), self._samurai({9, 10, 11}),
                                    12, 7, 1)
        kept = filtered["non_cond_frame_outputs"]

        self.assertEqual(len(kept), 6)  # num_maskmem - 1
        self.assertNotIn("out9", kept.values())
        self.assertIn("out8", kept.values())
        self.assertIn("out3", kept.values())

    def test_backfilled_frames_land_on_the_indices_upstream_looks_up(self):
        # Renumbering is what lets SAM 2's own stride arithmetic run untouched;
        # none of it is duplicated here.
        filtered = _filtered_memory(self._dict(range(12)), self._samurai({11}), 12, 7, 1)
        self.assertEqual(sorted(filtered["non_cond_frame_outputs"]), [6, 7, 8, 9, 10, 11])

    def test_conditioning_frames_are_never_filtered(self):
        # They carry the user's prompt; they are ground truth, not a prediction.
        filtered = _filtered_memory(self._dict(range(12)), self._samurai(set(range(12))),
                                    12, 7, 1)
        self.assertEqual(filtered["cond_frame_outputs"], {0: "prompt"})

    def test_everything_rejected_leaves_an_empty_bank_rather_than_bad_memory(self):
        filtered = _filtered_memory(self._dict(range(12)), self._samurai(set(range(12))),
                                    12, 7, 1)
        self.assertEqual(filtered["non_cond_frame_outputs"], {})

    def test_a_non_unit_stride_filters_without_renumbering(self):
        # Renumbering assumes upstream asks for consecutive indices. Under a
        # different stride it does not, so bad frames are dropped but not
        # backfilled: the bank runs shorter rather than wrong.
        filtered = _filtered_memory(self._dict(range(12)), self._samurai({10}), 12, 7, 2)
        kept = filtered["non_cond_frame_outputs"]
        self.assertNotIn(10, kept)
        self.assertIn(9, kept)

    def test_future_frames_are_never_candidates(self):
        filtered = _filtered_memory(self._dict(range(20)), self._samurai({5}), 8, 7, 1)
        self.assertTrue(all(f < 8 for f in filtered["non_cond_frame_outputs"]))


class TestSamurai(unittest.TestCase):
    def test_rescore_rewrites_the_scores_so_upstreams_argmax_picks_its_choice(self):
        # The whole integration rests on this: SAM 2 derives the mask, the
        # object pointer and the memory write from one argmax over these
        # values, so moving them moves all three together.
        samurai = Samurai(SamuraiConfig(stable_frames=0, kf_weight=1.0))
        masks = torch.from_numpy(candidates([(10, 10, 20, 20)])[None])
        samurai.rescore(masks, torch.tensor([[0.9]]), torch.tensor([[2.0]]))

        moved = torch.from_numpy(candidates([(0, 40, 10, 50), (12, 10, 22, 20)])[None])
        scores = samurai.rescore(moved, torch.tensor([[0.8, 0.2]]),
                                 torch.tensor([[2.0]]))
        self.assertEqual(int(scores.argmax(dim=-1)), 1)

    def test_a_single_candidate_frame_still_updates_the_filter(self):
        # The prompted frame can be single-mask. Returning early there would
        # leave the filter blind to it, and the scores are unchanged either way
        # because there is nothing to choose between.
        samurai = Samurai(SamuraiConfig(stable_frames=0))
        ious = torch.tensor([[0.7]])
        masks = torch.from_numpy(candidates([(10, 10, 20, 20)])[None])

        self.assertIs(samurai.rescore(masks, ious, torch.tensor([[1.0]])), ious)
        self.assertTrue(samurai.states[0].kalman.ready)

    def test_each_tracked_object_gets_its_own_filter(self):
        samurai = Samurai()
        masks = torch.from_numpy(np.stack([candidates([(10, 10, 20, 20)])[0],
                                           candidates([(40, 40, 50, 50)])[0]])[:, None])
        samurai.rescore(masks.repeat(1, 2, 1, 1), torch.tensor([[0.9, 0.1], [0.9, 0.1]]),
                        torch.tensor([[2.0], [2.0]]))
        self.assertEqual(len(samurai.states), 2)
        self.assertIsNot(samurai.states[0].kalman, samurai.states[1].kalman)

    def test_a_frame_bad_for_any_object_is_not_remembered(self):
        # The bank stores one entry per frame for every row at once, so the
        # gate has to be shared. Conservative is the right direction.
        samurai = Samurai()
        samurai.state(0).record(4, iou=0.9, object_score=2.0, kf_score=0.8)
        samurai.state(1).record(4, iou=0.9, object_score=-1.0, kf_score=0.8)
        self.assertFalse(samurai.keep(4))

    def test_reset_clears_every_object(self):
        samurai = Samurai()
        samurai.state(0).record(1, iou=0.9, object_score=1.0, kf_score=0.9)
        samurai.frame_idx = 12
        samurai.reset()
        self.assertEqual(samurai.states, [])
        self.assertEqual(samurai.frame_idx, 0)

    def test_object_score_is_taken_from_the_model_not_the_mask(self):
        samurai = Samurai(SamuraiConfig(stable_frames=0))
        masks = torch.from_numpy(candidates([(10, 10, 20, 20)])[None]).repeat(1, 2, 1, 1)
        samurai.frame_idx = 3
        samurai.rescore(masks, torch.tensor([[0.9, 0.1]]), torch.tensor([[-2.5]]))
        self.assertAlmostEqual(samurai.states[0].records[3].object_score, -2.5, places=4)
        self.assertFalse(samurai.keep(3))


class TestConfig(unittest.TestCase):
    def test_defaults_match_the_published_values(self):
        cfg = SamuraiConfig()
        self.assertAlmostEqual(cfg.kf_weight, 0.15)
        self.assertEqual(cfg.stable_frames, 15)
        self.assertAlmostEqual(cfg.stable_iou, 0.3)
        self.assertAlmostEqual(cfg.memory_iou, 0.5)

    def test_an_unknown_option_is_rejected_at_startup(self):
        from src.trackers.edgetam_tracker import _samurai_config

        with self.assertRaises(ValueError) as ctx:
            _samurai_config({"kf_wieght": 0.2})
        self.assertIn("kf_wieght", str(ctx.exception))

    def test_disabled_and_absent_both_mean_stock_behaviour(self):
        from src.trackers.edgetam_tracker import _samurai_config

        self.assertIsNone(_samurai_config(None))
        self.assertIsNone(_samurai_config({}))
        self.assertIsNone(_samurai_config({"enabled": False}))

    def test_a_valid_block_is_accepted(self):
        from src.trackers.edgetam_tracker import _samurai_config

        cfg = _samurai_config({"kf_weight": 0.25, "memory_iou": 0.4})
        self.assertAlmostEqual(cfg.kf_weight, 0.25)
        self.assertAlmostEqual(cfg.memory_iou, 0.4)


class TestEgoMotion(unittest.TestCase):
    """The term published SAMURAI has no place for, and a drone needs most.

    Its filter lives in image coordinates. On a moving camera a target that
    never moved appears to accelerate, a constant-velocity model learns that
    acceleration, and the frame the camera changes what it is doing the filter
    predicts a jump -- which is then used to *veto* candidates, so the error
    does not stay in the filter.
    """

    @staticmethod
    def _run(shifts, compensate: bool):
        """A target that is still on the ground while the camera pans."""
        kf = KalmanBoxTracker()
        box = np.array([100.0, 100.0, 120.0, 110.0])
        kf.initiate(box)
        travelled = 0.0
        for step in shifts:
            travelled += step
            kf.predict((step, 0.0) if compensate else None)
            kf.update(box + np.array([travelled, 0.0, travelled, 0.0]))
        return kf, box, travelled

    def test_a_camera_that_starts_moving_is_predicted_immediately(self):
        still = [0.0] * 5
        panning = [-6.0] * 4
        errors = {}
        for compensate in (True, False):
            kf, box, travelled = self._run(still + panning, compensate)
            predicted = kf.predict((-6.0, 0.0) if compensate else None)
            errors[compensate] = abs(float(predicted[0]) - (box[0] + travelled - 6.0))
        # The compensated filter is handed the camera's motion, so it is right
        # on the first frame of the pan. The uncompensated one has to learn it
        # through its velocity state, and is still several pixels behind after
        # four -- which on a 20-pixel target is a quarter of its own width, and
        # is used to score candidates.
        self.assertLess(errors[True], 1.0)
        self.assertGreater(errors[False], 3 * errors[True])

    def test_the_compensated_filter_is_the_closer_one_when_the_pan_stops(self):
        """The harder half: the camera stops, and an uncompensated filter keeps
        predicting the motion it learned."""
        panning = [-6.0] * 8
        errors = {}
        for compensate in (True, False):
            kf, box, travelled = self._run(panning, compensate)
            predicted = kf.predict((0.0, 0.0) if compensate else None)
            errors[compensate] = abs(float(predicted[0]) - (box[0] + travelled))
        self.assertLess(errors[True], errors[False])
        self.assertLess(errors[True], 1.5)

    def test_no_shift_is_the_behaviour_every_earlier_run_had(self):
        kf = KalmanBoxTracker()
        kf.initiate(np.array([0.0, 0.0, 20.0, 20.0]))
        kf.predict()
        before = kf.predicted_box().copy()
        kf.reset()
        kf.initiate(np.array([0.0, 0.0, 20.0, 20.0]))
        kf.predict(None)
        np.testing.assert_allclose(kf.predicted_box(), before)

    def test_a_measured_shift_widens_the_position_covariance(self):
        """It is a measurement off pixels, not an odometer, so the filter is
        told to trust it a little less than it trusts its own model."""
        kf = KalmanBoxTracker()
        kf.initiate(np.array([100.0, 100.0, 120.0, 110.0]))
        kf.predict()
        tight = kf.covariance[0, 0]
        kf.reset()
        kf.initiate(np.array([100.0, 100.0, 120.0, 110.0]))
        kf.predict((-20.0, 0.0))
        self.assertGreater(kf.covariance[0, 0], tight)


class TestShiftPlumbing(unittest.TestCase):
    """The shift arrives normalised and is used in the mask grid's pixels.

    Those are two different resolutions -- a decoded 1280x720 frame and a
    192x192 mask grid at image_size 768 -- and a fraction is the one form that
    needs no agreement between the caller and this module about either.
    """

    def test_a_fraction_becomes_the_mask_grids_own_pixels(self):
        import torch

        samurai = Samurai(SamuraiConfig())
        samurai.shift = (-0.02, 0.01)
        logits = torch.zeros(1, 3, 192, 192)
        self.assertEqual(samurai._shift_for(logits), (-3.84, 1.92))

    def test_the_same_fraction_scales_with_the_resolution(self):
        import torch

        samurai = Samurai(SamuraiConfig())
        samurai.shift = (-0.02, 0.0)
        at_512 = samurai._shift_for(torch.zeros(1, 3, 128, 128))[0]
        at_768 = samurai._shift_for(torch.zeros(1, 3, 192, 192))[0]
        self.assertAlmostEqual(at_768 / at_512, 192 / 128, places=6)

    def test_no_measurement_is_no_control_input(self):
        import torch

        samurai = Samurai(SamuraiConfig())
        self.assertIsNone(samurai._shift_for(torch.zeros(1, 3, 128, 128)))

    def test_a_reset_forgets_the_last_video_s_camera(self):
        samurai = Samurai(SamuraiConfig())
        samurai.shift = (0.5, 0.5)
        samurai.reset()
        self.assertIsNone(samurai.shift)


class TestInstalledShiftSource(unittest.TestCase):
    """`install` asks for the shift where the frame number is known.

    A fake predictor, because the point under test is *when* the control input
    reaches the filter, not what a network does with it: the memory hook is the
    one place that runs before the SAM head on every frame, so a shift set
    there belongs to the frame about to be scored rather than the one after it.
    """

    class FakePredictor:
        num_maskmem = 7
        memory_temporal_stride_for_eval = 1

        def __init__(self):
            class Decoder:
                def forward(self, *args, **kwargs):
                    return None, None, None, None
            self.sam_mask_decoder = Decoder()
            self.seen = []

        def _prepare_memory_conditioned_features(self, frame_idx, *args, **kwargs):
            self.seen.append(frame_idx)
            return "features"

    def test_the_frame_being_scored_is_the_frame_asked_about(self):
        from src.trackers.samurai import install

        predictor = self.FakePredictor()
        asked = []

        def shift_for(index):
            asked.append(index)
            return (0.01 * index, 0.0)

        samurai = install(predictor, SamuraiConfig(), shift_for=shift_for)
        predictor._prepare_memory_conditioned_features(
            7, False, None, None, None, {"non_cond_frame_outputs": {}}, 10)
        self.assertEqual(asked, [7])
        self.assertEqual(samurai.frame_idx, 7)
        self.assertEqual(samurai.shift, (0.07, 0.0))

    def test_no_source_leaves_the_filter_uncompensated(self):
        from src.trackers.samurai import install

        predictor = self.FakePredictor()
        samurai = install(predictor, SamuraiConfig())
        predictor._prepare_memory_conditioned_features(
            3, False, None, None, None, {"non_cond_frame_outputs": {}}, 10)
        self.assertIsNone(samurai.shift)

    def test_reverse_tracking_takes_no_control_input(self):
        """The shift is measured forwards. Handing it to a filter running
        backwards would push the prediction the wrong way."""
        from src.trackers.samurai import install

        predictor = self.FakePredictor()
        samurai = install(predictor, SamuraiConfig(),
                          shift_for=lambda index: (0.05, 0.0))
        predictor._prepare_memory_conditioned_features(
            4, False, None, None, None, {"non_cond_frame_outputs": {}}, 10,
            track_in_reverse=True)
        self.assertIsNone(samurai.shift)


class TestFrameRecord(unittest.TestCase):
    def test_acceptability_is_a_conjunction(self):
        cfg = SamuraiConfig()
        self.assertTrue(FrameRecord(0.9, 1.0, 0.9).acceptable(cfg))
        self.assertFalse(FrameRecord(0.9, 1.0, -0.1).acceptable(cfg))



class FilterGatingTest(unittest.TestCase):
    """The hole the three memory thresholds cannot close.

    They decide which frames enter the bank and they work -- once. The filter
    is then updated with whatever box was chosen, so a mask that jumped to a
    look-alike drags the filter onto it, and two frames later the filter agrees
    with the distractor and the gate has nothing to say.
    """

    SIZE = 26

    def walk(self, memory, kind, frames=34, jump_at=18):
        """Boxes moving smoothly, then either teleporting or accelerating."""
        position = np.array([300.0, 240.0])
        velocity = np.array([3.0, 1.0])
        kept = []
        for index in range(frames):
            if kind == "manoeuvre" and index >= jump_at:
                velocity = np.array([16.0, 6.0])
            position = position + velocity
            if kind == "jump" and index == jump_at:
                position = position + np.array([90.0, 0.0])
            box = np.array([position[0] - 13, position[1] - 9,
                            position[0] + 13, position[1] + 9])
            chosen = memory.select(np.stack([box, box + 2, box - 2]),
                                   np.array([0.9, 0.8, 0.8]))
            memory.record(index, chosen.mask_score, 5.0, chosen.kf_score)
            if index >= jump_at:
                kept.append(memory.keep(index))
        return sum(kept), len(kept)

    def test_without_the_gate_the_filter_follows_the_distractor(self):
        """The failure, stated as a number: the memory gate catches the jump
        for exactly one frame and then the filter has been captured."""
        memory = MotionAwareMemory(SamuraiConfig())
        kept, total = self.walk(memory, "jump")
        self.assertEqual(total, 16)
        self.assertGreaterEqual(kept, 14)

    def test_the_gate_keeps_most_of_a_jump_out_of_the_bank(self):
        memory = MotionAwareMemory(SamuraiConfig(kf_gate=0.10))
        kept, _ = self.walk(memory, "jump")
        self.assertLessEqual(kept, 9)

    def test_a_real_manoeuvre_pays_nothing_for_it(self):
        """The half that makes the gate safe. A filter that never updates would
        lock out a target that really did accelerate, so `kf_gate_patience`
        re-initiates it -- and at 0.10 the honest case is untouched."""
        plain = MotionAwareMemory(SamuraiConfig())
        gated = MotionAwareMemory(SamuraiConfig(kf_gate=0.10))
        self.assertEqual(self.walk(plain, "manoeuvre"),
                         self.walk(gated, "manoeuvre"))

    def test_a_filter_that_disagrees_for_long_enough_is_the_one_re_initiated(self):
        memory = MotionAwareMemory(SamuraiConfig(kf_gate=0.9, kf_gate_patience=3))
        self.walk(memory, "jump", frames=24)
        # It cannot still be coasting: patience is 3 and the walk is long.
        self.assertLess(memory.coasted, 3)

    def test_off_by_default_and_off_means_untouched(self):
        """This changes what the filter learns, so it is measured rather than
        assumed, so `plain` is what it is read against."""
        self.assertEqual(SamuraiConfig().kf_gate, 0.0)
        plain = MotionAwareMemory(SamuraiConfig())
        default = MotionAwareMemory(SamuraiConfig(kf_gate=0.0))
        self.assertEqual(self.walk(plain, "jump"), self.walk(default, "jump"))

    def swell(self, memory, factor, at=20, frames=30):
        """A mask that stays on the target and grows in place."""
        position = np.array([300.0, 240.0])
        kept = []
        for index in range(frames):
            position = position + np.array([3.0, 1.0])
            half_w, half_h = (13.0, 9.0) if index < at else (13.0 * factor,
                                                             9.0 * factor)
            box = np.array([position[0] - half_w, position[1] - half_h,
                            position[0] + half_w, position[1] + half_h])
            chosen = memory.select(np.stack([box, box + 2, box - 2]),
                                   np.array([0.9, 0.8, 0.8]))
            memory.record(index, chosen.mask_score, 5.0, chosen.kf_score,
                          chosen.area_ratio)
            if index >= at:
                kept.append(memory.keep(index))
        return sum(kept), len(kept)

    def test_a_mask_that_swells_in_place_passes_every_other_check(self):
        """The failure the jump gate cannot see: the prediction is *inside* the
        swollen box, so the IoU against it stays respectable."""
        for factor in (2.0, 3.0, 4.0):
            with self.subTest(factor=factor):
                memory = MotionAwareMemory(SamuraiConfig(kf_gate=0.10))
                kept, total = self.swell(memory, factor)
                self.assertEqual(kept, total)

    def test_the_area_ratio_keeps_a_balloon_out_of_the_bank(self):
        for factor in (2.0, 3.0, 4.0):
            with self.subTest(factor=factor):
                memory = MotionAwareMemory(
                    SamuraiConfig(kf_gate=0.10, memory_area_ratio=3.0))
                self.assertEqual(self.swell(memory, factor)[0], 0)

    def test_a_target_that_really_approaches_is_not_a_balloon(self):
        """The ratio is against the running median of what was already
        accepted, so it follows a target growing frame by frame."""
        memory = MotionAwareMemory(
            SamuraiConfig(kf_gate=0.10, memory_area_ratio=3.0))
        position = np.array([300.0, 240.0])
        kept = []
        for index in range(40):
            position = position + np.array([3.0, 1.0])
            scale = 1.0 + 0.04 * index                  # 2.6x over the clip
            box = np.array([position[0] - 13 * scale, position[1] - 9 * scale,
                            position[0] + 13 * scale, position[1] + 9 * scale])
            chosen = memory.select(np.stack([box, box + 2, box - 2]),
                                   np.array([0.9, 0.8, 0.8]))
            memory.record(index, chosen.mask_score, 5.0, chosen.kf_score,
                          chosen.area_ratio)
            kept.append(memory.keep(index))
        self.assertGreaterEqual(sum(kept), 38)

    def test_a_refused_area_never_joins_the_history(self):
        """Or one balloon raises the median and the next one looks reasonable
        beside it."""
        memory = MotionAwareMemory(
            SamuraiConfig(kf_gate=0.10, memory_area_ratio=3.0))
        self.swell(memory, 4.0)
        self.assertLess(float(np.median(memory.areas)), 13.0 * 9.0 * 4.0 * 4.0)

    def test_a_reset_forgets_that_it_was_coasting(self):
        memory = MotionAwareMemory(SamuraiConfig(kf_gate=0.9))
        self.walk(memory, "jump", frames=20)
        memory.reset()
        self.assertEqual(memory.coasted, 0)
        self.assertEqual(len(memory.areas), 0)



def memgate() -> SamuraiConfig:
    """The shipped `memgate` policy, read from the file the runs use.

    Read rather than repeated: a test written against numbers typed a second
    time passes while the policy on disk says something else.
    """
    import yaml

    body = yaml.safe_load((ROOT / "configs/policies/memgate.yaml").read_text())
    block = dict(body["samurai"])
    block.pop("enabled", None)
    return SamuraiConfig(**block)


class MemoryGateTest(unittest.TestCase):
    """The gate between the mask and the memory write, with no SAMURAI in it.

    This is the layer the classical guard could not be: the guard runs after
    `propagate_in_video` has yielded, which is after the frame is already in
    the bank. Everything here is arithmetic on the box, done in the hook that
    sits between the decoder's choice and the memory write.
    """

    def walk(self, memory, kind, frames=34, jump_at=18):
        """Boxes moving smoothly, then either teleporting or accelerating."""
        position = np.array([300.0, 240.0])
        velocity = np.array([3.0, 1.0])
        kept = []
        for index in range(frames):
            if kind == "manoeuvre" and index >= jump_at:
                velocity = np.array([16.0, 6.0])
            position = position + velocity
            if kind == "jump" and index == jump_at:
                position = position + np.array([90.0, 0.0])
            box = np.array([position[0] - 13, position[1] - 9,
                            position[0] + 13, position[1] + 9])
            chosen = memory.select(np.stack([box, box + 2, box - 2]),
                                   np.array([0.9, 0.8, 0.8]))
            memory.record(index, chosen.mask_score, 5.0, chosen.kf_score,
                          chosen.area_ratio, chosen.jump)
            if index >= jump_at:
                kept.append(memory.keep(index))
        return sum(kept), len(kept)

    def test_a_jump_is_kept_out_of_the_bank_without_any_filter(self):
        """The measurement the policy claims: 16 of 16 without, 8 with."""
        self.assertEqual(self.walk(MotionAwareMemory(
            SamuraiConfig(kf_weight=0.0, memory_kf_score=-1.0)), "jump"), (16, 16))
        self.assertEqual(self.walk(MotionAwareMemory(memgate()), "jump")[0], 8)

    def test_a_real_manoeuvre_pays_nothing_for_it(self):
        """The half that makes a gate safe. A target accelerating to 16 px a
        frame moves 0.66 of its own length, and the gate is at 2.5."""
        self.assertEqual(self.walk(MotionAwareMemory(memgate()), "manoeuvre"),
                         (16, 16))

    def test_the_target_is_not_locked_out_forever(self):
        """`memory_patience`: after that many refusals in a row the tracker is
        the one more likely to be right, so the history re-seeds. Without this
        a target that really did move is refused for the rest of the clip."""
        memory = MotionAwareMemory(memgate())
        kept, total = self.walk(memory, "jump", frames=60)
        self.assertEqual(total, 42)
        # It recovers rather than refusing all 42.
        self.assertGreater(kept, 30)
        self.assertLess(memory.refused, memory.config.memory_patience)

    def test_the_jump_is_measured_against_the_last_accepted_box(self):
        """Not against the previous frame's. A refused frame does not become
        the thing the next one is judged against, or a mask that jumped would
        be at home the moment it arrived -- which is exactly how the guard's
        own gate was fooled before the anchor was separated from the track."""
        memory = MotionAwareMemory(memgate())
        box = np.array([100.0, 100.0, 126.0, 118.0])
        memory.select(np.stack([box] * 3), np.array([0.9, 0.8, 0.8]))
        far = box + np.array([200.0, 0.0, 200.0, 0.0])
        first = memory.select(np.stack([far] * 3), np.array([0.9, 0.8, 0.8]))
        second = memory.select(np.stack([far] * 3), np.array([0.9, 0.8, 0.8]))
        self.assertGreater(first.jump, memory.config.memory_jump)
        self.assertAlmostEqual(second.jump, first.jump, places=6)

    def test_motion_below_the_gate_moves_the_anchor_with_it(self):
        """The other direction, and the reason the gate is in units of the
        target's own size: 20 px a frame on a 26 px box is honest travel, so
        three of them are accepted and none is measured against frame one."""
        memory = MotionAwareMemory(memgate())
        moved = np.array([100.0, 100.0, 126.0, 118.0])
        for _ in range(4):
            chosen = memory.select(np.stack([moved] * 3), np.array([0.9, 0.8, 0.8]))
            moved = moved + np.array([20.0, 0.0, 20.0, 0.0])
        self.assertLess(chosen.jump, memory.config.memory_jump)
        self.assertEqual(memory.refused, 0)

    def test_a_jump_and_a_balloon_at_once_is_still_a_jump(self):
        """The unit is the *anchor's* size, not the new box's -- or a mask that
        ballooned would measure its own leap against its own inflated size."""
        memory = MotionAwareMemory(memgate())
        box = np.array([100.0, 100.0, 126.0, 118.0])
        memory.select(np.stack([box] * 3), np.array([0.9, 0.8, 0.8]))
        big = np.array([300.0, 100.0, 404.0, 172.0])       # 4x, and 200 px away
        chosen = memory.select(np.stack([big] * 3), np.array([0.9, 0.8, 0.8]))
        memory.record(1, chosen.mask_score, 5.0, chosen.kf_score,
                      chosen.area_ratio, chosen.jump)
        self.assertFalse(memory.keep(1))
        self.assertGreater(chosen.jump, memory.config.memory_jump)

    def test_the_filter_is_not_run_when_nothing_reads_it(self):
        """`kf_weight: 0` and no motion gate leaves the Kalman with no reader,
        and a policy that gates on geometry should not pay for a motion model
        it does not consult."""
        config = memgate()
        self.assertFalse(config.uses_filter)
        memory = MotionAwareMemory(config)
        self.walk(memory, "jump")
        self.assertFalse(memory.kalman.ready)
        self.assertTrue(SamuraiConfig().uses_filter)

    def test_an_accepted_frame_is_the_plain_run_s_own_mask(self):
        """The whole comparison rests on this: `memgate` beside `plain` differs
        in which frames the encoder remembers and in nothing else, so the
        candidate the decoder's argmax would have picked is the one used."""
        config = memgate()
        self.assertEqual(config.kf_weight, 0.0)
        memory = MotionAwareMemory(config)
        boxes = np.stack([np.array([100.0, 100.0, 126.0, 118.0]),
                          np.array([10.0, 10.0, 200.0, 200.0]),
                          np.array([104.0, 104.0, 130.0, 122.0])])
        ious = np.array([0.71, 0.93, 0.44])
        chosen = memory.select(boxes, ious)
        self.assertEqual(chosen.index, int(np.argmax(ious)))
        np.testing.assert_allclose(chosen.weighted, ious, rtol=1e-6)

    def test_a_reset_forgets_the_anchor(self):
        memory = MotionAwareMemory(memgate())
        self.walk(memory, "jump")
        memory.reset()
        self.assertIsNone(memory.anchor)
        self.assertEqual(memory.refused, 0)


class MemoryJournalTest(unittest.TestCase):
    """What the run says about itself afterwards.

    A refused frame is invisible in the output: the mask the pipeline receives
    is the one the decoder chose either way, and what changed is what the
    *next* frames read back. So the journal is the only evidence the gate did
    anything, and "it refused nothing" is a result, not a missing file.
    """

    def frame(self, samurai, box, iou=0.9, obj=5.0):
        masks = torch.full((1, 3, 32, 32), -10.0)
        y0, x0, y1, x1 = int(box[1]), int(box[0]), int(box[3]), int(box[2])
        masks[:, :, y0:y1, x0:x1] = 10.0
        samurai.rescore(masks, torch.tensor([[iou, iou - 0.1, iou - 0.2]]),
                        torch.tensor([[obj]]))
        samurai.frame_idx += 1

    def test_a_refused_frame_names_the_gate_that_refused_it(self):
        samurai = Samurai(memgate())
        self.frame(samurai, (4, 4, 10, 10))
        self.frame(samurai, (20, 20, 26, 26))
        self.assertTrue(samurai.journal[0]["kept"])
        self.assertFalse(samurai.journal[1]["kept"])
        self.assertIn("jumped", samurai.journal[1]["reason"])
        summary = samurai.summary()
        self.assertEqual(summary["judged"], 2)
        self.assertEqual(summary["refused"], 1)
        self.assertEqual(summary["by_gate"], {"jumped": 1})
        self.assertEqual(summary["frames_refused"], [1])

    def test_a_quiet_run_says_so_rather_than_saying_nothing(self):
        samurai = Samurai(memgate())
        for step in range(6):
            self.frame(samurai, (4 + step, 4, 10 + step, 10))
        self.assertEqual(samurai.summary()["refused"], 0)
        self.assertEqual(len(samurai.journal), 6)

    def test_the_journal_carries_the_numbers_the_gate_judged_on(self):
        samurai = Samurai(memgate())
        self.frame(samurai, (4, 4, 10, 10), iou=0.77, obj=3.5)
        row = samurai.journal[0]
        self.assertEqual(row["iou"], 0.77)
        self.assertEqual(row["obj"], 3.5)
        for key in ("frame", "object", "kept", "reason", "kf", "area_ratio",
                    "jump"):
            self.assertIn(key, row)

    def test_a_long_bad_clip_does_not_bury_the_run_log(self):
        """Every refusal is a row in the journal; only the first forty print."""
        import contextlib
        import io

        samurai = Samurai(memgate())
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.frame(samurai, (2, 2, 8, 8))
            # A mask thrown to the far corner and back, so `memory_patience`
            # re-seeding on one corner does not make the other one honest.
            for step in range(120):
                near = (2, 2, 8, 8) if step % 2 else (24, 24, 30, 30)
                self.frame(samurai, near)
        printed = [line for line in out.getvalue().splitlines()
                   if line.startswith("[memory] frame")]
        self.assertEqual(len(printed), Samurai.PRINT_LIMIT)
        self.assertGreater(samurai.summary()["refused"], Samurai.PRINT_LIMIT)
        self.assertIn("further refusals not printed", out.getvalue())

    def test_a_reset_forgets_the_journal(self):
        samurai = Samurai(memgate())
        self.frame(samurai, (4, 4, 10, 10))
        samurai.reset()
        self.assertEqual(samurai.journal, [])



class AcceleratedPathTest(unittest.TestCase):
    """The gate has to judge the frame on the TensorRT path too.

    `_reselect` is where the accelerated tracker hands the frame to this
    module. It used to return early when the engine had no `obj_ptr_all`,
    before the call that judges the frame -- so on any engine set exported the
    ordinary way the memory gate saw no records at all, and an unrecorded frame
    is kept. A gate that refuses nothing is indistinguishable in the output
    from one that was never on.
    """

    class _Engine:
        def __init__(self, pointers):
            self.output_names = (["pred_masks", "ious", "obj_ptr",
                                  "object_score_logits"]
                                 + (["obj_ptr_all"] if pointers else []))

    def tracker(self, config):
        from src.trackers.edgetam_trt_tracker import EdgeTAMTRTTracker
        from src.trackers.samurai import Samurai

        shell = EdgeTAMTRTTracker.__new__(EdgeTAMTRTTracker)
        shell._samurai = Samurai(config)
        shell._warned = set()
        return shell

    def frame(self, shell, engine, box):
        masks = torch.full((1, 3, 32, 32), -10.0)
        masks[:, :, box[1]:box[3], box[0]:box[2]] = 10.0
        from src.trackers.edgetam_trt_tracker import EdgeTAMTRTTracker

        out = {"low_res_multimasks": masks,
               "ious": torch.tensor([[0.9, 0.8, 0.7]]),
               "object_score_logits": torch.tensor([[5.0]])}
        EdgeTAMTRTTracker._reselect(shell, engine, out)
        shell._samurai.frame_idx += 1

    def test_the_frame_is_judged_even_without_the_extra_output(self):
        shell = self.tracker(memgate())
        engine = self._Engine(pointers=False)
        self.frame(shell, engine, (2, 2, 8, 8))
        self.frame(shell, engine, (22, 22, 28, 28))
        self.assertEqual(len(shell._samurai.journal), 2)
        self.assertFalse(shell._samurai.keep(1))

    def test_it_stays_silent_about_a_re_selection_nobody_asked_for(self):
        """The warning is about SAMURAI's mask choice being off. With
        `kf_weight: 0` the engine's own choice is the wanted one, and a warning
        naming a missing export would send a reader to a re-export they do not
        need."""
        import contextlib
        import io

        out = io.StringIO()
        shell = self.tracker(memgate())
        with contextlib.redirect_stdout(out):
            self.frame(shell, self._Engine(pointers=False), (2, 2, 8, 8))
        self.assertNotIn("obj_ptr_all", out.getvalue())

        loud = self.tracker(SamuraiConfig())
        with contextlib.redirect_stdout(out):
            self.frame(loud, self._Engine(pointers=False), (2, 2, 8, 8))
        self.assertIn("obj_ptr_all", out.getvalue())



if __name__ == "__main__":
    unittest.main()
