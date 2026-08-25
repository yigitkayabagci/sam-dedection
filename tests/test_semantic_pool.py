"""The semantic rescue: seeds, gates, dedup, and the measurement that judges it.

Same contract as `test_mask_pool.py`: the teacher needs a GPU and a checkpoint,
so the fake here stands in for the one property of SAM this whole module rests
on -- **a click returns the object under it, and the image shows a boundary the
label map does not**. `FakePointTeacher` reads the crop's own pixels to decide
what the click landed on, which is exactly the information `decompose` cannot
use and the reason this route can separate what it cannot.

Cases that need OpenCV (the distance transform, `decompose` itself) skip
themselves cleanly when it is absent, as the rest of the suite does.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.aerial import DatasetSpec, decompose  # noqa: E402
from src.training.semantic import (  # noqa: E402
    SemanticGates,
    measure_rescue,
    class_purity,
    dedupe_masks,
    match_instances,
    measure_semantic,
    rescue_frame,
    seed_count,
    seed_points,
    semantic_from_instances,
    semantic_reject,
    summarise_rescue,
    summarise_semantic_pool,
    unit_areas,
)

try:
    import cv2  # noqa: F401
    HAVE_CV2 = True
except ImportError:  # pragma: no cover - environment, not logic
    HAVE_CV2 = False


SPEC = DatasetSpec(
    name="toy",
    masks="**/label/*.png",
    classes={"background": 0, "road": 1, "car": 2},
    things=("car",),
    thermal="**/tir/*.png",
    rgb="**/rgb/*.png",
    ignore=(0,),
    palette_source="synthetic, defined in this test",
)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def two_parked_cars(gap: int = 2):
    """`(pixels, semantic)` -- two cars the map fuses and the image separates.

    The semantic map paints both cars class 2 with **no gap**, which is what a
    human annotator produces for two cars parked flush: the boundary between
    them is not a class boundary, so it is not in the map. The image keeps them
    apart by intensity, which is the only place that boundary survives.
    """
    height, width = 120, 200
    pixels = np.full((height, width, 3), 30, np.uint8)
    semantic = np.zeros((height, width), np.uint8)
    semantic[:] = 1                                   # road everywhere
    for index, x0 in enumerate((40, 100)):
        pixels[50:80, x0:x0 + 56] = 200 + index * 20  # two distinguishable cars
    # one fused class-2 region across both, plus the gap between them
    semantic[50:80, 40:156] = 2
    if gap:
        pixels[50:80, 96:96 + gap] = 30               # the seam, in pixels only
    return pixels, semantic


class FakePointTeacher:
    """Returns the connected bright region the click landed on.

    Stands in for the property that makes the rescue possible: SAM answers a
    click with the object under it, and the object's extent comes from the
    *image*. `bleed` grows the answer past the object so the `purity` gate can
    be aimed at it; `whole_region` makes it answer with everything bright,
    which is the failure mode `unit_area` exists to catch.
    """

    model_id = "fake/point-teacher"

    def __init__(self, bleed: int = 0, whole_region: bool = False,
                 score: float = 0.9) -> None:
        self.bleed = bleed
        self.whole_region = whole_region
        self.score = score
        self.calls = 0

    def points_for(self, crops, points):
        out = []
        for crop, point in zip(crops, points):
            self.calls += 1
            grey = crop[..., 0].astype(np.int32)
            x, y = int(round(point[0])), int(round(point[1]))
            x = int(np.clip(x, 0, grey.shape[1] - 1))
            y = int(np.clip(y, 0, grey.shape[0] - 1))
            if self.whole_region:
                # One filled blob over every bright pixel: the "answered with
                # the whole parked row" failure, and connected, so `component`
                # cannot catch it and only the size prior can.
                mask = np.zeros(grey.shape, bool)
                ys, xs = np.nonzero(grey > 100)
                if len(ys):
                    mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = True
            else:
                value = grey[y, x]
                mask = np.abs(grey - value) <= 5
            if self.bleed:
                mask = mask.copy()
                ys, xs = np.nonzero(mask)
                if len(ys):
                    y0 = max(int(ys.min()) - self.bleed, 0)
                    y1 = min(int(ys.max()) + self.bleed + 1, mask.shape[0])
                    x0 = max(int(xs.min()) - self.bleed, 0)
                    x1 = min(int(xs.max()) + self.bleed + 1, mask.shape[1])
                    mask[y0:y1, x0:x1] = True
            out.append((mask, self.score))
        return out


# --------------------------------------------------------------------------
# Pieces
# --------------------------------------------------------------------------


class TestPurity(unittest.TestCase):
    def test_purity_reads_the_drawn_class_under_the_mask(self):
        semantic = np.zeros((10, 10), np.uint8)
        semantic[2:6, 2:6] = 2
        mask = np.zeros((10, 10), bool)
        mask[2:6, 2:8] = True                 # half of it is off the class
        self.assertAlmostEqual(class_purity(mask, semantic, [2]), 16 / 24, places=6)

    def test_an_empty_mask_scores_zero_rather_than_dividing_by_it(self):
        self.assertEqual(class_purity(np.zeros((4, 4), bool),
                                      np.zeros((4, 4), np.uint8), [2]), 0.0)


class TestSeeds(unittest.TestCase):
    def test_seed_count_follows_area_over_one_unit(self):
        self.assertEqual(seed_count(100.0, 100.0), 1)
        self.assertEqual(seed_count(600.0, 100.0), 6)
        self.assertEqual(seed_count(6000.0, 100.0, cap=4), 4)

    def test_a_class_with_no_unit_gets_one_seed_not_a_crash(self):
        self.assertEqual(seed_count(500.0, None), 1)
        self.assertEqual(seed_count(500.0, 0.0), 1)

    def test_unit_area_is_the_median_component_not_the_mean(self):
        class Fake:
            def __init__(self, cls, area):
                self.class_id, self.area = cls, area
        # four singles and one fused blob: the mean is dragged, the median is not
        instances = [Fake(2, 100), Fake(2, 110), Fake(2, 90), Fake(2, 100),
                     Fake(2, 900)]
        self.assertEqual(unit_areas(instances)[2], 100.0)

    def test_an_override_wins_over_the_frame(self):
        class Fake:
            def __init__(self, cls, area):
                self.class_id, self.area = cls, area
        self.assertEqual(unit_areas([Fake(2, 100)], {2: 42.0})[2], 42.0)

    @unittest.skipUnless(HAVE_CV2, "distance transform needs OpenCV")
    def test_seeds_land_inside_and_spread_apart(self):
        component = np.zeros((60, 200), bool)
        component[10:50, 10:70] = True
        component[10:50, 120:180] = True
        points = seed_points(component, 2)
        self.assertEqual(len(points), 2)
        for x, y in points:
            self.assertTrue(component[y, x], "a seed must be inside its component")
        self.assertGreater(abs(points[0][0] - points[1][0]), 40,
                           "two seeds must not land on the same lobe")

    @unittest.skipUnless(HAVE_CV2, "distance transform needs OpenCV")
    def test_an_empty_component_yields_no_seeds(self):
        self.assertEqual(seed_points(np.zeros((10, 10), bool), 3), [])


class TestDedupe(unittest.TestCase):
    def test_near_duplicates_collapse_and_the_larger_survives(self):
        small = np.zeros((20, 20), bool); small[5:10, 5:10] = True
        big = np.zeros((20, 20), bool); big[5:11, 5:11] = True
        far = np.zeros((20, 20), bool); far[15:19, 15:19] = True
        keep = dedupe_masks([small, big, far], threshold=0.5)
        self.assertIn(1, keep, "the larger of a duplicate pair is kept")
        self.assertIn(2, keep, "a disjoint mask is never a duplicate")
        self.assertNotIn(0, keep)


class TestGates(unittest.TestCase):
    def measurement(self, **kw):
        base = dict(purity=1.0, containment=1.0, unit_ratio=1.0,
                    component=1.0, teacher_iou=0.9)
        base.update(kw)
        from src.training.semantic import SemanticMeasurement
        return SemanticMeasurement(**base)

    def test_each_gate_names_itself(self):
        gates = SemanticGates()
        self.assertIsNone(semantic_reject(self.measurement(), gates))
        self.assertEqual(semantic_reject(self.measurement(purity=0.1), gates),
                         "purity")
        self.assertEqual(semantic_reject(self.measurement(containment=0.1), gates),
                         "containment")
        self.assertEqual(semantic_reject(self.measurement(unit_ratio=9.0), gates),
                         "unit_area")
        self.assertEqual(semantic_reject(self.measurement(component=0.1), gates),
                         "component")

    def test_a_mask_that_swallowed_the_road_fails_on_purity_alone(self):
        semantic = np.zeros((40, 40), np.uint8)
        semantic[:] = 1
        semantic[10:20, 10:20] = 2                 # the car
        mask = np.zeros((40, 40), bool)
        mask[10:20, 10:30] = True                  # car plus as much road again
        component = semantic == 2
        m = measure_semantic(mask, semantic, component, [2],
                             unit_area=100.0, teacher_iou=0.9)
        self.assertAlmostEqual(m.purity, 0.5, places=6)
        self.assertEqual(semantic_reject(m, SemanticGates()), "purity")


# --------------------------------------------------------------------------
# The whole frame -- the claim this module is built on
# --------------------------------------------------------------------------


@unittest.skipUnless(HAVE_CV2, "decompose and the distance transform need OpenCV")
class TestRescueFrame(unittest.TestCase):
    def test_decompose_fuses_two_parked_cars_and_the_rescue_separates_them(self):
        pixels, semantic = two_parked_cars()

        _, instances, _ = decompose(semantic, SPEC)
        self.assertEqual(len(instances), 1,
                         "the premise: one class region, so one component")

        masks, records = rescue_frame(
            pixels, semantic, SPEC, FakePointTeacher(),
            units={2: 30 * 56}, zoom=2.0, min_size=64)
        self.assertEqual(len(masks), 2, "a click each -> two cars")
        self.assertTrue(all(m.sum() > 0 for m in masks))
        # and they are genuinely different objects, not one answer twice
        from src.training.pool import iou
        self.assertLess(iou(masks[0], masks[1]), 0.1)
        self.assertTrue(any(r.get("kept") for r in records))

    def test_a_teacher_that_answers_with_the_whole_row_is_rejected(self):
        """The failure this route exists to avoid, caught by `unit_area`.

        Answering every click with the fused region would score perfectly on
        purity and containment -- it is the right class, inside the component --
        and would put a two-car mask into the store wearing an instance label.
        Only the size prior sees it.
        """
        pixels, semantic = two_parked_cars()
        masks, records = rescue_frame(
            pixels, semantic, SPEC, FakePointTeacher(whole_region=True),
            units={2: 30 * 56}, zoom=2.0, min_size=64)
        self.assertEqual(masks, [])
        self.assertTrue(any(r["verdict"] == "unit_area" for r in records),
                        f"expected a unit_area rejection, got "
                        f"{[r['verdict'] for r in records]}")

    def test_a_frame_with_no_thing_class_produces_nothing_quietly(self):
        semantic = np.ones((40, 40), np.uint8)          # road only
        pixels = np.zeros((40, 40, 3), np.uint8)
        self.assertEqual(rescue_frame(pixels, semantic, SPEC,
                                      FakePointTeacher()), ([], []))


# --------------------------------------------------------------------------
# The measurement
# --------------------------------------------------------------------------


class TestMeasurement(unittest.TestCase):
    def test_flattening_instances_throws_the_separation_away(self):
        instance_map = np.zeros((20, 40), np.int32)
        instance_map[5:15, 5:20] = 7
        instance_map[5:15, 20:35] = 9                  # touching neighbour
        semantic = semantic_from_instances(instance_map, {7: 2, 9: 2})
        self.assertEqual(sorted(np.unique(semantic).tolist()), [0, 2])
        self.assertEqual(int((semantic == 2).sum()), int((instance_map > 0).sum()))

    def test_matching_is_one_to_one_and_scores_both_directions(self):
        a = np.zeros((20, 20), bool); a[2:8, 2:8] = True
        b = np.zeros((20, 20), bool); b[12:18, 12:18] = True
        drawn = [a, b]
        # one perfect answer and one duplicate of it: recall 1/2, precision 1/2
        found = [a.copy(), a.copy()]
        stats = match_instances(found, drawn)
        self.assertEqual(stats["matched"], 1)
        self.assertAlmostEqual(stats["recall"], 0.5)
        self.assertAlmostEqual(stats["precision"], 0.5)

    @unittest.skipUnless(HAVE_CV2, "decompose needs OpenCV")
    def test_the_rescue_beats_the_baseline_on_the_case_it_exists_for(self):
        """End to end, on truth: two drawn cars the flattened map fuses.

        This is the whole claim in one assertion. `components` can recover at
        most one of the two, because after flattening there is one region;
        `rescue` gets both, because the image still has the seam. If this ever
        stops holding, the route is not worth its GPU-hours and the notebook's
        table will say so on real data too.
        """
        # A realistic set: four frames whose cars stand alone -- they are what
        # tells the estimator how big one car is -- and one where two are
        # parked flush. That mix is the assumption `unit_areas` rests on, so
        # the test has to contain it rather than assume it away.
        samples = []
        for offset in range(4):
            lone = np.full((120, 200, 3), 30, np.uint8)
            lone[50:80, 40 + offset:96 + offset] = 200
            ids = np.zeros((120, 200), np.int32)
            ids[50:80, 40 + offset:96 + offset] = 1
            samples.append((lone, ids, {1: 2}))

        pixels, _ = two_parked_cars()
        fused = np.zeros(pixels.shape[:2], np.int32)
        fused[50:80, 40:98] = 1                      # drawn car 1
        fused[50:80, 98:156] = 2                     # drawn car 2, touching
        samples.append((pixels, fused, {1: 2, 2: 2}))

        routes = measure_rescue(samples, SPEC, FakePointTeacher(),
                                zoom=2.0, min_size=64)
        base, rescued = routes["components"][-1], routes["rescue"][-1]
        self.assertEqual(base["drawn"], 2)
        self.assertEqual(base["found"], 1, "flattening fuses them into one")
        self.assertLessEqual(base["matched"], 1)
        self.assertEqual(rescued["found"], 2, "the teacher separated them")
        self.assertGreater(rescued["recall"], base["recall"])
        # and the four unfused frames are not made worse by the rescue
        for i in range(4):
            self.assertGreaterEqual(routes["rescue"][i]["recall"],
                                    routes["components"][i]["recall"])

    def test_the_tables_render_without_a_gpu(self):
        rows = [{"drawn": 10, "found": 12, "matched": 9, "recall": 0.9,
                 "precision": 0.75, "mean_iou": 0.81}]
        table = summarise_rescue({"components": rows, "rescue": rows})
        self.assertIn("| components |", table)
        self.assertIn("| rescue |", table)
        block = summarise_semantic_pool(
            {"frames": 3, "components": 10, "seeds": 22, "accepted": 15,
             "instances_per_component": 1.5, "acceptance_rate": 0.68,
             "rejected": {"purity": 4, "unit_area": 3}})
        self.assertIn("instances per component", block)
        self.assertIn("purity", block)


if __name__ == "__main__":
    unittest.main()
