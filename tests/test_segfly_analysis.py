"""The SegFly audit uses the same conversion as training, without forking it."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.aerial import SPECS, Source  # noqa: E402
from tools.analyze_segfly_instances import (  # noqa: E402
    SEGFLY_COLORS,
    audit,
    colorize_semantic,
)


class TestSegflyAudit(unittest.TestCase):
    def setUp(self):
        self.source = Source(SPECS["segfly"], role="train")

    def test_the_display_palette_does_not_change_class_ids(self):
        semantic = np.array([[0, 13, 36, 99]], dtype=np.int32)
        drawn = colorize_semantic(semantic)
        self.assertEqual(tuple(drawn[0, 0]), SEGFLY_COLORS[0])
        self.assertEqual(tuple(drawn[0, 1]), SEGFLY_COLORS[13])
        self.assertEqual(tuple(drawn[0, 2]), SEGFLY_COLORS[36])
        self.assertEqual(tuple(drawn[0, 3]), (255, 0, 255),
                         "an unexpected id must be visible, not silently black")
        np.testing.assert_array_equal(semantic, [[0, 13, 36, 99]])

    def test_touching_different_classes_are_never_one_instance(self):
        semantic = np.zeros((40, 40), dtype=np.uint8)
        semantic[8:20, 8:20] = 13
        semantic[8:20, 20:32] = 36       # touches vehicle along a full edge
        modes = audit(semantic, self.source)
        for name, result in modes.items():
            with self.subTest(mode=name):
                self.assertEqual(len(result.instances), 2)
                self.assertEqual({i.class_id for i in result.instances}, {13, 36})
                self.assertEqual(result.rejects, {})

    def test_stuff_classes_never_become_targets(self):
        semantic = np.full((40, 40), SPECS["segfly"].classes["tree"],
                           dtype=np.uint8)
        modes = audit(semantic, self.source)
        for result in modes.values():
            self.assertEqual(result.instances, ())
            self.assertEqual(result.rejects, {})

    def test_record_names_classes_instead_of_exposing_only_ids(self):
        semantic = np.zeros((40, 40), dtype=np.uint8)
        semantic[5:20, 5:20] = 13
        result = audit(semantic, self.source)["components"].record(self.source)
        self.assertEqual(result["by_class"], {"vehicle": 1})
        self.assertEqual(result["kept"], 1)


if __name__ == "__main__":
    unittest.main()
