"""Portable old-code fixtures complement the real saved-prediction CLI replay."""

import base64
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from densekp_retrain.evaluation.candidate_inputs import prepare_inputs
from densekp_retrain.evaluation.pairs import collect_pair_metrics
from densekp_retrain.io.json import write_json, write_json_ascii
from densekp_retrain.visualization.keypoints import render_candidate_panel
from tests import test_evaluation_contract as contract


class CandidateReportingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = contract.decode(
            json.loads((Path(__file__).parent / "fixtures/candidate_reporting_v1.json").read_text())
        )

    def test_pairing_matches_independent_old_results(self):
        for record in self.fixture["pairs"]:
            with self.subTest(scenario=record["scenario"]):
                pairs = {tuple(entry["key"]): entry["views"] for entry in record["pairs"]}
                results, buckets = collect_pair_metrics(pairs, record["states"])
                self.assert_exact(results, record["results"])
                buckets = {
                    key: {name: np.asarray(value) for name, value in bucket.items()}
                    for key, bucket in buckets.items()
                }
                self.assert_exact(buckets, record["buckets"])

    # Reuse the recursive assertion without inheriting and duplicating its tests.
    assert_exact = contract.EvaluationContractTests.assert_exact

    def test_panel_pixels_and_historical_keyword_signature(self):
        from tools.validate.evaluate_candidate_baseline import render_panel

        fixture = self.fixture["panel"]
        expected = np.array(Image.open(io.BytesIO(base64.b64decode(fixture["png_base64"]))))
        for panel in (
            render_candidate_panel(
                fixture["item"],
                fixture["prediction"],
                fixture["supervised"],
                fixture["visible"],
                fixture["label"],
            ),
            render_panel(
                item=fixture["item"],
                pred=fixture["prediction"],
                supervised=fixture["supervised"],
                visible=fixture["visible"],
                label=fixture["label"],
            ),
        ):
            self.assertEqual(panel.size, (512, 512))
            np.testing.assert_array_equal(np.array(panel), expected)

    def test_json_unicode_order_and_nonfinite_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "report.json"
            value = {"场景": [2, None], "status": "通过"}
            for writer, ascii_policy in ((write_json, False), (write_json_ascii, True)):
                writer(path, value)
                expected = (
                    json.dumps(value, ensure_ascii=ascii_policy, indent=2, allow_nan=False) + "\n"
                )
                self.assertEqual(path.read_text(), expected)
                for number in (float("nan"), float("inf")):
                    with self.assertRaises(ValueError):
                        writer(path, {"value": number})
                    self.assertEqual(path.read_text(), expected)

    def test_input_rejection_preserves_copy_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            output = directory / "output"
            output.mkdir()
            first, second = directory / "a.json", directory / "b.json"
            accepted = {"split": "val", "rows": [{"quality_state": "candidate_accepted"}]}
            first.write_text(json.dumps(accepted))
            second.write_text(json.dumps(accepted))
            with self.assertRaisesRegex(ValueError, "Use one index per partition per baseline run"):
                prepare_inputs([first, second], ["a", "b"], None, output)
            self.assertEqual((output / "a_frozen_index.json").read_bytes(), first.read_bytes())
            self.assertFalse((output / "b_frozen_index.json").exists())
            second.write_text(json.dumps({"rows": [{"quality_state": "rejected"}]}))
            with self.assertRaisesRegex(
                ValueError, "Inference requires pre-frozen, accepted candidate rows"
            ):
                prepare_inputs([second], ["b"], None, output)
            self.assertFalse((output / "b_frozen_index.json").exists())


if __name__ == "__main__":
    unittest.main()
