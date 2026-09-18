"""Characterize the existing analysis and export behavior without model downloads."""

from contextlib import contextmanager, redirect_stdout
from hashlib import sha256
from io import BytesIO, StringIO
import json
from pathlib import Path
import random
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from antipodality import analysis, cli, io, pipeline
from antipodality.similarity import cosine_matrix
from antipodality.utils import MatryoshkaUtils, prepare_for_json
from antipodality.viz import payloads, plots


def fixture_data():
    rng = np.random.default_rng(123456)
    W_enc = rng.normal(size=(140, 8)).astype(np.float32)
    W_dec = rng.normal(size=(140, 8)).astype(np.float32)
    for first, second in ((0, 1), (2, 3), (126, 128), (127, 129)):
        W_enc[second] = -W_enc[first]
        W_dec[second] = -W_dec[first]
    densities = np.linspace(0.001, 0.04, 140)
    dense_indices = np.r_[0:12, 126:132]
    densities[dense_indices] = 0.1 + 0.005 * np.arange(len(dense_indices))
    return densities, W_enc, W_dec


def run_fixture(render=False, make_umap=False, antipodal_only=True, build_within_cross=True):
    densities, W_enc, W_dec = fixture_data()
    text_outputs = {}
    image_hashes = {}
    log = StringIO()
    savefig = plots.plt.savefig

    @contextmanager
    def capture_text(path, *args, **kwargs):
        buffer = StringIO()
        yield buffer
        text_outputs[Path(path).name] = buffer.getvalue()
        buffer.close()

    def capture_figure(path, *args, **kwargs):
        if render:
            buffer = BytesIO()
            savefig(buffer, *args, format="png", **kwargs)
            image_hashes[Path(path).name] = sha256(buffer.getvalue()).hexdigest()

    with (
        patch.object(pipeline.io, "load_density_data", return_value=densities),
        patch.object(pipeline.io, "load_sae_weights", return_value=(
            W_enc, W_dec, {"d_sae": 140, "d_in": 8}
        )),
        patch.object(Path, "mkdir"),
        patch.object(pipeline, "open", capture_text, create=True),
        patch.object(plots.plt, "savefig", capture_figure),
        redirect_stdout(log),
    ):
        summary = pipeline.run(
            "fixture.npz", 2, out_dir="fixture_results", block_size=23,
            make_umap=make_umap, antipodal_only=antipodal_only,
            build_within_cross=build_within_cross,
        )
    return summary, text_outputs, image_hashes, log.getvalue()


def snapshot(**kwargs):
    summary, outputs, images, log = run_fixture(**kwargs)
    hashes = {name: sha256(text.encode()).hexdigest() for name, text in outputs.items()}
    hashes.update(images)
    hashes["stdout"] = sha256(log.encode()).hexdigest()
    hashes["summary"] = sha256(json.dumps(prepare_for_json(summary)).encode()).hexdigest()
    return hashes


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.stdout = redirect_stdout(StringIO())
        self.stdout.__enter__()
        self.addCleanup(self.stdout.__exit__, None, None, None)

    def test_blocked_scores_match_full_matrix(self):
        rng = np.random.default_rng(3)
        W_enc = rng.normal(size=(12, 5)).astype(np.float32)
        W_dec = rng.normal(size=(12, 5)).astype(np.float32)
        C_enc, C_dec = cosine_matrix(W_enc), cosine_matrix(W_dec)
        expected = np.where((C_enc < 0) & (C_dec < 0), C_enc * C_dec, -np.inf)
        for block_size in (1, 5, 12):
            with self.subTest(block_size=block_size):
                result = analysis.compute_antipodality_scores(
                    W_enc, W_dec, top_k=3, block_size=block_size
                )
                np.testing.assert_allclose(result["antipodality_scores"], expected.max(axis=1), atol=1e-6)

    def test_top_k_keeps_one_partner_per_feature(self):
        angles = np.array([0, 0.1, np.pi, np.pi + 0.1])
        weights = np.column_stack((np.cos(angles), np.sin(angles))).astype(np.float32)
        for top_k in (1, 3):
            with self.subTest(top_k=top_k):
                result = analysis.compute_antipodality_scores(weights, weights, top_k=top_k, block_size=2)
                np.testing.assert_allclose(result["antipodality_scores"], np.ones(4), atol=1e-6)
                np.testing.assert_array_equal(result["antipodal_partners"], [2, 3, 0, 1])
                pairs = analysis.find_top_pairs(
                    result["feature_indices"], result["antipodality_scores"],
                    result["antipodal_partners"], top_k=100,
                )
                self.assertEqual([(p["feature1_idx"], p["feature2_idx"]) for p in pairs], [(0, 2), (1, 3)])

    def test_dense_search_uses_only_selected_features(self):
        weights = np.array([[1, 0], [-1, 0], [-0.1, np.sqrt(0.99)]], dtype=np.float32)
        result = analysis.compute_antipodality_scores(weights, weights, feature_indices=np.array([0, 2]))
        np.testing.assert_array_equal(result["feature_indices"], [0, 2])
        np.testing.assert_array_equal(result["antipodal_partners"], [1, 0])
        np.testing.assert_allclose(result["antipodality_scores"], [0.01, 0.01], atol=1e-7)

    def test_combined_mode_includes_positive_alignment(self):
        weights = np.array([[1, 0], [1, 0]], dtype=np.float32)
        result = analysis.compute_antipodality_scores(weights, weights, antipodal_only=False)
        np.testing.assert_array_equal(result["antipodality_scores"], [1, 1])
        np.testing.assert_array_equal(result["antipodal_partners"], [1, 0])

    def test_density_threshold_is_strict(self):
        indices, levels = analysis.dense_feature_indices(np.array([0.0, 0.05, 0.06]), 0.05)
        np.testing.assert_array_equal(indices, [2])
        self.assertEqual(levels[128]["dense_count"], 1)
        self.assertEqual(levels[128]["total_count"], 128)

    def test_nonfinite_scores_are_excluded_from_summary(self):
        stats = analysis.validate_scores(np.array([1.0, -np.inf, np.nan, 0.5]))
        self.assertEqual(stats, {
            "count": 2, "mean": 0.75, "std": 0.25,
            "min": 0.5, "max": 1.0, "median": 0.75,
        })

    def test_level_boundaries_are_exclusive(self):
        indices = [0, 127, 128, 511, 512, 2047, 2048, 8191, 8192, 32767, 32768]
        expected = [128, 128, 512, 512, 2048, 2048, 8192, 8192, 32768, 32768, None]
        self.assertEqual([MatryoshkaUtils.get_level(i) for i in indices], expected)

    def test_json_preserves_keys_and_replaces_nonfinite_values(self):
        values = {"scores": np.array([0.5, np.inf, np.nan]), "count": np.int64(3), "path": Path("results")}
        self.assertEqual(prepare_for_json(values), {"scores": [0.5, None, None], "count": 3, "path": "results"})


class IOTests(unittest.TestCase):
    def test_density_loading_preserves_float32_conversion(self):
        buffer = BytesIO()
        np.savez(buffer, densities=np.array([0.0, 0.05, 0.2], dtype=np.float64))
        buffer.seek(0)
        with redirect_stdout(StringIO()):
            densities = io.load_density_data(buffer)
        self.assertEqual(densities.dtype, np.float32)
        np.testing.assert_array_equal(densities, np.array([0.0, 0.05, 0.2], dtype=np.float32))

    def test_weight_loading_preserves_feature_rows(self):
        sae = SimpleNamespace(
            W_enc=torch.arange(6, dtype=torch.float64).reshape(2, 3),
            W_dec=torch.arange(6, dtype=torch.float64).reshape(3, 2),
            cfg=SimpleNamespace(d_sae=3, d_in=2),
            eval=Mock(),
        )
        with patch.object(io.SAE, "from_pretrained", return_value=sae) as load, redirect_stdout(StringIO()):
            W_enc, W_dec, cfg = io.load_sae_weights("fixture/repo", 12)
        load.assert_called_once_with("fixture/repo", "blocks.12.hook_resid_post", device="cpu")
        sae.eval.assert_called_once_with()
        np.testing.assert_array_equal(W_enc, [[0, 3], [1, 4], [2, 5]])
        np.testing.assert_array_equal(W_dec, [[0, 1], [2, 3], [4, 5]])
        self.assertEqual(W_enc.dtype, np.float32)
        self.assertEqual(W_dec.dtype, np.float32)
        self.assertEqual(cfg, {
            "repo": "fixture/repo", "layer": 12,
            "sae_id": "blocks.12.hook_resid_post", "d_sae": 3, "d_in": 2,
        })


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary, cls.outputs, _, _ = run_fixture()
        cls.results = json.loads(cls.outputs["antipodality_analysis_layer_2.json"])

    def test_export_schema_and_counts(self):
        self.assertEqual(self.summary["key_statistics"]["total_features"], 140)
        self.assertEqual(self.summary["key_statistics"]["dense_features"], 18)
        self.assertEqual(self.summary["counts"]["matryoshka_levels"], {128: 14, 512: 4})
        self.assertEqual(len(self.summary["output_files"]), 8)
        self.assertEqual(self.results["analysis_metadata"]["density_threshold"], 0.05)
        self.assertFalse({"densities", "W_enc", "W_dec"} & self.results["analysis_metadata"].keys())
        csv_lines = self.outputs["antipodal_pairs_layer_2.csv"].splitlines()
        self.assertEqual(csv_lines[0], "pair_id,feature1_idx,feature2_idx,feature1_density,feature2_density,encoder_sim,decoder_sim,antipodal_score,feature1_level,feature2_level")
        self.assertEqual(len(csv_lines) - 1, self.summary["key_statistics"]["top_antipodal_pairs"])

    def test_optional_plots_do_not_change_exports(self):
        summary, outputs, _, _ = run_fixture(build_within_cross=False)
        self.assertNotIn("within_cross_comparison", summary["output_files"])
        self.assertEqual(outputs, self.outputs)

    def test_scatter_sampling_uses_fixed_seed(self):
        densities, W_enc, W_dec = fixture_data()
        results = {**self.results, "analysis_metadata": {
            **self.results["analysis_metadata"], "densities": densities,
            "W_enc": W_enc, "W_dec": W_dec,
        }}
        first = payloads.build_enc_dec_scatter_payload(results, 8, 10, rng=random.Random(1))
        second = payloads.build_enc_dec_scatter_payload(results, 8, 10, rng=random.Random(99))
        np.testing.assert_array_equal(first.enc_sim, second.enc_sim)
        np.testing.assert_array_equal(first.dec_sim, second.dec_sim)
        np.testing.assert_array_equal(first.pair_score, second.pair_score)

    def test_empty_matrix_payload_retains_current_error(self):
        with self.assertRaisesRegex(KeyError, "Missing payload keys"):
            plots.plot_unbiased_antipodal_analysis({"layer": 2, "has_data": False}, BytesIO())
        with self.assertRaisesRegex(KeyError, "Missing payload keys"):
            plots.plot_dense_focused_matrix({"layer": 2, "has_data": False}, BytesIO())

    def test_cli_forwards_options(self):
        args = ["antipodality", "fixture.npz", "--layer", "2", "--top-k-pairs", "20", "--seed", "7",
                "--no-antipodal-only", "--no-within-cross", "--no-umap"]
        with (
            patch("sys.argv", args), patch.object(Path, "exists", return_value=True),
            patch.object(cli.pipeline, "run", return_value=self.summary) as run,
            redirect_stdout(StringIO()),
        ):
            cli.main()
        run.assert_called_once_with(
            npz_path="fixture.npz", layer=2, sae_repo="gemma-2-2b-res-matryoshka-dc",
            out_dir="antipodality_analysis", density_threshold=None, top_k_pairs=20,
            block_size=2048, antipodal_only=False, build_within_cross=False,
            make_umap=False, umap_neighbors=15, rng_seed=7,
        )


if __name__ == "__main__":
    unittest.main()
