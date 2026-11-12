"""Synthetic correctness checks; no datasets or pretrained weights required."""

import contextlib
import io
import json
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest

import h5py
import numpy as np
import torch

from profuse_st.data import build_graph, gene_embeddings, load_slide, prepare_slide
from profuse_st.loss import expression_loss
from profuse_st.metrics import expression_metrics
from profuse_st.model import ProFuseST
from profuse_st.splits import make_split, spatial_labels
from profuse_st.train import train, validate_split


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.genes = [f"gene{i}" for i in range(5)]
        self.barcodes = [f"spot{i}" for i in range(32)]
        self.rng = np.random.default_rng(42)
        self.target = self.rng.random((32, 5), dtype=np.float32)
        for name in ("train_section", "val_section", "test_section"):
            directory = self.root / name
            directory.mkdir()
            with open(directory / "gene_expression_ground_truth.pkl", "wb") as f:
                pickle.dump({"expression": self.target, "barcodes": self.barcodes,
                             "genes": self.genes, "normalization": "log1p(1e4)"}, f)
            with h5py.File(directory / "feature_uni.h5", "w") as f:
                order = np.arange(32)[::-1]
                features = self.rng.normal(size=(32, 8)).astype(np.float32)
                f["embeddings"] = features[order]
                f["coords"] = np.array([(i % 8, i // 8) for i in order], dtype=np.float32)
                f["barcodes"] = np.array([self.barcodes[i] for i in order], dtype="S")
            with h5py.File(directory / "slide_feature_gigapath.h5", "w") as f:
                f["features"] = self.rng.normal(size=(1, 4)).astype(np.float32)
        self.embeddings = self.root / "genes.txt"
        # Deliberately reversed rows, with a word2vec header.
        self.embeddings.write_text("5 4\n" + "\n".join(
            gene + " " + " ".join([str(i + 0.1)] * 4)
            for i, gene in reversed(list(enumerate(self.genes)))))
        self.split = {"train": ["train_section"], "val": ["val_section"], "test": ["test_section"]}

    def test_identifier_alignment(self):
        slide = load_slide(self.root, "train_section")
        self.assertEqual(slide["barcodes"], self.barcodes)
        self.assertEqual(slide["coords"][0].tolist(), [0.0, 0.0])
        vectors = gene_embeddings(self.embeddings, self.genes)
        self.assertAlmostEqual(float(vectors[0, 0]), 0.1, places=6)
        self.assertAlmostEqual(float(vectors[-1, 0]), 4.1, places=5)

    def test_graphs_and_subset_cache(self):
        slide = load_slide(self.root, "train_section")
        graph = build_graph(slide["features"], slide["coords"], k=6)
        for neighbors in graph.values():
            self.assertFalse((neighbors == torch.arange(32)[:, None]).any())
        graph["barcodes"] = slide["barcodes"]
        torch.save(graph, self.root / "train_section" / "graph_faiss.pt")
        subset = prepare_slide(self.root, "train_section", selection=[0, 5, 9, 11], k=6)
        for neighbors in subset["graph"].values():
            self.assertEqual(tuple(neighbors.shape), (4, 6))
            self.assertTrue(((neighbors >= 0) & (neighbors < 4)).all())
        # Feature rows are reversed; inference must remap a graph in expression order.
        inference = prepare_slide(self.root, "train_section", require_expression=False)
        for key in ("spatial_neighbors", "semantic_neighbors"):
            expected = 31 - graph[key].flip(0)
            torch.testing.assert_close(inference["graph"][key], expected)

    def test_complete_model_backward(self):
        slide = prepare_slide(self.root, "train_section")
        model = ProFuseST(8, 4, 5, hidden_dim=8, num_heads=2, num_programs=3,
                          global_dim=4, local_depth=1, global_depth=1)
        prediction = model(slide["features"], gene_embeddings(self.embeddings, self.genes),
                           slide["coords"], slide["global_feature"], slide["graph"])["y_hat"]
        self.assertEqual(tuple(prediction.shape), (32, 5))
        self.assertTrue((prediction >= 0).all())
        expression_loss(prediction, slide["expression"])["loss"].backward()
        self.assertIsNotNone(model.global_context_to_film.weight.grad)
        self.assertTrue(torch.isfinite(model.global_context_to_film.weight.grad).all())

    def test_metrics_and_loss(self):
        metrics = expression_metrics(self.target, self.target)
        self.assertEqual(metrics["mse"], 0.0)
        self.assertAlmostEqual(metrics["pcc"], 1.0)
        target = torch.tensor(self.target)
        self.assertLess(abs(float(expression_loss(target, target)["loss"])), 1e-6)
        undefined = expression_metrics(np.ones((3, 2)), np.ones((3, 2)))
        self.assertIsNone(undefined["pcc"])
        json.dumps(undefined, allow_nan=False)

    def test_split_integrity(self):
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            validate_split({"train": ["train_section"], "val": ["train_section"],
                            "test": ["test_section"]}, self.root)
        coords = np.array([(x, y) for x in range(8) for y in range(8)])
        labels = spatial_labels(coords)
        for x in range(4):
            for y in range(4):
                block = (coords[:, 0] // 2 == x) & (coords[:, 1] // 2 == y)
                self.assertEqual(len(set(labels[block])), 1)
        manifest = make_split("DLPFC", "cross-patient", "Human1")
        self.assertFalse(set(manifest["train"]) & set(manifest["test"]))
        self.assertEqual(manifest["val"], ["151672", "151676"])

    def test_training_checkpoint_and_expression_free_prediction(self):
        config = {"model": {"hidden_dim": 8, "num_heads": 2, "num_programs": 3,
                            "local_depth": 1, "global_depth": 1, "dropout": 0.0},
                  "training": {"epochs": 2, "patience": 2,
                               "learning_rate": 1e-3, "weight_decay": 1e-4}}
        output = self.root / "run"
        with contextlib.redirect_stdout(io.StringIO()):
            train(config, self.root, self.embeddings, self.split, output, device="cpu")
        checkpoint = torch.load(output / "best.pt", weights_only=True)
        self.assertEqual(checkpoint["genes"], self.genes)
        self.assertTrue((output / "metrics.json").is_file())
        metrics = json.loads((output / "metrics.json").read_text())
        self.assertTrue(np.isfinite(metrics["val_mse"]))
        with np.load(output / "test_section.npz") as f:
            self.assertEqual(f["prediction"].shape, (32, 5))
        (self.root / "test_section" / "gene_expression_ground_truth.pkl").unlink()
        subprocess.run([sys.executable, "-m", "profuse_st.predict", "--checkpoint", str(output / "best.pt"),
                        "--data-root", str(self.root), "--slides", "test_section",
                        "--output", str(self.root / "prediction"), "--device", "cpu"],
                       check=True, capture_output=True, text=True)
        self.assertTrue((self.root / "prediction" / "test_section.npz").is_file())


if __name__ == "__main__":
    unittest.main()
