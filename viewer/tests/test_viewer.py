from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from viewer.adapters import discover_runs, load_manifests, safe_artifact_key
from viewer.app import create_app
from viewer.results import discover_runs as discover_output_runs


class ViewerFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.repo = Path(self.tempdir.name)
        self._write_robotwin_legacy()
        self._write_manifest()
        self._write_libero_trials()
        (self.repo / "secret.txt").write_text("do not serve")
        self.app = create_app(self.repo, testing=True)
        self.client = self.app.test_client()

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _write_robotwin_legacy(self) -> None:
        exp = self.repo / "exp" / "exp01a"
        results = exp / "results"
        results.mkdir(parents=True)
        (exp / "config.yaml").write_text(
            """runtime:\n  host: xdlab23_yang\n  gpu: 5\nrobotwin:\n  task: stack_blocks_two\n  embodiment: aloha-agilex\n  seed: 0\n"""
        )
        (results / "run_legacy.json").write_text(
            json.dumps(
                {
                    "run_id": "run_legacy",
                    "results": [
                        {
                            "seed": 0,
                            "task": "stack_blocks_two",
                            "embodiment": "aloha-agilex",
                            "plan_success": True,
                            "success": True,
                            "elapsed_seconds": 4.5,
                            "error": "",
                        }
                    ],
                }
            )
        )

    def _write_manifest(self) -> None:
        results = self.repo / "exp" / "exp01b" / "results"
        artifacts = results / "artifacts"
        artifacts.mkdir(parents=True)
        (artifacts / "episode-0.mp4").write_bytes(b"0123456789abcdef")
        (artifacts / "code.py").write_text("print('robotwin')\n")
        (results / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": "capx.result.v1",
                    "run_id": "exp01b-capx",
                    "benchmark": "robotwin",
                    "policy": "capx",
                    "status": "complete",
                    "artifact_root": "artifacts",
                    "config": {
                        "task": "stack_blocks_two",
                        "embodiment": "aloha-agilex",
                    },
                    "episodes": [
                        {
                            "episode_id": "seed-1",
                            "seed": 1,
                            "status": "complete",
                            "metrics": {
                                "code_execution_success": True,
                                "plan_success": True,
                                "task_success": False,
                                "reward": 0.25,
                                "elapsed_seconds": 8.0,
                            },
                            "artifacts": {
                                "video": "episode-0.mp4",
                                "generated_code": "code.py",
                                "summary": "../../../../secret.txt"
                            },
                        }
                    ],
                }
            )
        )

    def _write_libero_trials(self) -> None:
        run = (
            self.repo
            / "outputs"
            / "libero_spatial"
            / "pick_up_black_bowl"
            / "test-model"
            / "run"
        )
        success = run / "trial_01_sandboxrc_0_reward_1.000_taskcompleted_1"
        incomplete = run / "trial_02_sandboxrc_0_reward_0.000_taskcompleted_0"
        failed = run / "trial_03_sandboxrc_2_reward_0.000_taskcompleted_0"
        for directory in (success, incomplete, failed):
            directory.mkdir(parents=True)
            (directory / "code.py").write_text("print('libero')\n")
            (directory / "summary.txt").write_text(f"summary for {directory.name}\n")
        (success / "video_combined.mp4").write_bytes(b"libero-video")
        (failed / "summary.txt").write_text("Traceback: simulated failure\n")
        (run / "summaries.txt").write_text(
            "Config Path: env_configs/libero/franka_libero_spatial_0.yaml\n"
        )
        done = run / "aaa_done_flag"
        done.mkdir()
        (done / "aaa_done_flag.txt").write_text("1")

    def _runs(self):
        return discover_runs(self.repo)

    def test_discovers_manifest_legacy_and_libero(self) -> None:
        runs = self._runs()
        self.assertEqual(len(runs), 3)
        self.assertEqual({run["benchmark"] for run in runs}, {"robotwin", "libero"})
        self.assertEqual(
            {run["source"] for run in runs},
            {"manifest", "robotwin-legacy", "libero-trial-folders"},
        )

    def test_legacy_robotwin_mapping_preserves_oracle_semantics(self) -> None:
        run = next(run for run in self._runs() if run["source"] == "robotwin-legacy")
        self.assertEqual(run["policy"], "oracle")
        episode = run["episodes"][0]
        self.assertTrue(episode["metrics"]["plan_success"])
        self.assertTrue(episode["metrics"]["task_success"])
        self.assertIsNone(episode["metrics"]["code_execution_success"])
        self.assertIsNone(episode["artifacts"]["video"])

    def test_manifest_mapping_and_external_artifact_rejection(self) -> None:
        run = next(run for run in self._runs() if run["run_id"] == "exp01b-capx")
        episode = run["episodes"][0]
        self.assertFalse(episode["metrics"]["task_success"])
        self.assertEqual(run["metrics"]["task_success_rate"], 0.0)
        self.assertIsNotNone(episode["artifacts"]["video"])
        self.assertIsNone(episode["artifacts"]["summary"])
        self.assertNotIn(str(self.repo / "secret.txt"), run["_artifact_files"].values())

    def test_libero_keeps_code_execution_separate_from_task_success(self) -> None:
        run = next(run for run in self._runs() if run["benchmark"] == "libero")
        self.assertEqual(run["metrics"]["code_execution_rate"], 2 / 3)
        self.assertEqual(run["metrics"]["task_success_rate"], 1 / 3)
        second = next(item for item in run["episodes"] if item["episode_id"] == "trial-2")
        self.assertTrue(second["metrics"]["code_execution_success"])
        self.assertFalse(second["metrics"]["task_success"])
        third = next(item for item in run["episodes"] if item["episode_id"] == "trial-3")
        self.assertEqual(third["error"]["type"], "SandboxError")

    def test_exp01b_output_indexer_and_wrapper_manifest(self) -> None:
        normalized = discover_output_runs(self.repo / "outputs", repo_root=self.repo)
        self.assertEqual(len(normalized), 1)
        run = normalized[0].to_dict()
        self.assertEqual(run["benchmark"], "libero")
        self.assertEqual(run["policy"], "capx")
        self.assertEqual(run["metrics"]["task_success_rate"], 1 / 3)
        self.assertTrue(run["artifact_root"].startswith("outputs/"))
        first_video = run["episodes"][0]["artifacts"]["video"]
        self.assertTrue(first_video.startswith("outputs/"))

        destination = self.repo / "exp" / "exp01c" / "results" / "manifest.json"
        destination.parent.mkdir(parents=True)
        destination.write_text(json.dumps({"schema": "capx.result.v1", "runs": [run]}))
        loaded = load_manifests(destination, self.repo)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["run_id"], run["run_id"])
        episode = loaded[0]["episodes"][0]
        self.assertIsNotNone(episode["artifacts"]["video"])
        self.assertIn(episode["artifacts"]["video"], loaded[0]["_artifact_files"])

    def test_summary_and_filters(self) -> None:
        response = self.client.get("/api/summary")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["runs"], 3)
        self.assertEqual({item["benchmark"] for item in payload["benchmarks"]}, {"robotwin", "libero"})

        response = self.client.get("/api/runs?benchmark=libero&task=black_bowl")
        self.assertEqual(response.status_code, 200)
        runs = response.get_json()["runs"]
        self.assertEqual(len(runs), 1)
        self.assertNotIn("episodes", runs[0])

    def test_run_and_episode_detail_expose_same_origin_artifact_urls(self) -> None:
        response = self.client.get("/api/runs/exp01b-capx")
        self.assertEqual(response.status_code, 200)
        episode = response.get_json()["episodes"][0]
        self.assertEqual(episode["episode_id"], "seed-1")
        self.assertTrue(episode["artifacts"]["video"].startswith("/artifacts/exp01b-capx/"))

        response = self.client.get("/api/runs/exp01b-capx/episodes/seed-1")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["episode"]["metrics"]["reward"], 0.25)

    def test_video_supports_http_range(self) -> None:
        run = self.client.get("/api/runs/exp01b-capx").get_json()
        url = run["episodes"][0]["artifacts"]["video"]
        response = self.client.get(url, headers={"Range": "bytes=2-5"})
        try:
            self.assertEqual(response.status_code, 206)
            self.assertEqual(response.data, b"2345")
            self.assertEqual(response.headers["Content-Range"], "bytes 2-5/16")
        finally:
            response.close()

    def test_path_traversal_and_unregistered_files_are_rejected(self) -> None:
        self.assertFalse(safe_artifact_key("../secret.txt"))
        self.assertFalse(safe_artifact_key("folder/../../secret.txt"))
        self.assertFalse(safe_artifact_key("/etc/passwd"))
        for path in (
            "/artifacts/exp01b-capx/%2e%2e/secret.txt",
            "/artifacts/exp01b-capx/secret.txt",
            "/artifacts/exp01b-capx/%2Fetc%2Fpasswd",
        ):
            with self.subTest(path=path):
                response = self.client.get(path, follow_redirects=True)
                try:
                    self.assertEqual(response.status_code, 404)
                finally:
                    response.close()

    def test_page_and_static_assets_are_served_without_build_step(self) -> None:
        page = self.client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn(b"RobotTwin / LIBERO", page.data)
        javascript = self.client.get("/static/app.js")
        stylesheet = self.client.get("/static/style.css")
        try:
            self.assertEqual(javascript.status_code, 200)
            self.assertEqual(stylesheet.status_code, 200)
        finally:
            javascript.close()
            stylesheet.close()


if __name__ == "__main__":
    unittest.main()
