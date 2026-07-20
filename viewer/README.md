# CaP-X Benchmark Viewer

Read-only Flask dashboard for normalized CaP-X results, the legacy RoboTwin
`run_*.json` format, and CaP-X/LIBERO `trial_*` folders. It has no frontend
build step.

## Run locally

From the repository root:

```bash
python viewer/app.py --host 127.0.0.1 --port 5001
```

Set `CAPX_REPO_ROOT` or pass `--repo-root` when artifacts live in a different
checkout.

On `xdlab23_yang`, use the already provisioned RoboTwin environment:

```bash
PYTHONNOUSERSITE=1 /data1/ybyang/envs/robotwin-capx/bin/python \
  viewer/app.py --host 127.0.0.1 --port 8891
```

Forward the server without exposing a public port:

```bash
ssh -N -L 8891:127.0.0.1:8891 xdlab23_yang
```

Then open <http://127.0.0.1:8891>.

## Artifact discovery

The viewer scans:

- `exp/**/manifest.json` and `outputs/**/manifest.json`;
- `exp/*/results/run_*.json` when that experiment has no manifest;
- `outputs/**/trial_*_sandboxrc_*_reward_*_taskcompleted_*`.

`viewer.results.discover_runs(...)` is the producer used by `exp01b/analyze.py`
to turn the last form into the portable `{schema, runs}` manifest wrapper.

Manifest artifact paths are relative to the manifest directory or its optional
`artifact_root`. Only declared, existing files inside the repository are
served. Relative paths containing `..`, absolute URL paths, undeclared files,
and files outside the repository are rejected.

## Tests

The test suite uses only `unittest` and Flask:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s viewer/tests -v
```
