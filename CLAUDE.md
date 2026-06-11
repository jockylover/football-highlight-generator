# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A football (soccer) highlight generator. A CLIP-based classifier detects "shot on target / goal" moments frame-by-frame in match video, and ffmpeg stitches the detected moments into a highlight reel. The repo spans the full pipeline: data download → clip extraction → model training → inference → a Flask + React web app for end users.

## Repository layout & data flow

The pipeline is a linear chain across the top-level directories:

1. **`data_download.py`** — downloads SoccerNet match videos (`*_720p.mkv`) and `Labels-v2.json` annotations (runs under `__main__`; target dir from `config.SOCCERNET_DIR`).
2. **`data_processing/label_parser.py`** — reads `Labels-v2.json`, keeps annotations labelled `shots on target` / `goal`, and uses ffmpeg to cut 5-second positive clips (`shot/`) and randomly-sampled negative clips (`non_shot/`) from each match half. Output tree: `data/clips/{train,valid,test}/{shot,non_shot}/<match>/<half>/`. The `process_all_matches(...)` call runs under `if __name__ == "__main__"`.
3. **`data_processing/video_utils.py`** — `SoccerNetDataset` walks the clip tree (with an optional `cache_path` manifest to skip re-validating every clip on startup), and on `__getitem__` **seeks to one random frame** (not full decode) returning the CLIP-preprocessed frame + label. The model is effectively a single-frame image classifier; clips are just frame sources.
4. **`model/`** — model definitions + training + inference + evaluation (see below).
5. **`app/`** — Flask backend + React frontend that wrap inference for end users.
6. **`config.py`** (repo root) — single source of truth for all data/model paths, env-overridable (`PAPER_ROOT`, `DATA_ROOT`, `MODEL_DIR`, `MODEL_PATH`, `SOCCERNET_DIR`). `clip_dirs(split)` returns the `(shot, non_shot)` dirs for a split. All scripts import from here instead of hardcoding `E:\...`.

## Models

Two parallel modelling approaches exist:

- **`model/model.py` → `FineTunedCLIP`** (the one actually used in production). CLIP `ViT-B/32` image encoder with the last `unfreeze_layers` transformer blocks unfrozen, feeding a 512→256→1 MLP head. Output is a single logit (use `sigmoid` for probability). Trained by **`model/train.py`**. This class is re-exported from `model/__init__.py`, so `from model import FineTunedCLIP` works — `model/` is both a package and a directory of scripts.
- **`model/model_tuned.py` → `ImprovedCLIPShotDetector` / `ZeroShotCLIPShotDetector`** (experimental). A multimodal variant that scores image features against hand-written shot / non-shot text prompts and fuses similarity diffs into the classifier. Trained by **`model/train-tuned.py`** via `from model.model_tuned import ...`.

**`model/inference.py` → `HighlightGenerator`** is the inference entry point used by the app. It loads a trained `FineTunedCLIP` checkpoint (`torch.load(..., map_location, weights_only=True)`), samples one frame per second of input video and classifies them **in batches** (`batch_size`, default 32), then `merge_shot_times` groups nearby positive seconds into ranges and `generate_highlight` cuts and concatenates them (separate ffmpeg video/audio concat) into the output mp4. The decision threshold defaults to `0.5` but is overridden by a `best_threshold.json` sitting next to the checkpoint (written by `train.py`).

**`model/evaluate.py`** — standalone test-split evaluation: loads the checkpoint, runs `SoccerNetDataset` over `clip_dirs("test")`, and reports precision/recall/F1, Average Precision, and a PR curve.

## Web app (`app/`)

- **`app/backend.py`** — Flask server on port `5000`. Upload is async: `/upload` saves the file and spawns a daemon thread (`process_video_async`) that runs `HighlightGenerator.detect_shots` → `generate_highlight`. The `HighlightGenerator` is a **module-level singleton** loaded once via `get_generator()` and reused across requests; GPU inference is serialized with `_inference_lock`. Progress lives in the in-memory `processing_status` dict (lost on restart). Key routes: `POST /upload`, `GET /status/<id>`, `GET /download/<id>`, plus `/list`, `/stats`, `/health`. A background `threading.Timer` cleans uploads/outputs older than 24h.
- **`app/frontend/`** — Create React App (react-scripts 5). `src/App.js` polls `/status/<id>` and talks to the backend at `http://localhost:5000` (also has a `proxy` entry in `package.json`).

## Commands

Install Python deps from `requirements.txt` (note: OpenAI `clip` installs from GitHub, and `ffmpeg`/`ffprobe` must be on PATH):

```powershell
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git   # if the clip line didn't resolve
```

All scripts read paths from `config.py`; to run on another machine, set `PAPER_ROOT` (or the finer-grained env vars) instead of editing source. Scripts assume the **repo root is on `sys.path`** (run from repo root, or via an IDE that marks it as a source root — `from model import ...`, `from config import ...`, `from data_processing... import ...` all rely on this).

```powershell
# Train the production model (writes best_model_<ts>.pth, best_threshold.json, curve/ROC/confusion PNGs to cwd)
python model/train.py

# Evaluate a checkpoint on the test split (precision/recall/F1/AP + PR curve)
python model/evaluate.py

# Run inference standalone (see __main__ block in inference.py)
python model/inference.py

# Regenerate training clips from downloaded SoccerNet matches
python data_processing/label_parser.py

# Backend
python app/backend.py                     # serves on 0.0.0.0:5000

# Frontend (from app/frontend/) — note the openssl-legacy-provider flag is required
cd app/frontend; npm install; npm start   # CRA dev server, proxies to :5000
npm run build                             # production build
```

## Important gotchas

- **Paths come from `config.py`, defaulting to `PAPER_ROOT = E:\System Default\table\学习\大四下\paper`** — a sibling `paper\` tree, *not* this `paper code\` repo. Data at `<PAPER_ROOT>\data\clips\...`, checkpoint at `<PAPER_ROOT>\model\best_model_20250406-201708.pth` (a 578 MB copy also sits in this repo's `model/`). Override via env vars rather than editing source.
- **The `.pth` checkpoints (~578 MB each) are git-ignored** — they exceed GitHub's 100 MB file limit, so they are not in the remote. Obtain them out-of-band before running inference/training.
- **The two trainers still duplicate a lot** (`train.py` vs `train-tuned.py`: training loop + plotting). Deferred refactor — extract a shared `training_utils` if touching both.
- `clip_duration=5.0` (clip length) and 1-frame-per-second sampling are still inline constants in `label_parser.py` / `inference.py`.
