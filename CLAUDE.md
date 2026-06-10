# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A football (soccer) highlight generator. A CLIP-based classifier detects "shot on target / goal" moments frame-by-frame in match video, and ffmpeg stitches the detected moments into a highlight reel. The repo spans the full pipeline: data download → clip extraction → model training → inference → a Flask + React web app for end users.

## Repository layout & data flow

The pipeline is a linear chain across the top-level directories:

1. **`data_download.py`** — downloads SoccerNet match videos (`*_720p.mkv`) and `Labels-v2.json` annotations into a local directory (requires the SoccerNet password set in the script).
2. **`data_processing/label_parser.py`** — reads `Labels-v2.json`, keeps annotations labelled `shots on target` / `goal`, and uses ffmpeg to cut 5-second positive clips (`shot/`) and randomly-sampled negative clips (`non_shot/`) from each match half. Output tree: `data/clips/{train,valid}/{shot,non_shot}/<match>/<half>/`. **Note:** the `process_all_matches(...)` call runs at import time (module bottom), not under `if __name__ == "__main__"`.
3. **`data_processing/video_utils.py`** — `SoccerNetDataset` walks the clip tree, validates each video opens, and on `__getitem__` decodes the clip and returns **one random frame** (CLIP-preprocessed) plus its label. The model is effectively a single-frame image classifier; clips are just frame sources.
4. **`model/`** — model definitions + training + inference (see below).
5. **`app/`** — Flask backend + React frontend that wrap inference for end users.

## Models

Two parallel modelling approaches exist:

- **`model/model.py` → `FineTunedCLIP`** (the one actually used in production). CLIP `ViT-B/32` image encoder with the last `unfreeze_layers` transformer blocks unfrozen, feeding a 512→256→1 MLP head. Output is a single logit (use `sigmoid` for probability). Trained by **`model/train.py`**. This class is re-exported from `model/__init__.py`, so `from model import FineTunedCLIP` works — `model/` is both a package and a directory of scripts.
- **`model/model_tuned.py` → `ImprovedCLIPShotDetector` / `ZeroShotCLIPShotDetector`** (experimental). A multimodal variant that scores image features against hand-written shot / non-shot text prompts and fuses similarity diffs into the classifier. Trained by **`model/train-tuned.py`** via `from model.model_tuned import ...`.

**`model/inference.py` → `HighlightGenerator`** is the inference entry point used by the app. It loads a trained `FineTunedCLIP` checkpoint, samples one frame per second of input video, classifies each (threshold `0.85`), then `merge_shot_times` groups nearby positive seconds into ranges and `generate_highlight` cuts and concatenates them (separate ffmpeg video/audio concat) into the output mp4.

## Web app (`app/`)

- **`app/backend.py`** — Flask server on port `5000`. Upload is async: `/upload` saves the file and spawns a daemon thread (`process_video_async`) that runs `HighlightGenerator.detect_shots` → `generate_highlight`. Progress lives in the in-memory `processing_status` dict (lost on restart). Key routes: `POST /upload`, `GET /status/<id>`, `GET /download/<id>`, plus `/list`, `/stats`, `/health`. A background `threading.Timer` cleans uploads/outputs older than 24h.
- **`app/frontend/`** — Create React App (react-scripts 5). `src/App.js` polls `/status/<id>` and talks to the backend at `http://localhost:5000` (also has a `proxy` entry in `package.json`).

## Commands

There is **no `requirements.txt`**. Python deps must be installed manually: `torch`, OpenAI `clip` (`pip install git+https://github.com/openai/CLIP.git`), `opencv-python`, `ffmpeg-python`, `flask`, `flask-cors`, `SoccerNet`, `matplotlib`, `seaborn`, `scikit-learn`, `Pillow`. **`ffmpeg` and `ffprobe` must be on PATH.**

```powershell
# Train the production model (writes best_model_<timestamp>.pth + curve/ROC/confusion PNGs to cwd)
python model/train.py

# Run inference standalone (see __main__ block in inference.py)
python model/inference.py

# Regenerate training clips from downloaded SoccerNet matches
python data_processing/label_parser.py   # NOTE: runs on import via the module-bottom call

# Backend (from app/)
cd app; python backend.py                 # serves on 0.0.0.0:5000

# Frontend (from app/frontend/) — note the openssl-legacy-provider flag is required
cd app/frontend; npm install; npm start   # CRA dev server, proxies to :5000
npm run build                             # production build
```

## Important gotchas

- **Hardcoded absolute Windows paths everywhere.** Data lives at `E:\System Default\table\学习\大四下\paper\data\clips\...` (a sibling `paper\` tree, *not* this `paper code\` repo) and the trained checkpoint at `E:\System Default\table\学习\大四下\paper\model\best_model_20250406-201708.pth` (a 578 MB copy also sits in this repo's `model/`). `train.py`, `train-tuned.py`, `inference.py`, and `Video.py` all embed these literal `E:\...` paths. Expect to edit them before anything runs on another machine.
- **The `.pth` checkpoints (~578 MB each) are git-ignored** — they exceed GitHub's 100 MB file limit, so they are not in the remote. Obtain them out-of-band before running inference/training.
- Threshold/window constants for detection (`0.85`, 1-frame-per-second sampling, `clip_duration=5.0`) are tuned inline in `inference.py`.
