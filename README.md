# MediaPipe Webcam Pose Template

Small Python template for capturing webcam video, running MediaPipe landmark models, and exporting graph and CSV artifacts.

## Branch Note

This branch also includes the ElevenLabs test branch work.

## What It Outputs

By default, a run creates a timestamped folder under `outputs/` containing:

- `raw_capture.mp4`: the unmodified webcam clip.
- `annotated_capture.mp4`: the clip with MediaPipe pose landmarks drawn on top.
- `pose_landmarks.csv`: per-frame pose landmark coordinates and visibility.
- `hand_landmarks.csv`: per-frame 21-point hand landmarks for each detected hand.
- `pose_results.json`: structured pose landmarks for every processed frame.
- `pose_graph.png`: a 2D pose skeleton graph from the middle detected frame.
- `hand_graph.png`: a 2D hand skeleton graph from the middle detected hand frame.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The scripts use the current MediaPipe Tasks API and download default `.task` models into `models/` on first run.

## Run

Serve the SignNinja browser game with the live prediction backend:

```bash
python game_server.py
```

Then open http://127.0.0.1:8000. The page sends webcam frames to `/api/predict`
and advances the stream when the backend prediction matches the center target
letter. Pass a trained checkpoint with `--checkpoint path/to/model.pt`; without
one, the server still runs but predictions use random weights.

Capture a five-second clip from the default webcam:

```bash
python src/capture_holistic_pose.py --duration 5
```

Use a different camera or output location:

```bash
python src/capture_holistic_pose.py --camera 1 --duration 8 --output-dir outputs/test-run
```

List named cameras:

```bash
python src/capture_holistic_pose.py --list-cameras
```

Record webcam footage and holistic landmark CSV data:

```bash
python src/record_webcam.py
```

This writes `webcam_capture.mp4`, `holistic_landmarks.csv`, and `recording_metadata.json` into a timestamped `outputs/recording-*` directory. It samples the live feed every `0.2` seconds by default and writes CSV rows in real time as each sample is processed. Use `--debug-overlay` to save the output video with landmark markers rendered on each frame. Press `q` in the preview window to stop.

Change the sampling interval or run for a bounded test duration:

```bash
python src/record_webcam.py --sample-interval 0.1 --max-duration 5
```

## Notes

- Press `q` in the preview window to stop recording early.
- The scripts mirror the webcam preview by default. Use `--no-mirror` for raw camera orientation.
- Hand tracking exports all 21 landmarks per hand, including each finger joint and fingertip.
