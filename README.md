# animation

A procedurally rendered, Shinkai-style anime scenery montage, written in pure Python (NumPy / OpenCV / Numba).
Ten painted shots -- summer sky, railway crossing, city dusk, rain street, sakura, seaside sunset,
comet night, snow station, classroom, sea of clouds -- are rendered frame by frame, then cut to an
original synthesised score.

## Layout

| Path | What |
|---|---|
| `run.py` | Shot runner: renders a scene module to stills, motion strips, or video |
| `assemble.py` | Cuts the finished shots to the beat (96 BPM, 15 frames/beat @ 24 fps), synthesises the score, and encodes `out/montage.mp4` |
| `lib/` | Shared rendering toolkit: `core` (math, compositing, tonemapping, encode), `sky`, `fx` (optical effects), `clouds*` / `billow` (painted cloud engines) |
| `scenes/` | Scene modules. `sNN_<name>.py` are the shots; suffixed files are layers/variants; `_`-prefixed files are experiments |
| `out/` | Generated renders (git-ignored) |

Each scene module exposes `DURATION` (seconds) and `class Scene(W, H)` with `frame(t) -> float32 (H, W, 3)`
RGB in ~0..1 (HDR allowed before `finish()`). Scenes are resolution-independent.

## Requirements

- Python 3.11
- `numpy`, `opencv-python`, `numba`, `scipy`, `pillow`
- `ffmpeg` on `PATH` (with libx264)

```sh
pip install numpy opencv-python numba scipy pillow
```

## Usage

Render a shot:

```sh
python run.py scenes/s01_summer_sky.py --stills          # 960x540 stills + contact sheet -> out/previews/
python run.py scenes/s01_summer_sky.py --stills --full   # same at 1920x1080
python run.py scenes/s01_summer_sky.py --strip           # 6 consecutive frames (motion check)
python run.py scenes/s01_summer_sky.py --video --half    # half-res preview -> out/shots/<name>_half.mp4
python run.py scenes/s01_summer_sky.py --video           # final 1920x1080 -> out/shots/<name>.mp4
```

Assemble the montage (needs all ten final shots in `out/shots/`):

```sh
python assemble.py                 # writes out/score.wav and out/montage.mp4
python assemble.py --score         # only (re)synthesise the score
python assemble.py --reuse-score   # re-cut picture, keep existing score
```

The edit decision list (`EDL`) at the top of `assemble.py` controls shot order, in-points, lengths and transitions.
