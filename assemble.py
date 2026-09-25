"""Assemble the finished shots into the OP-style scenery montage with an original score.

    python assemble.py              # writes out/score.wav and out/montage.mp4
    python assemble.py --score      # only (re)synthesise the score
    python assemble.py --reuse-score

Round-2 edit: cut to the score's beat grid (96 BPM = exactly 15 frames per beat at 24 fps).
Round-3: three hero holds (s01 opening, s07 comet, s10 finale), everything else trimmed to
its highest-motion 3-5 beat window (lengths 8,4,5,3,5,3,8,3,5,8 beats).  Mostly hard cuts on
the beat, three short (6-10 frame) smoothstep dissolves centred on the beat, one short white
pop (3+8 frames) into the comet, the lamp-glare -> window-sun match cut s08 -> s09, and a hard
cut on the finale downbeat carrying a golden light leak (no full white) into s10.
Fade in from black; the finale blooms to white and fades to black.
Round-4: re-ordered into one time-of-day arc (day -> golden -> night -> dawn), lengths
8,4,5,4,3,5,3,4,8,8 beats.  Cuts on the beat, 2 short dissolves (8, 6 frames), a sun-glare
match cut s05 -> s09 capped at ~70%, a single white pop into the s07 chorus, a soft golden
leak into s10.  s10 gets an eased digital push-in + crane-up; the finale rises to a warm,
soft-kneed golden halation (never clipped), holds 4 frames and drops quickly to black through
amber (no grey plate) while the bell chord rings out.
Round-5: 15-bar score (post-chorus "breath" bar 11), lengths 8,4,5,4,5,5,5,8,4,8 beats; s08 moved
after the comet so s04/s08 are no longer adjacent; no dissolves; additive-only pop, leak and
finale halation (see EDL notes).

The picture is composited in float (sources decoded to 16-bit RGB), converted to BT.709
Y'CbCr with TPDF dither and piped to x264 as 4:2:0 -- no 8-bit re-quantisation of
gradients, so no banding/seams are introduced by the edit.
"""
import os
import subprocess
import sys
import wave

import cv2
import numpy as np
from scipy import signal

ROOT = os.path.dirname(os.path.abspath(__file__))
SHOTS = os.path.join(ROOT, 'out', 'shots')
OUT = os.path.join(ROOT, 'out', 'montage.mp4')
SCORE = os.path.join(ROOT, 'out', 'score.wav')
FPS = 24
W, H = 1920, 1080

BPM = 96.0
FPB = 15                     # frames per beat (60 / 96 * 24)
BEAT = 60.0 / BPM            # 0.625 s
BAR = 4 * BEAT               # 2.5 s

# Edit decision list.  (shot id, source in-frame, length in beats, transition INTO this shot)
# transition: ('cut',) | ('dissolve', frames) | ('flash', pre_frames, post_frames) | ('glare', pre, post)
#             | ('leak', pre, post): hard cut with a golden light-leak bloom, never above ~30% white
# Source in-points are the highest-motion window of each shot (see motion scan), except
# s02 (starts on the barrier arm coming down) and s03 (ends on the "いつか、あの空へ" billboard).
# Round-4 re-order: one deliberate time-of-day arc  noon -> afternoon -> sunset -> dusk -> night -> dawn
#   day:    s01 sky (hero) | s02 crossing | s05 sakura --sun glare--> s09 classroom (afternoon sun)
#   golden: s06 seaside sunset | s03 city dusk
#   night:  s04 rain | s08 snow --white pop--> s07 comet (hero, chorus)
#   dawn:   --golden light leak--> s10 sea of clouds (hero finale, slow digital push-in)
#
# Round-5 (re-rendered s01/s03/s05/s07/s09, reconceived s10 = summit torii at sunrise):
#   * s04 rain and s08 snow no longer back to back (both lateral trucks): s08 moves after the
#     comet chorus as a one-bar quiet "breath" and hands the lamp light to the dawn finale.
#   * s06 seaside 3 -> 5 beats.  Dissolves removed (no busy superimpositions): hard cuts on the beat
#     plus three optical transitions only (sun glare match, additive bloom pop, lamp -> sun leak).
#   * s07 pop is built from additive bloom + exposure push (no lerp to white = no milky grey veil).
#   * s10 opens on a saturated, readable frame: the leak is a small local warm glow near the sun,
#     additive and <= 0.20 at its peak, gone in half a second.  No digital push (the shot now has
#     its own crane/truck move).
# Round-6 (re-rendered s01/s03/s05/s07/s09, s10 torii summit re-rendered): windows re-picked by motion scan
#   (s01 crane 12-132, s05 crane 45-120 ending on the sun, s09 30-90, s03 50-125 ending on the billboard,
#   s07 tilt 6-126, s10 truck 0-144).  s06 5 -> 6 beats, s04 5 -> 4 (its peak-motion window).
#   Pop into s07 rebuilt as LOCAL additive bloom: sourced from the brightest neon sign (pre) and the comet
#   core (post), additive glow capped at 0.5, cleared from the landscape in 5 frames -- no exposure push,
#   no full-frame veil.  Glare-out centre re-measured on the new s05 sun.
# Round-7 (re-rendered s01/s03/s05/s06/s07/s08/s09/s10): windows re-picked by motion scan + contact sheets.
#   s01 now starts on frame 0 so the city strip holds under the crane-up before the tower takes the frame.
#   s06 6 -> 7 beats (src 15-120, the whole crane move), s03 5 -> 4 beats (66-126, its fastest stretch that
#   still lands on the billboard) so s03 cuts in on the bar-7 downbeat.
#   Transition anchors re-measured: s05 sun, s09 window sun, s07 comet head, s08 lamp head, s10 sun.
#   s04 and s08 (both slow leftward trucks) stay separated by the s07 comet chorus.
# Round-8 (re-rendered s01/s04/s05/s06/s07/s10): windows re-picked by motion scan (per-frame MAD + phase-corr
#   shift at 480x270) + contact sheets.  s01 is now 132 frames: 12-132 holds the whole ease-in/ease-out crane
#   (0-12 is static and was only under the fade-in).  s05 22-97 is the peak of its crane (the sun stays in frame,
#   glare-out re-measured on frame 96).  s04 31-91 (peak), s06 15-120 and s07 6-126 unchanged (still the peaks),
#   s10 0-144 (forced: the finale + fade needs all 144 frames).  Anchors re-measured: s05 sun, s04 red neon,
#   s07 comet head, s10 sun (in / end).
# Round-9 (re-rendered s01/s04/s05/s06/s07/s10): windows re-checked by the same motion scan -- all unchanged, each
#   is still the peak window of its render: s01 12-132 (0-12 static; the crane now keeps drifting to the cut, tail
#   MAD ~1.7, no dead hold), s04 31-91 (best 60-frame MAD 5.5), s05 22-97 (best 75), s06 15-120 (best 105),
#   s07 6-126 (best 120), s10 0-144 (forced).  Anchors re-measured on the new renders: s05 sun 0.170/0.295 (frame 96),
#   s07 comet head 0.629/0.627 (frame 6), s10 sun 0.615/0.394 (frame 0) -> 0.615/0.372 (frame 119) -- only the glare-out
#   centre moved.  The encode goes to montage.tmp.mp4 and replaces montage.mp4 only after ffmpeg exits 0 AND the temp
#   file probes to the expected frame count.
EDL = [
    ('s01_summer_sky',       12, 8, ('fadein', 16)),          # HERO  crane-up from the rooftops to the cumulus tower, 5.0 s
    ('s02_railway_crossing', 12, 4, ('cut',)),                # barrier arm coming down
    ('s05_sakura',           22, 5, ('cut',)),                # peak of the crane-up through the blossom, sun top-left
    ('s09_classroom',        30, 4, ('glare', 7, 10)),        # sakura sun glare -> window sun (match cut, capped ~70%)
    ('s06_seaside',          15, 7, ('cut',)),                # sunset road by the sea (7 beats: the longest non-hero hold)
    ('s03_city_dusk',        66, 4, ('cut',)),                # dusk skyline truck, ends on the billboard (cut on bar 7)
    ('s04_rain_street',      31, 4, ('cut',)),               # neon rain push (its peak-motion 4 beats)
    ('s07_comet_night',       6, 8, ('flash', 3, 5)),         # HERO  comet, chorus downbeat - local additive bloom pop
    ('s08_snow_station',     36, 4, ('cut',)),                # post-chorus breath; lamp light carries into the dawn
    ('s10_sea_of_clouds',     0, 8, ('leak', 5, 12)),         # HERO  finale: hard cut on the downbeat, local warm leak
]
END_WHITE = 2 * FPB     # last 2 beats of s10: sun bloom swells
END_HALO = 10           # ...and the last 10 frames rise to a warm golden halation (never clipped)
HALO_MAX = 0.20         # additive warm glow around the sun at the peak (torii + clouds still read through)
HALO_TINT = np.array([1.0, 0.78, 0.50], np.float32)   # warm gold sun bloom
END_HOLD = 0            # no held peak: the fade starts on the halation peak so the camera keeps moving to the last source frame
END_BLACK = 24          # 24-frame halation -> black, warming to amber as it darkens (camera still moving)
END_PAD = 17            # black while the reverb rings out (total length unchanged)
S10_SRC_FRAMES = 144

# glare centres (normalised x, y-of-height) for the s05 -> s09 match cut: sakura sun / window sun
GLARE_OUT = (0.170, 0.295)     # s05 sun (measured on the round-9 render, frame 96 = last frame of the window)
GLARE_IN = (0.426, 0.239)      # s09 window sun at its in-point
GLARE_CAP = 0.70        # max white contribution of the glare

# s08 lamp head (x, y-of-height) at the end of its window, and the s10 sun, for the dawn leak
LAMP_POS = (0.572, 0.070)   # measured on the round-7 s08 render, frame 95
S10_SUN = (0.615, 0.395)  # sun at the s10 in-point (measured on the round-8 render, frame 0)
S10_SUN_END = (0.615, 0.372)  # the sun has climbed by the end of the s10 window (frame 119: 0.374, 143: 0.369)
LEAK_MAX = 0.20         # max additive value of the s10 leak

# s04 -> s07 pop sources (x, y-of-height): the brightest red neon in s04 (round-8 render, frame 90),
# the comet head in s07 (round-8 render, frame 6)
SIGN_POS = (0.413, 0.297)
COMET_POS = (0.629, 0.626)
POP_MAX = 0.50          # cap of the additive pop glow


def timeline():
    """Start frame of every shot, content length and total length in frames."""
    starts, f = [], 0
    for _, _, beats, _ in EDL:
        starts.append(f)
        f += beats * FPB
    content = f
    total = content + END_HOLD + END_BLACK + END_PAD
    last_src = EDL[-1][1] + (content - starts[-1]) + END_HOLD + END_BLACK
    assert last_src <= S10_SRC_FRAMES, 's10 window runs past the end of the render'
    return starts, content, total


def probe_frames(path):
    """Decoded video frame count of a file (ffprobe -count_frames)."""
    out = subprocess.run(['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
                          '-show_entries', 'stream=nb_read_frames', '-of', 'csv=p=0', path],
                         capture_output=True, text=True, check=True).stdout.strip()
    return int(out.split(',')[0])


def check_windows(starts, content):
    """Every source window must lie inside its render (s10 also feeds the post-content fade)."""
    for i, (sid, src_in, beats, _) in enumerate(EDL):
        need = src_in + beats * FPB + (END_HOLD + END_BLACK if i == len(EDL) - 1 else 0)
        have = probe_frames(os.path.join(SHOTS, sid + '.mp4'))
        if need > have:
            raise RuntimeError(f'{sid}: window {src_in}-{need} runs past the render ({have} frames)')


# ----------------------------------------------------------------------------- score
SR = 48000


def mtof(m):
    return 440.0 * 2 ** ((m - 69) / 12.0)


class Mix:
    def __init__(self, dur):
        self.n = int(dur * SR) + SR * 4
        self.bus = {k: np.zeros((2, self.n)) for k in ('piano', 'pad', 'strings', 'lead', 'bass', 'fx')}

    def add(self, bus, t0, mono, pan=0.0, width=None):
        i0 = int(t0 * SR)
        if i0 >= self.n:
            return
        if mono.ndim == 1:
            L = mono * np.sqrt(0.5 * (1 - pan))
            R = mono * np.sqrt(0.5 * (1 + pan))
            st = np.stack([L, R])
        else:
            st = mono
        if i0 < 0:                      # events that start before t=0 (pre-roll swells)
            st, i0 = st[:, -i0:], 0
        m = min(st.shape[1], self.n - i0)
        self.bus[bus][:, i0:i0 + m] += st[:, :m]


RNG = np.random.default_rng(7)


def piano_note(midi, vel, ring):
    """Additive piano-ish tone: stretched inharmonic partials, per-partial decay, damper release."""
    f0 = mtof(midi)
    dur = ring + 0.6
    t = np.arange(int(dur * SR)) / SR
    B = 0.00035 * (f0 / 261.6) ** 0.5
    tau0 = np.clip(4.2 * (261.6 / f0) ** 0.55, 0.9, 7.0)
    bright = 0.55 + 0.6 * vel
    out = np.zeros_like(t)
    for n in range(1, 15):
        fn = f0 * n * np.sqrt(1 + B * n * n)
        if fn > 16000:
            break
        a = (1.0 / n ** 1.25) * bright ** (n - 1) * (1.0 if n > 1 else 1.0)
        if n == 2:
            a *= 1.1
        tau = tau0 / (1 + 0.45 * (n - 1) * (fn / 2000 + 0.6))
        # double decay: fast prompt sound + slow aftersound
        env = 0.7 * np.exp(-t / (tau * 0.25)) + 0.3 * np.exp(-t / tau)
        det = 1 + RNG.uniform(-0.6, 0.6) * 1e-3 / 1.2
        ph = RNG.uniform(0, 2 * np.pi)
        out += a * env * (np.sin(2 * np.pi * fn * t + ph) + 0.6 * np.sin(2 * np.pi * fn * det * 1.0007 * t + ph * 0.7))
    # hammer: a tiny filtered noise tick
    nk = int(0.012 * SR)
    tick = RNG.standard_normal(nk) * np.exp(-np.arange(nk) / (0.0025 * SR))
    b, a_ = signal.butter(2, [min(f0 * 2, 3000) / (SR / 2), min(f0 * 8, 12000) / (SR / 2)], 'band')
    out[:nk] += 0.05 * vel * signal.lfilter(b, a_, tick)
    atk = np.minimum(1, t / 0.003)
    rel = np.where(t < ring, 1.0, np.exp(-(t - ring) / 0.18))
    out *= atk * rel * vel
    return out * 0.18


def saw_voice(f, dur, detunes=(-6, 0, 6), vib=0.0035, n_harm=24, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * SR)) / SR
    out = np.zeros_like(t)
    for d in detunes:
        fd = f * 2 ** (d / 1200)
        vr = rng.uniform(4.6, 5.6)
        lfo = 1 + vib * np.sin(2 * np.pi * vr * t + rng.uniform(0, 6.28)) * np.minimum(1, t / 1.2)
        phase = 2 * np.pi * np.cumsum(fd * lfo) / SR + rng.uniform(0, 6.28)
        for n in range(1, n_harm + 1):
            if fd * n > 12000:
                break
            out += np.sin(n * phase) / n
    return out / len(detunes)


def swell_env(dur, atk, rel, shape=None):
    t = np.arange(int(dur * SR)) / SR
    e = np.minimum(1, t / atk) ** 1.5 * np.clip((dur - t) / rel, 0, 1) ** 1.2
    if shape is not None:
        e *= shape(t)
    return e


def lowpass(x, fc, order=2):
    b, a = signal.butter(order, fc / (SR / 2))
    return signal.lfilter(b, a, x, axis=-1)


def highpass(x, fc, order=2):
    b, a = signal.butter(order, fc / (SR / 2), 'high')
    return signal.lfilter(b, a, x, axis=-1)


def reverb_ir(rt60=2.8, pre=0.022, seed=3):
    n = int((rt60 + 0.3) * SR)
    t = np.arange(n) / SR
    rng = np.random.default_rng(seed)
    ir = np.zeros((2, n))
    for c in range(2):
        noise = rng.standard_normal(n)
        # progressively darker tail: blend a bright and a dark decaying noise
        bright = noise * np.exp(-6.9 * t / (rt60 * 0.55))
        dark = lowpass(rng.standard_normal(n), 2500, 2) * np.exp(-6.9 * t / rt60)
        ir[c] = 0.5 * bright + 1.2 * dark
        # early reflections
        for k in range(10):
            d = int(rng.uniform(0.006, 0.07) * SR)
            ir[c, d] += rng.uniform(-0.8, 0.8) * (1 - k / 12)
        ir[c] *= np.minimum(1, t / 0.01)
    p = int(pre * SR)
    ir = np.concatenate([np.zeros((2, p)), ir], axis=1)
    ir = highpass(ir, 120)
    return ir / np.sqrt(np.sum(ir ** 2) / 2)


def k_weight_lufs(x):
    """Approximate integrated loudness (BS.1770 K-weighting at 48 kHz, 400ms blocks, abs+rel gate)."""
    b1 = [1.53512485958697, -2.69169618940638, 1.19839281085285]
    a1 = [1.0, -1.69065929318241, 0.73248077421585]
    b2 = [1.0, -2.0, 1.0]
    a2 = [1.0, -1.99004745483398, 0.99007225036621]
    y = signal.lfilter(b2, a2, signal.lfilter(b1, a1, x, axis=-1), axis=-1)
    blk, hop = int(0.4 * SR), int(0.1 * SR)
    ms = []
    for i in range(0, y.shape[1] - blk, hop):
        ms.append(np.sum(np.mean(y[:, i:i + blk] ** 2, axis=1)))
    ms = np.array(ms)
    L = -0.691 + 10 * np.log10(ms + 1e-12)
    ms = ms[L > -70]
    rel = -0.691 + 10 * np.log10(ms.mean()) - 10
    ms = ms[(-0.691 + 10 * np.log10(ms)) > rel]
    return -0.691 + 10 * np.log10(ms.mean())



# D major.  chord: (root pitch class, quality)
CH = {'I': (2, 'maj'), 'ii': (4, 'min'), 'iii': (6, 'min'), 'IV': (7, 'maj'), 'V': (9, 'maj'),
      'Vsus': (9, 'sus'), 'vi': (11, 'min')}
THIRD = {'maj': 4, 'min': 3, 'sus': 5}

# One bar = 4 beats = 2.5 s = 60 frames, bar 0 downbeat at t = 0.
#   bars 0-1  intro         s01 hero (beats 0-7)
#   bars 2-5  verse (day)   s02 b8, s05 b12, s09 b17 (glare swell), s06 b21
#   bars 6-8  build (dusk/night) s03 b28, s04 b32; strings creep in, riser through bar 8
#   bar  9    FLASH         -> chorus bars 9-10: s07 hero (b36)
#   bar  11   BREATH        post-chorus bar, dynamics drop back: s08 snow (b44)
#   bar  12   LEAK          -> finale bars 12-13: s10 hero (b48), IV-V | I, golden halation peak on bar 14
BARS = [
    ['IV'], ['Vsus', 'V'],
    ['I'], ['V'], ['vi'], ['iii'],
    ['IV'], ['iii', 'vi'], ['ii', 'Vsus'],
    ['IV'], ['V', 'vi'],
    ['ii', 'Vsus'],
    ['IV', 'V'], ['I'], ['I'],
]
FLASH_BARS = (9, 12)
BREATH_BAR = 11
STRING_CREEP = {6: 0.07, 7: 0.12, 8: 0.18, BREATH_BAR: 0.20}   # strings under the build and the breath
RESOLVE_BAR = 13          # tonic arrival, arpeggio slows to quarters and rings out
END_BAR = 14              # halation peak

# melody: (bar, beat, midi, beats)
MELODY = [
    # quiet motif in the verse (bars 4-5)
    (4, 0, 78, 1.5), (4, 1.5, 76, 0.5), (4, 2, 74, 2),
    (5, 0, 73, 1), (5, 1, 76, 1), (5, 2, 78, 2),
    # build (bars 6-8)
    (6, 0, 79, 1.5), (6, 1.5, 81, 0.5), (6, 2, 83, 2),
    (7, 0, 81, 1), (7, 1, 78, 1), (7, 2, 78, 1.5), (7, 3.5, 76, 0.5),
    (8, 0, 79, 1.5), (8, 1.5, 78, 0.5), (8, 2, 76, 1.5), (8, 3.5, 81, 0.5),
    # chorus (bars 9-10)
    (9, 0, 83, 1), (9, 1, 81, 0.5), (9, 1.5, 83, 0.5), (9, 2, 86, 1.5), (9, 3.5, 83, 0.5),
    (10, 0, 81, 1), (10, 1, 83, 1), (10, 2, 78, 1.5), (10, 3.5, 85, 0.5),
    # breath: a soft answer to the chorus
    (11, 0, 81, 1.5), (11, 1.5, 79, 0.5), (11, 2, 76, 1.5), (11, 3.5, 81, 0.5),
    # finale
    (12, 0, 86, 1.5), (12, 1.5, 85, 0.5), (12, 2, 85, 1), (12, 3, 88, 1),
    (13, 0, 86, 6),
]


def chord_tones(name):
    pc, q = CH[name]
    return pc, THIRD[q]


def section_level(bar):
    if bar < 2:
        return 0
    if bar < FLASH_BARS[0] or bar == BREATH_BAR:
        return 1
    if bar < FLASH_BARS[1]:
        return 2
    return 3


def synth_score(total, cut_beats, glare_beats=()):
    """total: seconds.  cut_beats: beat index of every picture cut (for accents)."""
    t0 = 0.0
    beat = BEAT
    bar = BAR
    print(f'score: {BPM:.0f} BPM, bar {bar:.3f}s, flashes on bars {FLASH_BARS} '
          f'({FLASH_BARS[0] * bar:.2f}s, {FLASH_BARS[1] * bar:.2f}s), white peak bar {END_BAR} ({END_BAR * bar:.2f}s)')

    mix = Mix(total)

    def T(b, bt=0.0):
        return t0 + b * bar + bt * beat

    cut_times = {round(c * beat, 4) for c in cut_beats}

    halves = []
    for b, chs in enumerate(BARS):
        if len(chs) == 1:
            halves.append((b, 0, 4, chs[0]))
        else:
            halves.append((b, 0, 2, chs[0]))
            halves.append((b, 2, 2, chs[1]))

    ring_final = T(END_BAR, 3)
    for (b, bt, nb, name) in halves:
        pc, th = chord_tones(name)
        sec = section_level(b)
        root3 = 48 + pc if pc < 7 else 36 + pc      # root between C3..F#3 / G2..B2
        root3 = root3 if root3 >= 43 else root3 + 12
        # ------------- piano arpeggio (8ths, sustain pedal per chord)
        if b <= RESOLVE_BAR:
            pattern = [0, 7, 12, 12 + th, 19, 12 + th, 12, 7]
            if sec >= 2:
                pattern = [0, 7, 12, 12 + th, 19, 24, 19, 12 + th]
            final = b == RESOLVE_BAR
            step = 1.0 if final else 0.5
            nsteps = int(nb / step)
            ring_end = ring_final if final else T(b, bt + nb)
            base_vel = [0.40, 0.52, 0.62, 0.66][sec]
            if b < 2:
                base_vel *= 0.78 + 0.12 * b
            for k in range(nsteps):
                st = bt + k * step
                m = root3 + (pattern[(int(st / 0.5)) % 8] if not final else [0, 7, 12, 12 + th][k])
                v = base_vel * (1.0 if k % 4 == 0 else 0.78 if k % 2 == 0 else 0.66)
                ts_nom = T(b, st)
                if round(ts_nom, 4) in cut_times:        # lean on the note under every picture cut
                    v *= 1.22
                v *= 1 + 0.05 * RNG.standard_normal()
                ts = ts_nom + RNG.normal(0, 0.004)
                ring = max(0.25, ring_end - ts + 0.05)
                pan = np.clip((m - 57) / 26, -0.45, 0.45)
                mix.add('piano', ts, piano_note(m, v, ring), pan)
                if sec == 3 and b == FLASH_BARS[1] and k % 2 == 1:     # sparkle octave in the finale
                    mix.add('piano', ts, piano_note(m + 24, v * 0.45, ring), -pan)
            # left-hand octave on the chord change
            lh_vel = [0.33, 0.45, 0.55, 0.62][sec]
            mix.add('piano', T(b, bt), piano_note(root3 - 12, lh_vel, ring_end - T(b, bt)), -0.08)
            if sec >= 2 and root3 - 24 >= 26:
                mix.add('piano', T(b, bt) + 0.01, piano_note(root3 - 24, lh_vel * 0.45, ring_end - T(b, bt)), -0.05)

        # ------------- pad (warm, lowpassed, wide) : every chord
        if b > RESOLVE_BAR:
            continue
        start, dur = T(b, bt), nb * beat
        tail = 1.4
        if b == RESOLVE_BAR:
            dur = ring_final - start
        voices = sorted([(pc + o) % 12 for o in (0, th, 7)])
        pad_notes = [55 + ((p - 55) % 12) for p in voices]
        lvl = [0.10, 0.20, 0.30, 0.36][sec]
        if b < 2:
            lvl *= (b + 1) / 2.5
        for i, m in enumerate(pad_notes):
            x = saw_voice(mtof(m), dur + tail, seed=b * 17 + i + bt)
            x = lowpass(x, [900, 1300, 1900, 2400][sec], 2)
            e = swell_env(dur + tail, atk=0.45 * nb / 4 + 0.25, rel=tail + 0.2)
            mix.add('pad', start - 0.12, x * e * lvl * 0.35, pan=(-0.4, 0.4, 0.0)[i])

        # ------------- strings (high, bright, from the chorus; creeping in during bars 5-6) + cello
        if sec >= 2 or b in STRING_CREEP:
            sl = STRING_CREEP.get(b, [0, 0, 0.26, 0.34][sec])
            str_notes = [67 + ((p - 67) % 12) for p in voices]
            for i, m in enumerate(str_notes):
                x = saw_voice(mtof(m), dur + tail, detunes=(-9, -3, 3, 9), vib=0.0045, seed=500 + b * 7 + i)
                x = highpass(lowpass(x, 3200, 2), 250)
                e = swell_env(dur + tail, atk=0.7 if nb == 4 else 0.35, rel=tail + 0.3,
                              shape=lambda t, d=dur: 0.75 + 0.25 * np.sin(np.pi * np.clip(t / d, 0, 1)))
                mix.add('strings', start - 0.05, x * e * sl * 0.3, pan=(-0.5, 0.0, 0.5)[i])
            cm = root3 - 12 if root3 - 12 >= 36 else root3
            x = saw_voice(mtof(cm), dur + tail, detunes=(-4, 4), vib=0.004, seed=900 + b)
            x = lowpass(x, 1100, 2)
            mix.add('strings', start - 0.05, x * swell_env(dur + tail, 0.4, tail + 0.3) * sl * 0.26, 0.1)

        # ------------- bass (soft sine+2nd harmonic), from the verse
        if sec >= 1:
            f = mtof(root3 - 12 if root3 - 12 >= 31 else root3)
            d = dur + 0.8
            tt = np.arange(int(d * SR)) / SR
            x = np.sin(2 * np.pi * f * tt) + 0.25 * np.sin(4 * np.pi * f * tt) + 0.08 * np.sin(6 * np.pi * f * tt)
            e = np.minimum(1, tt / 0.03) * np.exp(-tt / 3.5) * np.clip((d - tt) / 0.8, 0, 1)
            mix.add('bass', start, x * e * [0, 0.055, 0.07, 0.08][sec])

    # ------------- lead melody: soft bell-piano doubled by a flute-like sine in the chorus/finale
    for (b, bt, m, nb_) in MELODY:
        ts = T(b, bt)
        dur = nb_ * beat
        vel = 0.62 if b < FLASH_BARS[0] or b == BREATH_BAR else 0.78
        mix.add('lead', ts, piano_note(m, vel, dur + 0.05), 0.05)
        if b >= FLASH_BARS[0] and b != BREATH_BAR:
            tt = np.arange(int((dur + 0.5) * SR)) / SR
            vib = 1 + 0.004 * np.sin(2 * np.pi * 5.2 * tt) * np.clip((tt - 0.25) / 0.4, 0, 1)
            ph = 2 * np.pi * np.cumsum(mtof(m) * vib) / SR
            x = np.sin(ph) + 0.18 * np.sin(2 * ph) + 0.05 * np.sin(3 * ph)
            breath = lowpass(RNG.standard_normal(len(tt)), 4000) * 0.02
            e = np.minimum(1, tt / 0.07) * np.clip((dur + 0.25 - tt) / 0.35, 0, 1)
            mix.add('lead', ts, (x + breath) * e * 0.05, -0.05)

    # ------------- off-bar cuts get a soft high "glint" (chord fifth, two octaves up)
    for c in cut_beats:
        if c % 4 == 0 or c == 0:
            continue
        b = int(c // 4)
        name = BARS[b][0] if len(BARS[b]) == 1 or (c % 4) < 2 else BARS[b][1]
        pc, _ = chord_tones(name)
        m = 86 + ((pc + 7 - 86) % 12)
        mix.add('lead', T(0, c), piano_note(m, 0.38, 1.2), 0.3)
        if m + 12 <= 100:
            mix.add('lead', T(0, c) + 0.004, piano_note(m + 12, 0.22, 1.0), -0.3)

    # ------------- risers into each flash + a soft cymbal bloom + sub "whoomp" on the flash downbeat
    def riser(t_end, L, lvl):
        n = int(L * SR)
        tt = np.arange(n) / SR
        noise = RNG.standard_normal((2, n))
        lo = highpass(lowpass(noise, 2500), 400)
        hi = highpass(noise, 3500)
        k = (tt / L) ** 2
        mix.add('fx', t_end - L, (lo * (1 - k) + hi * k) * (tt / L) ** 2.6 * lvl)

    def bloom(t, lvl):
        n2 = int(3.5 * SR)
        t2 = np.arange(n2) / SR
        cym = highpass(RNG.standard_normal((2, n2)), 4500)
        mix.add('fx', t, cym * (np.exp(-t2 / 1.1) * np.minimum(1, t2 / 0.004)) * 0.05 * lvl)
        sub = np.sin(2 * np.pi * (46 + 20 * np.exp(-t2 / 0.08)) * t2) * np.exp(-t2 / 0.7) * np.minimum(1, t2 / 0.01)
        mix.add('fx', t, sub * 0.05 * lvl)

    for fb in FLASH_BARS:
        riser(T(fb), bar, 0.035)
        bloom(T(fb), 1.0)
    # the glare match-cut gets a short airy swell (no hit)
    for c in glare_beats:
        riser(T(0, c), 0.35, 0.018)
    # final white peak: long gentle swell + a quiet bell chord (D6 F#6 A6 D7)
    riser(T(END_BAR), 2 * beat, 0.02)
    bloom(T(END_BAR), 0.45)
    for i, m in enumerate((86, 90, 93, 98)):
        mix.add('lead', T(END_BAR) + 0.012 * i, piano_note(m, 0.34 - 0.04 * i, 2.5), (-0.3, 0.1, 0.3, -0.1)[i])

    # ------------- mixdown + reverb
    ir = reverb_ir()
    sends = {'piano': 0.32, 'pad': 0.45, 'strings': 0.40, 'lead': 0.36, 'bass': 0.06, 'fx': 0.5}
    dry = np.zeros((2, mix.n))
    send = np.zeros((2, mix.n))
    for k, x in mix.bus.items():
        dry += x
        send += x * sends[k]
    wet = np.stack([signal.fftconvolve(send[0], ir[0])[:mix.n], signal.fftconvolve(send[1], ir[1])[:mix.n]])
    out = dry * 0.85 + wet * 0.55
    out = highpass(out, 38, 2)
    out = out - 0.25 * lowpass(out, 120) + 0.35 * highpass(out, 2500) + 0.10 * highpass(out, 8000)

    n = int(round(total * SR))
    out = out[:, :n]
    tt = np.arange(n) / SR
    fade_in = np.clip(tt / 0.4, 0, 1) ** 2
    fade_out = np.clip((total - tt) / 1.1, 0, 1) ** 1.6      # bell + reverb ring out over the black, then fade
    out *= fade_in * fade_out

    r = np.sqrt(np.mean(out ** 2, axis=1))
    out *= (np.sqrt(r.prod()) / r)[:, None]

    lufs = k_weight_lufs(out)
    out *= 10 ** ((-16.0 - lufs) / 20)
    ceil = 10 ** (-1.0 / 20)
    knee = 0.7 * ceil
    a = np.abs(out)
    out = np.sign(out) * np.where(a > knee, knee + (ceil - knee) * np.tanh((a - knee) / (ceil - knee)), a)
    print(f'score loudness: {lufs:.1f} -> {k_weight_lufs(out):.1f} LUFS, peak {20 * np.log10(np.abs(out).max()):.2f} dBFS')

    pcm = (np.clip(out.T, -1, 1) * 32767).astype('<i2')
    with wave.open(SCORE, 'wb') as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return SCORE


# ----------------------------------------------------------------------------- picture
def ensure_shots():
    for sid, *_ in EDL:
        p = os.path.join(SHOTS, sid + '.mp4')
        if not os.path.exists(p):
            print('rendering missing shot', sid)
            subprocess.check_call([sys.executable, 'run.py', f'scenes/{sid}.py', '--video'], cwd=ROOT)


class Reader:
    """Sequential 16-bit RGB frame reader for one shot (sources are untagged BT.601 limited range,
    which is what lib.core.VideoWriter's rgb24 -> yuv420p conversion produced)."""

    def __init__(self, sid, first):
        self.sid, self.next = sid, first
        vf = (f'select=gte(n\\,{first}),'
              'scale=in_color_matrix=bt601:in_range=tv:flags=accurate_rnd+full_chroma_int+bicubic,format=rgb48le')
        self.p = subprocess.Popen(['ffmpeg', '-v', 'error', '-i', os.path.join(SHOTS, sid + '.mp4'), '-vf', vf,
                                   '-fps_mode', 'passthrough', '-f', 'rawvideo', '-'],
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=W * H * 6 * 2)

    def read(self):
        b = self.p.stdout.read(W * H * 6)
        if len(b) < W * H * 6:
            raise RuntimeError(f'{self.sid}: ran out of frames at {self.next}')
        self.next += 1
        return np.frombuffer(b, '<u2').reshape(H, W, 3).astype(np.float32) * (1.0 / 65535.0)

    def close(self):
        self.p.stdout.close()
        self.p.kill()
        self.p.wait()


class Sources:
    def __init__(self):
        self.r = {}

    def get(self, i, n):
        sid, src_in = EDL[i][0], EDL[i][1]
        idx = src_in + n
        r = self.r.get(i)
        if r is None or r.next > idx:
            if r is not None:
                r.close()
            r = self.r[i] = Reader(sid, idx)
        while r.next < idx:
            r.read()
        return r.read()

    def release_before(self, i):
        for k in list(self.r):
            if k < i:
                self.r.pop(k).close()

    def close(self):
        for r in self.r.values():
            r.close()


# normalised coordinate grids (x in 0..1, y in 0..H/W) for glare / light-leak shapes
_YY, _XX = np.mgrid[0:H, 0:W].astype(np.float32)
_XX /= W
_YY /= W


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def soft_blur(img, sigma_px):
    """Wide, band-free blur: downsample 8x, blur, upsample."""
    small = cv2.resize(img, (W // 8, H // 8), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma_px / 8)
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)


def bloom(img, gain, thresh=0.6, sigma=60):
    lum = img @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    hl = img * (np.clip((lum - thresh) / (1 - thresh), 0, 1) ** 1.5)[..., None]
    return img + gain * (0.6 * soft_blur(hl, sigma) + 0.4 * soft_blur(hl, sigma * 3))


def screen(img, layer):
    return 1 - (1 - img) * (1 - np.clip(layer, 0, 1))


FLASH_TINT = np.array([1.0, 0.985, 0.96], np.float32)


def whiteout(img, a, tint=FLASH_TINT):
    return img * (1 - a) + tint * a


def leak(center, radius, color, amount):
    cx, cy = center
    g = np.exp(-(((_XX - cx) ** 2 + (_YY - cy) ** 2) / (2 * radius ** 2)))
    return g[..., None] * (np.array(color, np.float32) * amount)


def glare(img, center, p, color=(1.0, 0.93, 0.80)):
    """Lamp/sun glare that swells to (nearly) fill frame as p -> 1."""
    cx, cy = center[0], center[1] * H / W
    r = np.sqrt((_XX - cx) ** 2 + (_YY - cy) ** 2)
    sigma = 0.02 + 0.38 * p ** 2         # glare stays centred on the sun, not a uniform veil
    g = np.exp(-(r / sigma) ** 1.5) * (0.5 + 0.7 * p)
    streak = np.exp(-((_YY - cy) / (0.004 + 0.02 * p)) ** 2) * np.exp(-((_XX - cx) / (0.25 + 0.6 * p)) ** 2) * 0.5 * p
    col = np.array(color, np.float32)
    # capped: the layer tops out at GLARE_CAP so the match-cut picture always reads through it
    lay = GLARE_CAP * (1 - np.exp(-1.6 * (g + streak)))
    out = screen(bloom(img, 0.5 * p), lay[..., None] * col)
    return knee(out, 0.82, 0.965)


def add_glow(img, center, radius, color, amount):
    """Additive local glow; center is (x, y-of-height)."""
    return img + leak((center[0], center[1] * H / W), radius, color, amount)


def exposure_pop(img, e, b, thresh):
    """White pop built additively: exposure push (blacks stay black, hues kept) + highlight bloom,
    rolled off by a soft shoulder instead of lerping the frame toward a white plate."""
    return knee(bloom(img * (1.0 + e), b, thresh=thresh, sigma=55), 0.80, 0.985)


def knee(img, k0=0.84, top=0.955):
    """Soft shoulder: values above k0 roll off smoothly toward `top` (no flat clipped plateau)."""
    r = top - k0
    return np.where(img > k0, k0 + r * np.tanh((img - k0) / r), img).astype(np.float32)


def finale_halo(img, pb, ph, fade=None):
    """Finale grade.  pb 0..1: gentle sun bloom over the last 2 beats (soft-kneed, never clipped).
    ph 0..1: warm golden halation built additively (exposure lift + wide sun bloom + a warm glow
    around the sun) -- no lerp toward a white plate, so the silhouettes stay dark and saturated."""
    img = knee(bloom(img, 0.5 * pb ** 1.5, thresh=0.6))
    if ph <= 0:
        return img
    # capped: a warm bloom around the sun only (high threshold, small exposure lift) so the torii and the
    # near clouds keep their silhouette contrast under the glow
    img = bloom(img * (1.0 + 0.08 * ph), 0.3 * ph, thresh=0.75, sigma=60)
    img = add_glow(img, S10_SUN_END, 0.16, HALO_TINT, HALO_MAX * ph ** 1.4)
    if fade is not None:           # darken before the shoulder: the sun core sinks through amber, not khaki
        img = img * fade
    return knee(img, 0.80, 0.97)


def s10_frame(src, f, starts, content):
    i = len(EDL) - 1
    n = f - starts[i]
    return knee(src.get(i, n))


def compose(src, f, starts, content):
    """Return the float RGB picture for output frame f."""
    if f >= content:
        k = f - content
        if k >= END_HOLD + END_BLACK:
            return np.zeros((H, W, 3), np.float32)
        if k < END_HOLD:
            return finale_halo(s10_frame(src, f, starts, content), 1.0, 1.0)
        k -= END_HOLD
        v = 1.0 - smoothstep((k + 1) / END_BLACK)
        # darken toward amber rather than grey: blue drops first, red last
        return finale_halo(s10_frame(src, f, starts, content), 1.0, 1.0,
                           fade=np.array([v ** 1.0, v ** 1.25, v ** 1.6], np.float32))

    i = max(j for j, s in enumerate(starts) if s <= f)
    img = s10_frame(src, f, starts, content) if i == len(EDL) - 1 else src.get(i, f - starts[i])
    tr_in = EDL[i][3]

    # --- transition into shot i (we are at/after its cut)
    if tr_in[0] == 'dissolve':
        n = tr_in[1]
        k = f - starts[i]
        if k < n // 2:
            a = smoothstep((k + n // 2 + 0.5) / n)
            prev = src.get(i - 1, starts[i] - starts[i - 1] + k)
            img = prev * (1 - a) + img * a
    elif tr_in[0] == 'flash':
        post = tr_in[2]
        k = f - starts[i]
        if k < post:
            q = (1 - k / post) ** 1.6      # gone from the landscape within `post` frames
            img = bloom(img, 0.9 * q, thresh=0.55, sigma=50)
            img = add_glow(img, COMET_POS, 0.10, (0.80, 0.92, 1.0), POP_MAX * q)
            img = add_glow(img, COMET_POS, 0.035, (1.0, 1.0, 1.0), 0.35 * q)
            img = knee(img, 0.80, 0.97)
    elif tr_in[0] == 'leak':
        post = tr_in[2]
        k = f - starts[i]
        if k < post:
            q = (1 - k / post) ** 1.3
            lay = leak((S10_SUN[0], S10_SUN[1] * H / W), 0.16, (1.0, 0.80, 0.52), 1.0)
            lay += leak((S10_SUN[0] + 0.05, S10_SUN[1] * H / W - 0.02), 0.07, (1.0, 0.90, 0.72), 0.5)
            img = img + np.minimum(lay, 1.0) * (LEAK_MAX * q)       # local, additive, <= LEAK_MAX
    elif tr_in[0] == 'glare':
        post = tr_in[2]
        k = f - starts[i]
        if k < post:
            img = glare(img, GLARE_IN, (1 - k / post) ** 1.3, color=(1.0, 0.96, 0.88))
    elif tr_in[0] == 'fadein':
        n = tr_in[1]
        k = f - starts[i]
        if k < n:
            img = img * smoothstep(k / n) ** 1.2

    # --- transition out of shot i (we are before the next cut)
    if i + 1 < len(EDL):
        nxt, c = EDL[i + 1][3], starts[i + 1]
        if nxt[0] == 'dissolve':
            n = nxt[1]
            k = c - f                     # frames left before the cut (1..)
            if k <= n - n // 2:
                a = smoothstep((n // 2 - k + 0.5) / n)
                img = img * (1 - a) + src.get(i + 1, f - c) * a
        elif nxt[0] == 'flash':
            pre = nxt[1]
            k = c - f
            if k <= pre:
                p = ((pre - k + 1) / pre) ** 1.5
                img = bloom(img, 0.8 * p, thresh=0.6, sigma=45)
                img = add_glow(img, SIGN_POS, 0.11, (1.0, 0.72, 0.84), POP_MAX * p)
                img = knee(img, 0.80, 0.97)
        elif nxt[0] == 'leak':
            pre = nxt[1]
            k = c - f
            if k <= pre:
                p = (pre - k + 1) / pre
                img = bloom(img, 1.4 * p ** 1.5, thresh=0.55, sigma=45)
                img = add_glow(img, LAMP_POS, 0.10, (1.0, 0.80, 0.50), 0.30 * p ** 1.6)
                img = knee(img, 0.82, 0.975)
        elif nxt[0] == 'glare':
            pre = nxt[1]
            k = c - f
            if k <= pre:
                img = glare(img, GLARE_OUT, ((pre - k + 1) / pre) ** 1.3)
    else:
        # the finale: the sun blooms gently over the last 2 beats (sun core kept below clipping),
        # then the last END_HALO frames rise to the warm golden halation peak on the bar-13 downbeat
        k = f - (content - END_WHITE)
        kh = f - (content - END_HALO)
        if k >= 0:
            img = finale_halo(img, (k + 1) / END_WHITE, smoothstep((kh + 1) / END_HALO) if kh >= 0 else 0.0)
    return img


_KR, _KB = 0.2126, 0.0722
_KG = 1 - _KR - _KB


def to_yuv420(img, rng):
    """Float RGB -> BT.709 limited-range 8-bit 4:2:0 (left-sited chroma) with TPDF dither."""
    img = np.clip(img, 0.0, 1.0)
    R, G, B = img[..., 0], img[..., 1], img[..., 2]
    Yn = _KR * R + _KG * G + _KB * B
    Cb = (B - Yn) / (2 * (1 - _KB))
    Cr = (R - Yn) / (2 * (1 - _KR))
    Y = 16 + 219 * Yn
    # 4:2:0: average row pairs, then [1 2 1]/4 around even columns (MPEG-2/H.264 default siting)
    def sub(c):
        v = 0.5 * (c[0::2] + c[1::2])
        vp = np.pad(v, ((0, 0), (1, 1)), mode='edge')
        return 0.25 * vp[:, 0:-2:2] + 0.5 * vp[:, 1:-1:2] + 0.25 * vp[:, 2::2]
    U = 128 + 224 * sub(Cb)
    V = 128 + 224 * sub(Cr)
    planes = []
    for P in (Y, U, V):
        d = rng.random(P.shape, dtype=np.float32) - rng.random(P.shape, dtype=np.float32)
        planes.append(np.clip(np.rint(P + d), 0, 255).astype(np.uint8))
    return b''.join(p.tobytes() for p in planes)


def build_video(starts, content, total, frames_only=None):
    src = Sources()
    if frames_only is not None:
        outdir = os.path.join(ROOT, 'out', 'previews', 'edit')
        os.makedirs(outdir, exist_ok=True)
        for f in sorted(frames_only):
            img = compose(src, f, starts, content)
            cv2.imwrite(os.path.join(outdir, f'f{f:04d}.png'),
                        (np.clip(img[..., ::-1], 0, 1) * 255 + 0.5).astype(np.uint8))
        src.close()
        print('wrote', len(frames_only), 'preview frames to', outdir)
        return
    # Encode to a temp file next to OUT and only replace OUT once the encode fully succeeded and the
    # result has the expected frame count (an aborted run must never leave a truncated montage.mp4).
    tmp = OUT[:-4] + '.tmp.mp4'
    if os.path.exists(tmp):
        os.remove(tmp)
    enc = subprocess.Popen(
        ['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
         '-f', 'rawvideo', '-pix_fmt', 'yuv420p', '-s', f'{W}x{H}', '-r', str(FPS),
         '-color_range', 'tv', '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709',
         '-i', '-', '-i', SCORE,
         '-map', '0:v', '-map', '1:a',
         '-c:v', 'libx264', '-preset', 'slow', '-crf', '14', '-profile:v', 'high', '-tune', 'film',
         '-x264-params', 'aq-mode=3:aq-strength=0.9:deblock=-1,-1',
         '-pix_fmt', 'yuv420p', '-r', str(FPS), '-g', '48',
         '-color_range', 'tv', '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709',
         '-c:a', 'aac', '-b:a', '256k', '-ar', '48000',
         '-t', f'{total / FPS:.4f}', '-movflags', '+faststart', '-f', 'mp4', tmp],
        stdin=subprocess.PIPE)
    try:
        for f in range(total):
            i = max(j for j, s in enumerate(starts) if s <= f)
            src.release_before(i - 1)
            img = compose(src, f, starts, content)
            enc.stdin.write(to_yuv420(img, np.random.default_rng(1000 + f)))
            if f % 48 == 0:
                print(f'  frame {f}/{total}', flush=True)
        enc.stdin.close()
        if enc.wait() != 0:
            raise RuntimeError('ffmpeg encode failed')
        got = probe_frames(tmp)
        if got != total:
            raise RuntimeError(f'encoded {got} frames, expected {total}')
    except BaseException:
        enc.kill()
        enc.wait()
        if os.path.exists(tmp):
            os.remove(tmp)
        print('encode failed -- left', OUT, 'untouched')
        raise
    finally:
        src.close()
    os.replace(tmp, OUT)


def edl_report(starts, content):
    rows = []
    for (sid, src_in, beats, tr), s in zip(EDL, starts):
        rows.append((sid, tr, s, beats, src_in))
        print(f'{s / FPS:7.3f}s  beat {s // FPB:2d}  {sid:22s} src {src_in / FPS:5.2f}-{(src_in + beats * FPB) / FPS:5.2f}s '
              f'({beats * FPB / FPS:.3f}s, {beats} beats)  in: {tr}')
    return rows


def main():
    starts, content, total = timeline()
    print(f'timeline: content {content} frames ({content / FPS:.3f}s), total {total} frames ({total / FPS:.3f}s)')
    edl_report(starts, content)
    if '--frames' in sys.argv:
        spec = sys.argv[sys.argv.index('--frames') + 1]
        build_video(starts, content, total, frames_only=[int(x) for x in spec.split(',')])
        return
    cut_beats = [s // FPB for s in starts]
    glare_beats = [s // FPB for s, e in zip(starts, EDL) if e[3][0] == 'glare']
    if '--reuse-score' not in sys.argv or not os.path.exists(SCORE):
        synth_score(total / FPS, cut_beats, glare_beats)
    if '--score' in sys.argv:
        return
    ensure_shots()
    check_windows(starts, content)
    build_video(starts, content, total)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
