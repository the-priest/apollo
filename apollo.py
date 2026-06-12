#!/usr/bin/env python3
"""
APOLLO — AI song forge.
Pick a vibe, click GENERATE, get a song. Two engines:
  cloud : DeepSeek (SiliconFlow) writes title/lyrics/style -> MiniMax music-2.6-free renders a real song with sung vocals (mp3)
  local : DeepSeek composes structure/chords/lyrics -> numpy synthesizes drums/bass/synths + espeak-ng robo-vocals (wav), fully offline
Single file, stdlib + numpy. Config: ~/.config/apollo/config.json
Author: The Priest's toolbench. License: do crimes responsibly (MIT).
"""
import argparse, hashlib, io, json, math, os, random, re, shutil, struct, subprocess, sys, tempfile, threading, time, traceback, urllib.request, urllib.error, wave, zlib

try:
    import numpy as np
except ImportError:
    sys.exit("[apollo] numpy missing. Install it:  sudo apt install python3-numpy")

SR = 44100
APP = "apollo"
VERSION = "1.2.0"
CONF_DIR = os.path.expanduser("~/.config/apollo")
CONF_PATH = os.path.join(CONF_DIR, "config.json")
LIB_DIR = os.path.expanduser("~/Music/Apollo")
TMP_DIR = os.path.join(tempfile.gettempdir(), "apollo_audio")
os.makedirs(TMP_DIR, exist_ok=True)

DEFAULT_CONF = {
    "siliconflow_api_key": "",
    "siliconflow_base_url": "https://api.siliconflow.com/v1",
    "siliconflow_model": "deepseek-ai/DeepSeek-V4-Flash",
    "groq_api_key": "",                       # fallback ONLY
    "groq_model": "llama-3.3-70b-versatile",
    "minimax_api_key": "",
    "minimax_model": "music-2.6-free",
    "neural_python": "",          # python interpreter of the ACE-Step venv (set by --setup-neural)
    "neural_checkpoint": "",      # optional local checkpoint dir; blank = auto-download to ~/.cache/ace-step
    "neural_steps": 27,           # diffusion steps (higher = better + slower)
    "neural_guidance": 12.0,
    "hosted_space": "ACE-Step/ACE-Step",   # free HF Space for no-GPU neural vocals
    "hf_token": "",               # optional: your own HF token = your own queue/quota (free)
    "port": 8585,
}

NEURAL_DIR = os.path.expanduser("~/.local/share/apollo/neural-venv")
NEURAL_SECONDS = {"short": 75, "standard": 120, "full": 180}

def load_conf():
    conf = dict(DEFAULT_CONF)
    try:
        with open(CONF_PATH) as f:
            conf.update({k: v for k, v in json.load(f).items() if k in DEFAULT_CONF})
    except Exception:
        pass
    # env overrides
    env_map = {"SILICONFLOW_API_KEY": "siliconflow_api_key", "MINIMAX_API_KEY": "minimax_api_key", "GROQ_API_KEY": "groq_api_key"}
    for env, key in env_map.items():
        if os.environ.get(env):
            conf[key] = os.environ[env]
    return conf

def save_conf(conf):
    os.makedirs(CONF_DIR, exist_ok=True)
    with open(CONF_PATH, "w") as f:
        json.dump(conf, f, indent=2)
    try: os.chmod(CONF_PATH, 0o600)
    except Exception: pass

CONF = load_conf()

# ----------------------------------------------------------------------------- music theory
NOTE_SEMI = {"C":0,"C#":1,"DB":1,"D":2,"D#":3,"EB":3,"E":4,"F":5,"F#":6,"GB":6,"G":7,"G#":8,"AB":8,"A":9,"A#":10,"BB":10,"B":11}
MAJOR = [0,2,4,5,7,9,11]
MINOR = [0,2,3,5,7,8,10]
CHORD_Q = {"":[0,4,7],"maj":[0,4,7],"m":[0,3,7],"min":[0,3,7],"7":[0,4,7,10],"maj7":[0,4,7,11],
           "m7":[0,3,7,10],"dim":[0,3,6],"aug":[0,4,8],"sus2":[0,2,7],"sus4":[0,5,7],
           "add9":[0,4,7,14],"5":[0,7,12],"m7b5":[0,3,6,10],"6":[0,4,7,9],"m6":[0,3,7,9]}

def midi_hz(m): return 440.0 * 2.0 ** ((m - 69) / 12.0)

def parse_key(s):
    """'A minor' -> (tonic_semi, scale_list, 'minor')"""
    try:
        parts = str(s).strip().split()
        tonic = NOTE_SEMI[parts[0].upper()]
        mode = "minor" if (len(parts) > 1 and parts[1].lower().startswith("min")) else "major"
        return tonic, (MINOR if mode == "minor" else MAJOR), mode
    except Exception:
        return 9, MINOR, "minor"  # A minor

def parse_chord(sym, key_tonic=9, minor=True):
    """'Am7' -> list of midi pitches (root octave 3). Falls back to i chord."""
    m = re.match(r"^([A-Ga-g][#b]?)(.*)$", str(sym).strip())
    if m:
        root = NOTE_SEMI.get(m.group(1).upper())
        if root is None:
            root = NOTE_SEMI.get(m.group(1)[0].upper())
        qual = m.group(2).strip()
        ints = CHORD_Q.get(qual)
        if ints is None:
            ints = CHORD_Q["m"] if qual.startswith("m") and not qual.startswith("maj") else CHORD_Q[""]
        return [48 + root + i for i in ints]
    ints = CHORD_Q["m"] if minor else CHORD_Q[""]
    return [48 + key_tonic + i for i in ints]

def syllables(word):
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w: return 1
    groups = re.findall(r"[aeiouy]+", w)
    n = len(groups)
    if w.endswith("e") and n > 1 and not w.endswith(("le","ee","ye")): n -= 1
    return max(1, n)

# ----------------------------------------------------------------------------- synth primitives
def _t(dur): return np.arange(int(dur * SR), dtype=np.float32) / SR

def env_adsr(n, a=0.01, d=0.05, s=0.8, r=0.05):
    a_n, d_n, r_n = int(a*SR), int(d*SR), int(r*SR)
    a_n = min(a_n, n); d_n = min(d_n, max(0, n - a_n)); r_n = min(r_n, max(0, n - a_n - d_n))
    e = np.full(n, s, dtype=np.float32)
    if a_n: e[:a_n] = np.linspace(0, 1, a_n, dtype=np.float32)
    if d_n: e[a_n:a_n+d_n] = np.linspace(1, s, d_n, dtype=np.float32)
    if r_n: e[n-r_n:] *= np.linspace(1, 0, r_n, dtype=np.float32)
    return e

def osc_sine(f, dur): return np.sin(2*np.pi*f*_t(dur)).astype(np.float32)

def osc_saw(f, dur, harmonics=None):
    t = _t(dur); y = np.zeros_like(t)
    K = harmonics or max(1, min(18, int((SR/2) / max(f, 20.0) / 1.2)))
    for k in range(1, K+1):
        y += np.sin(2*np.pi*k*f*t) / k
    return (y * (2/np.pi)).astype(np.float32)

def supersaw(f, dur, detune_cents=9.0, voices=3):
    y = np.zeros(int(dur*SR), dtype=np.float32)
    offs = np.linspace(-detune_cents, detune_cents, voices)
    for c in offs:
        y += osc_saw(f * 2**(c/1200.0), dur)
    return y / voices

def osc_tri(f, dur):
    t = _t(dur)
    return (2/np.pi*np.arcsin(np.sin(2*np.pi*f*t))).astype(np.float32)

def noise(dur): return (np.random.default_rng(0xC0FFEE ^ int(dur*1e6)).standard_normal(int(dur*SR))).astype(np.float32)

def fft_filter(x, lp=None, hp=None, order=4):
    """Cheap zero-phase butterworth-ish magnitude filter via rFFT."""
    n = len(x)
    if n == 0: return x
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1/SR)
    mag = np.ones_like(f)
    if lp: mag *= 1.0 / np.sqrt(1.0 + (f / lp) ** (2*order))
    if hp:
        with np.errstate(divide="ignore"):
            ratio = np.where(f > 0, hp / np.maximum(f, 1e-9), 1e9)
        mag *= 1.0 / np.sqrt(1.0 + ratio ** (2*order))
    return np.fft.irfft(X * mag, n).astype(np.float32)

def soft_drive(x, amt=1.5): return np.tanh(x * amt) / np.tanh(amt)

def fft_convolve(x, ir):
    n = len(x) + len(ir) - 1
    nfft = 1 << (n - 1).bit_length()
    Y = np.fft.rfft(x, nfft) * np.fft.rfft(ir, nfft)
    return np.fft.irfft(Y, nfft)[:n].astype(np.float32)

def make_reverb_ir(seconds=1.6, seed=7):
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    decay = np.exp(-t / (seconds * 0.32)).astype(np.float32)
    irl = (rng.standard_normal(n).astype(np.float32)) * decay
    irr = (rng.standard_normal(n).astype(np.float32)) * decay
    irl = fft_filter(irl, lp=7500, hp=300); irr = fft_filter(irr, lp=7500, hp=300)
    g = 0.32 / max(np.max(np.abs(irl)), 1e-9)
    return irl * g, irr * g

# ----------------------------------------------------------------------------- drum kit (rendered once, cached)
_DRUMS = {}
def drum(name):
    if name in _DRUMS: return _DRUMS[name]
    if name == "kick":
        t = _t(0.45)
        f = 150*np.exp(-t*28) + 48
        phase = 2*np.pi*np.cumsum(f)/SR
        y = np.sin(phase)*np.exp(-t*7)*1.1
        y[:int(0.005*SR)] += noise(0.005)*0.5
        y = soft_drive(y, 1.6)
    elif name == "snare":
        t = _t(0.30)
        tone = np.sin(2*np.pi*186*t)*np.exp(-t*30)*0.5
        nz = fft_filter(noise(0.30)*np.exp(-t*16), hp=900)
        y = tone + nz*0.9
    elif name == "hat":
        t = _t(0.08); y = fft_filter(noise(0.08)*np.exp(-t*70), hp=6500)
    elif name == "ohat":
        t = _t(0.38); y = fft_filter(noise(0.38)*np.exp(-t*9), hp=6000)
    elif name == "clap":
        y = np.zeros(int(0.32*SR), dtype=np.float32)
        for i, off in enumerate((0.0, 0.011, 0.022)):
            s = int(off*SR); seg = fft_filter(noise(0.012), hp=1200)
            y[s:s+len(seg)] += seg*(0.9-0.15*i)
        t = _t(0.32); tail = fft_filter(noise(0.32)*np.exp(-t*11), hp=1100)
        y += tail*0.7
    elif name == "shaker":
        t = _t(0.09); y = fft_filter(noise(0.09)*np.exp(-t*45), hp=4500, lp=11000)*0.8
    elif name == "rim":
        t = _t(0.06); y = (np.sin(2*np.pi*820*t)*0.6 + fft_filter(noise(0.06), hp=2500)*0.5)*np.exp(-t*60)
    else:
        y = np.zeros(64, dtype=np.float32)
    peak = np.max(np.abs(y)) or 1.0
    _DRUMS[name] = (y / peak).astype(np.float32)
    return _DRUMS[name]

def place(track, sample, t_sec, vel=1.0):
    i = int(t_sec * SR)
    if i < 0 or i >= len(track): return
    seg = sample[: len(track) - i]
    track[i:i+len(seg)] += seg * vel

def addto(dst, src, gain=1.0):
    """Add src into dst from index 0, tolerating any length mismatch (no broadcast errors)."""
    m = min(len(dst), len(src))
    if m: dst[:m] += src[:m] * gain
    return dst

def fit(x, n):
    """Force x to exactly length n (pad with zeros or truncate)."""
    if len(x) == n: return x
    if len(x) > n: return x[:n]
    out = np.zeros(n, dtype=np.float32); out[:len(x)] = x; return out

# ----------------------------------------------------------------------------- genres
GENRES = {
    "synthwave": dict(bpm=(96,118), swing=0.50, harm="pad", bass="eights", lead="saw", arp=True, side=True,
                      kit="electro", verb=0.40, tags="80s synthwave, retrowave, analog synths, gated reverb drums, neon nostalgia"),
    "pop":       dict(bpm=(100,126), swing=0.50, harm="pluck", bass="roots8", lead="tri", arp=False, side=False,
                      kit="pop", verb=0.28, tags="modern pop, catchy hooks, polished production, radio-ready"),
    "rock":      dict(bpm=(118,150), swing=0.50, harm="power", bass="eights", lead="saw", arp=False, side=False,
                      kit="rock", verb=0.22, tags="rock, driving electric guitars, live drums, anthemic"),
    "hiphop":    dict(bpm=(80,96),  swing=0.56, harm="epiano", bass="syncop", lead="sine", arp=False, side=False,
                      kit="hiphop", verb=0.25, tags="hip-hop, boom bap, heavy 808 bass, head-nod groove"),
    "edm":       dict(bpm=(124,130), swing=0.50, harm="pad", bass="eights", lead="saw", arp=True, side=True,
                      kit="edm", verb=0.30, tags="EDM, festival big-room, four-on-the-floor, euphoric drops"),
    "lofi":      dict(bpm=(70,86),  swing=0.58, harm="epiano", bass="roots", lead="sine", arp=False, side=False,
                      kit="lofi", verb=0.35, tags="lo-fi hip hop, dusty vinyl, mellow jazzy chords, rainy-day chill"),
    "metal":     dict(bpm=(140,180), swing=0.50, harm="power", bass="eights", lead="saw", arp=False, side=False,
                      kit="metal", verb=0.18, tags="heavy metal, distorted riffing, double kick, aggressive"),
    "folk":      dict(bpm=(88,112), swing=0.52, harm="epiano", bass="roots", lead="tri", arp=False, side=False,
                      kit="folk", verb=0.30, tags="indie folk, acoustic, warm and intimate, storytelling"),
}
MOOD_TAGS = {"dark":"dark, brooding","upbeat":"upbeat, energetic","melancholy":"melancholic, wistful",
             "aggressive":"aggressive, intense","chill":"chill, relaxed","epic":"epic, cinematic","romantic":"romantic, tender"}

def drum_events(kit, bars, energy, rng, last=False):
    """yield (name, beat, vel) for a section. beat is absolute within section (4/4)."""
    ev = []
    e = max(0.2, min(1.0, energy/10.0))
    for b in range(bars):
        o = b*4
        if kit in ("pop","rock","metal","folk"):
            ev += [("kick", o+0, 1.0), ("kick", o+2, 0.95)]
            if kit in ("rock","metal") and e > 0.55: ev.append(("kick", o+2.5, 0.8))
            sn = "rim" if (kit=="folk" and e < 0.5) else "snare"
            ev += [(sn, o+1, 0.95), (sn, o+3, 1.0)]
            hat = "shaker" if kit == "folk" else "hat"
            step = 0.5 if (kit!="metal" or e<0.75) else 0.25
            for h in np.arange(0, 4, step): ev.append((hat, o+h, 0.55 + 0.25*(h%1==0)))
            if kit=="metal" and e>0.75:
                for h in np.arange(0, 4, 0.25): ev.append(("kick", o+h, 0.55))
        elif kit == "hiphop":
            ev += [("kick", o+0, 1.0), ("kick", o+1.75, 0.85), ("kick", o+2.5, 0.95)]
            ev += [("snare", o+1, 0.95), ("clap", o+1, 0.5), ("snare", o+3, 1.0), ("clap", o+3, 0.55)]
            for h in np.arange(0, 4, 0.5): ev.append(("hat", o+h, 0.5 + 0.3*(h%1==0) + rng.uniform(-.08,.08)))
            if rng.random() < 0.35: ev += [("hat", o+3.25, 0.45), ("hat", o+3.75, 0.5)]
        elif kit == "lofi":
            ev += [("kick", o+0, 0.95), ("kick", o+2.5, 0.8)]
            ev += [("snare", o+1+rng.uniform(0,.03), 0.8), ("snare", o+3+rng.uniform(0,.03), 0.85)]
            for h in np.arange(0, 4, 0.5): ev.append(("hat", o+h, 0.35 + 0.25*(h%1==0) + rng.uniform(-.1,.1)))
        else:  # edm / electro(synthwave)
            four = (kit == "edm") or e > 0.65
            if four: ev += [("kick", o+k, 1.0) for k in range(4)]
            else:    ev += [("kick", o+0, 1.0), ("kick", o+2, 0.95), ("kick", o+2.75, 0.7)]
            ev += [("clap", o+1, 0.85), ("clap", o+3, 0.9)] if kit=="edm" else [("snare", o+1, 0.95), ("snare", o+3, 1.0)]
            for h in (0.5,1.5,2.5,3.5): ev.append(("ohat" if kit=="edm" else "hat", o+h, 0.6))
            if kit != "edm":
                for h in np.arange(0,4,0.5): ev.append(("hat", o+h, 0.4))
        # section-end fill
        if last and b == bars-1:
            for i, h in enumerate(np.arange(3, 4, 0.25)): ev.append(("snare", o+h, 0.5 + i*0.12))
    return [(n, t, v*(0.65+0.35*e)) for (n, t, v) in ev]

def bass_events(style, chords_midi, bars, energy, rng):
    """yield (beat, dur_beats, midi). chords_midi: one chord (list) per bar."""
    ev = []
    for b in range(bars):
        root = chords_midi[b][0] - 12  # octave 2
        fifth = root + 7
        o = b*4
        if style == "roots":
            ev.append((o, 4, root))
        elif style == "roots8":
            ev += [(o, 2, root), (o+2, 1.5, root), (o+3.5, 0.5, fifth)]
        elif style == "eights":
            for h in np.arange(0, 4, 0.5):
                ev.append((o+h, 0.5, root if (h % 2) or energy < 6 else (fifth if h==2 else root)))
        elif style == "syncop":
            ev += [(o, 0.75, root), (o+1.75, 0.75, root), (o+2.5, 1.0, root if rng.random()<.7 else fifth)]
        else:
            ev.append((o, 4, root))
    return ev

# ----------------------------------------------------------------------------- melody
CONTOURS = {"rise": [0,1,2,3,4,5], "fall": [5,4,3,2,1,0], "arc": [0,2,4,5,3,1],
            "hook": [3,1,3,1,3,1], "wave": [2,4,1,3,0,4]}

def gen_line_melody(words, chord_midi, key_tonic, scale, contour, rng, beats=8.0):
    """Map words of one lyric line across `beats` beats. Returns [(beat, dur, midi, word)]."""
    if not words: return []
    weights = np.array([syllables(w) for w in words], dtype=float)
    usable = beats - 1.5  # breath at line end
    durs = np.maximum(0.5, np.round(weights / weights.sum() * usable * 2) / 2)
    while durs.sum() > usable: durs[np.argmax(durs)] -= 0.5
    while durs.sum() < usable - 0.49: durs[np.argmin(durs)] += 0.5
    # candidate pitches: scale notes in a singable octave around tonic+12
    base = 57 + ((key_tonic + 3) % 12)  # roughly A3..G#4 region
    pool = sorted({base + o*12 + s for s in scale for o in (0, 1) if 55 <= base + o*12 + s <= 79})
    path = CONTOURS.get(contour, CONTOURS["arc"])
    lo, hi = 0, len(pool) - 1
    out, t = [], 0.0
    chord_pcs = {p % 12 for p in chord_midi}
    for i, (w, d) in enumerate(zip(words, durs)):
        frac = path[min(i, len(path)-1) % len(path)] / 5.0
        idx = int(round(lo + frac * (hi - lo) * 0.55 + (hi-lo)*0.2)) + rng.integers(-1, 2)
        idx = max(0, min(len(pool)-1, idx))
        p = pool[idx]
        strong = (t % 2.0) < 0.01 or i == len(words)-1
        if strong and (p % 12) not in chord_pcs:  # snap strong beats to chord tones
            cand = [q for q in pool if (q % 12) in chord_pcs]
            p = min(cand, key=lambda q: abs(q - p)) if cand else p
        out.append((t, float(d), p, w))
        t += float(d)
    return out

# ----------------------------------------------------------------------------- robo vocals (espeak-ng)
ESPEAK = shutil.which("espeak-ng") or shutil.which("espeak")
VOICE_VARIANT = {"male": "+m3", "female": "+f4", "croak": "+croak", "whisper": "+whisper"}
_WORD_CACHE = {}

def espeak_word(word, variant="+m3"):
    key = (word.lower(), variant)
    if key in _WORD_CACHE: return _WORD_CACHE[key]
    clean = re.sub(r"[^A-Za-z' \-]", "", word) or "ah"
    try:
        out = subprocess.run([ESPEAK, "--stdout", "-v", "en-us" + variant, "-s", "150", "-a", "180", clean],
                             capture_output=True, timeout=10).stdout
        with wave.open(io.BytesIO(out)) as w:
            sr_in = w.getframerate()
            raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        if sr_in != SR and len(raw) > 1:
            x_old = np.linspace(0, 1, len(raw)); x_new = np.linspace(0, 1, int(len(raw) * SR / sr_in))
            raw = np.interp(x_new, x_old, raw).astype(np.float32)
        # trim silence
        env = np.abs(raw); thr = max(1e-4, env.max() * 0.03)
        idx = np.where(env > thr)[0]
        raw = raw[idx[0]:idx[-1]+1] if len(idx) else raw
    except Exception:
        raw = np.zeros(int(0.1*SR), dtype=np.float32)
    _WORD_CACHE[key] = raw
    return raw

def estimate_f0(x):
    if len(x) < 1024: return 140.0
    core = x[len(x)//5: -len(x)//5] if len(x) > 2048 else x
    seg = core[:4096] - np.mean(core[:4096])
    if np.max(np.abs(seg)) < 1e-4: return 140.0
    corr = np.correlate(seg, seg, "full")[len(seg)-1:]
    lo, hi = int(SR/350), int(SR/70)
    if hi >= len(corr): return 140.0
    lag = lo + int(np.argmax(corr[lo:hi]))
    if corr[lag] < 0.22 * corr[0]: return 140.0
    return SR / lag

def sing_word(word, target_hz, dur, variant="+m3"):
    raw = espeak_word(word, variant)
    if not len(raw) or np.max(np.abs(raw)) < 1e-4:
        return np.zeros(int(dur*SR), dtype=np.float32)
    f0 = estimate_f0(raw)
    # fold target into a comfortable band so the robot doesn't chipmunk
    t = target_hz
    while t > 320: t /= 2
    while t < 100: t *= 2
    ratio = max(0.55, min(1.9, t / f0))
    x_old = np.arange(len(raw)); n_new = max(8, int(len(raw) / ratio))
    y = np.interp(np.linspace(0, len(raw)-1, n_new), x_old, raw).astype(np.float32)
    n_target = int(dur * SR)
    if len(y) >= n_target:
        y = y[:n_target]
    else:  # sustain: loop the tail with crossfade + vibrato
        tail_len = max(int(0.06*SR), int(len(y)*0.35))
        tail = y[-tail_len:]
        xf = int(0.008*SR)
        chunks = [y]
        need = n_target - len(y)
        lt = tail.copy()
        while need > 0:
            piece = lt[:min(len(lt), need + xf)]
            chunks.append(piece)
            need -= (len(piece) - xf)
        y = np.zeros(sum(len(c) for c in chunks), dtype=np.float32)
        pos = 0
        for c in chunks:
            if pos > 0 and xf > 0 and len(c) > xf:
                fade = np.linspace(0, 1, xf, dtype=np.float32)
                y[pos-xf:pos] = y[pos-xf:pos]*(1-fade) + c[:xf]*fade
                y[pos:pos+len(c)-xf] = c[xf:]
                pos += len(c) - xf
            else:
                y[pos:pos+len(c)] = c; pos += len(c)
        y = y[:n_target]
        # vibrato on the sustained part
        vt = np.arange(n_target)/SR
        depth = 0.018 * np.clip((vt - len(raw)/ratio/SR) / 0.25, 0, 1)
        warp = vt + depth*np.sin(2*np.pi*5.3*vt)/ (2*np.pi*5.3)
        y = np.interp(np.clip(warp*SR, 0, n_target-1), np.arange(n_target), y).astype(np.float32)
    n = len(y)
    y *= env_adsr(n, a=0.012, d=0.02, s=1.0, r=min(0.05, dur*0.25))
    pk = np.max(np.abs(y)) or 1.0
    return (y / pk * 0.9).astype(np.float32)

# ----------------------------------------------------------------------------- instrument renderers
def render_bass_note(midi, dur, style):
    f = midi_hz(midi); n = int(dur*SR)
    if style in ("syncop",):  # 808-ish sub
        y = osc_sine(f, dur) + 0.3*osc_sine(f*2, dur)
        y = soft_drive(y, 1.8)
    elif style in ("eights", "roots8"):
        y = fft_filter(addto(supersaw(f, dur, 5, 2)*0.8, osc_sine(f, dur), 0.5), lp=900)
    else:
        y = osc_sine(f, dur)*0.8 + osc_tri(f, dur)*0.4
    return fit(y, n) * env_adsr(n, a=0.004, d=0.03, s=0.85, r=min(0.06, dur*0.3))

def render_harm(chord, dur, kind, energy):
    n = int(dur*SR); y = np.zeros(n, dtype=np.float32)
    chord = list(chord) or [57]
    if kind == "pad":
        for p in chord:
            addto(y, fft_filter(supersaw(midi_hz(p+12), dur, 8, 3), lp=1800 + 1800*energy/10), 1.0/len(chord))
        y *= env_adsr(n, a=min(0.5, dur*0.4), d=0.1, s=0.9, r=min(0.5, dur*0.3))
    elif kind == "pluck":
        hits = np.arange(0, dur-0.01, 0.5)
        tone = np.zeros(int(min(0.4, dur)*SR), dtype=np.float32)
        for p in chord:
            addto(tone, fft_filter(supersaw(midi_hz(p+12), min(0.4, dur), 7, 2), lp=3000), 1.0/len(chord))
        decay = np.exp(-np.arange(len(tone))/SR/0.16).astype(np.float32)
        tone = tone * decay
        for i, h in enumerate(hits):
            place(y, tone, h, 0.9 if i % 2 == 0 else 0.6)
    elif kind == "power":
        root = chord[0]
        stack = [root, root+7, root+12]
        hits = np.arange(0, dur-0.01, 0.5)
        tone = np.zeros(int(0.45*SR), dtype=np.float32)
        for p in stack:
            addto(tone, supersaw(midi_hz(p), 0.45, 11, 3), 1.0/3)
        tone = fft_filter(soft_drive(tone, 3.2), lp=4200, hp=90)
        tone = tone * np.exp(-np.arange(len(tone))/SR/0.13).astype(np.float32)
        for h in hits: place(y, tone, h, 0.85)
    else:  # epiano
        t = _t(dur)
        for p in chord:
            f = midi_hz(p+12)
            v = (np.sin(2*np.pi*f*t) + 0.45*np.sin(2*np.pi*2*f*t)*np.exp(-t*6)
                 + 0.18*np.sin(2*np.pi*3.01*f*t)*np.exp(-t*9))
            v *= np.exp(-t/1.4) * (1 + 0.07*np.sin(2*np.pi*4.4*t))
            addto(y, v.astype(np.float32), 1.0/len(chord))
        y *= env_adsr(n, a=0.005, d=0.1, s=0.9, r=min(0.3, dur*0.3))
    return fit(y, n)

def render_lead_note(midi, dur, wavekind):
    f = midi_hz(midi + 12); n = int(dur*SR)
    if wavekind == "saw": y = fft_filter(supersaw(f, dur, 7, 3), lp=5200)
    elif wavekind == "tri": y = osc_tri(f, dur)
    else: y = addto(osc_sine(f, dur), osc_sine(2*f, dur), 0.2)
    return fit(y, n) * env_adsr(n, a=0.01, d=0.05, s=0.8, r=min(0.08, dur*0.3))

def render_arp(chord, dur, bps_beat_sec):
    n = int(dur*SR); y = np.zeros(n, dtype=np.float32)
    step = bps_beat_sec / 4.0
    tones = sorted(chord) + [chord[0] + 12]
    i = 0; t = 0.0
    while t < dur - 0.01:
        f = midi_hz(tones[i % len(tones)] + 12)
        seg_d = min(step*1.5, dur - t)
        seg = fft_filter(supersaw(f, seg_d, 6, 2), lp=4500)
        seg *= np.exp(-np.arange(len(seg))/SR/0.09).astype(np.float32)
        place(y, seg, t, 0.8)
        t += step; i += 1
    return y

# ----------------------------------------------------------------------------- local engine: spec -> wav
def render_local(spec, opts, progress=lambda s: None):
    rng = np.random.default_rng(opts.get("seed", int(time.time())))
    g = GENRES.get(spec.get("_genre", "synthwave"), GENRES["synthwave"])
    bpm = float(spec.get("bpm", 110)); bpm = max(56, min(200, bpm))
    beat = 60.0 / bpm
    tonic, scale, mode = parse_key(spec.get("key", "A minor"))
    sections = spec.get("sections", [])
    total_beats = sum(int(s.get("bars", 4)) * 4 for s in sections)
    tail = 2.0
    total = total_beats * beat + tail
    N = int(total * SR)
    tr = {k: np.zeros(N, dtype=np.float32) for k in ("drums","bass","harm","arp","lead","vox","snare_send")}

    vocal_variant = VOICE_VARIANT.get(opts.get("voice", "male"), "+m3")
    want_vox = bool(ESPEAK) and opts.get("voice") != "instrumental" and not opts.get("no_vocals")
    melody_cache = {}
    swing = g["swing"]

    def sw(b):  # swing offbeat 8ths
        fr = b % 1.0
        return b + (swing - 0.5) if abs(fr - 0.5) < 0.01 else b

    cur_beat = 0.0
    n_sec = len(sections)
    for si, sec in enumerate(sections):
        progress(f"synthesizing {sec.get('type','section')} ({si+1}/{n_sec})")
        bars = int(sec.get("bars", 4))
        energy = float(sec.get("energy", 6))
        chords = [parse_chord(c, tonic, mode == "minor") for c in sec.get("chords", [])]
        while len(chords) < bars: chords.append(chords[-1] if chords else parse_chord("Am"))
        sec_start = cur_beat * beat
        is_last_of_block = si + 1 < n_sec and sections[si+1].get("type") != sec.get("type")

        for (name, b, v) in drum_events(g["kit"], bars, energy, rng, last=is_last_of_block):
            t = sec_start + sw(b) * beat
            place(tr["drums"], drum(name), t, v)
            if name == "snare": place(tr["snare_send"], drum(name), t, v*0.8)

        for (b, d, m) in bass_events(g["bass"], chords, bars, energy, rng):
            place(tr["bass"], render_bass_note(m, d*beat, g["bass"]), sec_start + sw(b)*beat, 0.95)

        for bi, ch in enumerate(chords):
            t = sec_start + bi*4*beat
            place(tr["harm"], render_harm(ch, 4*beat, g["harm"], energy), t, 0.9)
            if g["arp"] and energy >= 5:
                place(tr["arp"], render_arp(ch, 4*beat, beat), t, 0.85)

        lyrics = [l for l in sec.get("lyrics", []) if str(l).strip()]
        contours = sec.get("contour", [])
        if lyrics:
            ck = (sec.get("type",""), tuple(lyrics))
            if ck not in melody_cache:
                lines = []
                for li, line in enumerate(lyrics):
                    words = [w for w in re.split(r"\s+", str(line).strip()) if w]
                    bar_i = min(li*2, bars-1)
                    cont = contours[li] if li < len(contours) else rng.choice(list(CONTOURS))
                    lines.append((li, gen_line_melody(words, chords[bar_i], tonic, scale, cont, rng)))
                melody_cache[ck] = lines
            for li, notes in melody_cache[ck]:
                line_start = sec_start + li*8*beat
                for (b, d, m, w) in notes:
                    t = line_start + b*beat
                    if want_vox:
                        place(tr["vox"], sing_word(w, midi_hz(m), d*beat*0.95, vocal_variant), t, 1.0)
                    lead_gain = 0.55 if not want_vox else 0.22
                    place(tr["lead"], render_lead_note(m, d*beat*0.95, g["lead"]), t, lead_gain)
        elif sec.get("type") in ("intro","outro","solo","bridge"):
            # instrumental motif: arpeggiate scale around chord tones
            for bi, ch in enumerate(chords):
                if bi % 2 == 0 or sec.get("type") == "solo":
                    seq = [ch[0]+12, ch[1 % len(ch)]+12, ch[0]+19, ch[-1]+12]
                    for k, m in enumerate(seq):
                        t = sec_start + (bi*4 + k*0.5) * beat
                        place(tr["lead"], render_lead_note(m, beat*0.5, g["lead"]), t, 0.5)
        cur_beat += bars * 4

    progress("mixing")
    # sidechain pump
    if g["side"]:
        t = np.arange(N)/SR
        ph = (t / beat) % 1.0
        duck = 1.0 - 0.55*np.exp(-ph/0.16)
        for k in ("harm","arp","bass"): tr[k] *= duck.astype(np.float32)

    gains = dict(drums=0.95, bass=0.78, harm=0.42, arp=0.32, lead=1.0, vox=1.05)
    dry = np.zeros(N, dtype=np.float32)
    for k, gv in gains.items(): addto(dry, tr[k], gv)
    irl, irr = make_reverb_ir(1.7 if g["verb"] > 0.3 else 1.1)
    send = np.zeros(N, dtype=np.float32)
    addto(send, tr["vox"], g["verb"]); addto(send, tr["lead"], 0.25); addto(send, tr["snare_send"], g["verb"]*0.6)
    wetL = fit(fft_convolve(send, irl), N); wetR = fit(fft_convolve(send, irr), N)
    width = np.zeros(N, dtype=np.float32); addto(width, tr["harm"], 0.18); addto(width, tr["arp"], 0.22)
    L = dry - width*0.5 + wetL
    R = dry + width*0.5 + wetR
    peak = max(np.max(np.abs(L)), np.max(np.abs(R)), 1e-9)
    L = soft_drive(L/peak*0.92, 1.25); R = soft_drive(R/peak*0.92, 1.25)
    return (L*0.97).astype(np.float32), (R*0.97).astype(np.float32), total

def write_wav(path, L, R):
    data = np.empty(len(L)*2, dtype=np.int16)
    data[0::2] = np.clip(L*32767, -32768, 32767).astype(np.int16)
    data[1::2] = np.clip(R*32767, -32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(data.tobytes())

def maybe_mp3(wav_path):
    if not shutil.which("ffmpeg"): return wav_path
    mp3 = wav_path[:-4] + ".mp3"
    r = subprocess.run(["ffmpeg","-y","-loglevel","error","-i",wav_path,"-b:a","224k",mp3], capture_output=True)
    return mp3 if r.returncode == 0 and os.path.exists(mp3) else wav_path

# ----------------------------------------------------------------------------- fallback template (no-API demo)
def template_spec(genre="synthwave"):
    return {
        "_genre": genre, "title": "Neon Litany", "bpm": 108, "key": "A minor",
        "sections": [
            {"type":"intro","bars":4,"chords":["Am","F","C","G"],"energy":4,"lyrics":[],"contour":[]},
            {"type":"verse","bars":8,"chords":["Am","F","C","G","Am","F","C","E"],"energy":5,
             "lyrics":["Streetlight static in my veins","Every signal calls my name","Midnight engines hum below","Chasing ghosts in monochrome"],
             "contour":["arc","rise","arc","fall"]},
            {"type":"chorus","bars":8,"chords":["F","G","Am","Am","F","G","C","E"],"energy":8,
             "lyrics":["We ride the neon down","Until the dark turns gold","We ride the neon down","No signal left to hold"],
             "contour":["hook","rise","hook","fall"]},
            {"type":"verse","bars":8,"chords":["Am","F","C","G","Am","F","C","E"],"energy":6,
             "lyrics":["Wires whisper through the rain","Every memory a flame","Turn the dial and disappear","Static angels gather near"],
             "contour":["arc","rise","wave","fall"]},
            {"type":"chorus","bars":8,"chords":["F","G","Am","Am","F","G","C","E"],"energy":9,
             "lyrics":["We ride the neon down","Until the dark turns gold","We ride the neon down","No signal left to hold"],
             "contour":["hook","rise","hook","fall"]},
            {"type":"outro","bars":4,"chords":["Am","F","C","G"],"energy":3,"lyrics":[],"contour":[]},
        ],
    }

# ----------------------------------------------------------------------------- LLM (SiliconFlow primary, Groq fallback ONLY)
class LLMError(Exception): pass

def _post_json(url, payload, headers, timeout=120):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

def call_llm(messages, max_tokens=3000, progress=lambda s: None):
    errs = []
    if CONF.get("siliconflow_api_key"):
        try:
            progress(f"composing with {CONF['siliconflow_model'].split('/')[-1]}")
            data = _post_json(CONF["siliconflow_base_url"].rstrip("/") + "/chat/completions",
                              {"model": CONF["siliconflow_model"], "messages": messages,
                               "max_tokens": max_tokens, "temperature": 0.85},
                              {"Authorization": f"Bearer {CONF['siliconflow_api_key']}"})
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            errs.append(f"siliconflow: {e}")
    if CONF.get("groq_api_key"):
        try:
            progress("siliconflow failed — falling back to groq")
            data = _post_json("https://api.groq.com/openai/v1/chat/completions",
                              {"model": CONF["groq_model"], "messages": messages,
                               "max_tokens": max_tokens, "temperature": 0.85},
                              {"Authorization": f"Bearer {CONF['groq_api_key']}"})
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            errs.append(f"groq: {e}")
    raise LLMError("; ".join(errs) if errs else "no LLM API key configured")

def extract_json(text):
    text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.M).strip()
    i, j = text.find("{"), text.rfind("}")
    if i == -1 or j == -1: raise ValueError("no JSON object found")
    return json.loads(text[i:j+1])

LENGTH_BARS = {"short": 30, "standard": 46, "full": 64}

def llm_local_spec(opts, progress):
    genre = opts["genre"]; g = GENRES[genre]
    bars_target = LENGTH_BARS.get(opts.get("length", "standard"), 46)
    mood = opts.get("mood", "auto"); tempo_hint = opts.get("tempo", "auto")
    bpm_lo, bpm_hi = g["bpm"]
    sys_p = "You are a songwriting engine. Reply with ONLY a single minified JSON object. No markdown, no commentary."
    user_p = f"""Write a complete original song spec as JSON with this exact schema:
{{"title":str,"bpm":int,"key":"<Note> major|minor","sections":[{{"type":"intro|verse|prechorus|chorus|bridge|solo|outro","bars":4|8,"chords":[one chord symbol per bar e.g. "Am","F","Cmaj7"],"energy":1-10,"lyrics":[lyric lines, [] for instrumental sections],"contour":["rise"|"fall"|"arc"|"hook"|"wave" per lyric line]}}]}}
Rules:
- Genre: {genre}. Mood: {mood}. Tempo: {tempo_hint} (bpm must be {bpm_lo}-{bpm_hi}).
- 5-9 sections, total bars about {bars_target}. Always start with intro (lyrics []) and end with outro (lyrics []).
- chords array length MUST equal bars. lyrics length MUST equal bars/2 for vocal sections.
- Every chorus section MUST have identical lyrics (same hook). Lines are 4-8 simple singable words, no punctuation except apostrophes.
- All chord roots must fit the chosen key.
Song idea from the user: {opts.get('idea') or 'your choice — surprise me'}"""
    raw = call_llm([{"role":"system","content":sys_p},{"role":"user","content":user_p}], progress=progress)
    try:
        spec = extract_json(raw)
    except Exception:
        progress("model returned bad JSON — asking it to fix")
        raw = call_llm([{"role":"system","content":sys_p},
                        {"role":"user","content":user_p},
                        {"role":"assistant","content":raw},
                        {"role":"user","content":"That was not valid JSON. Reply again with ONLY the corrected minified JSON object."}],
                       progress=progress)
        spec = extract_json(raw)
    return validate_spec(spec, genre)

def validate_spec(spec, genre):
    out = {"_genre": genre}
    out["title"] = str(spec.get("title", "Untitled"))[:80] or "Untitled"
    try: out["bpm"] = max(56, min(200, int(spec.get("bpm", 110))))
    except Exception: out["bpm"] = 110
    out["key"] = str(spec.get("key", "A minor"))
    secs = spec.get("sections", [])
    if not isinstance(secs, list) or not secs: raise ValueError("spec has no sections")
    clean, chorus_lyrics = [], None
    for s in secs[:10]:
        typ = str(s.get("type", "verse")).lower()
        bars = int(s.get("bars", 8)); bars = 8 if bars not in (2, 4, 8, 16) else bars
        chords = [str(c) for c in (s.get("chords") or [])][:bars]
        while len(chords) < bars: chords.append(chords[-1] if chords else "Am")
        lyr = [str(l).strip() for l in (s.get("lyrics") or []) if str(l).strip()]
        if typ in ("intro", "outro", "solo"): lyr = []
        lyr = lyr[: max(1, bars // 2)]
        if typ == "chorus":
            if chorus_lyrics is None: chorus_lyrics = lyr
            else: lyr = chorus_lyrics
        cont = [str(c) for c in (s.get("contour") or [])]
        try: energy = max(1, min(10, int(s.get("energy", 6))))
        except Exception: energy = 6
        clean.append({"type": typ, "bars": bars, "chords": chords, "lyrics": lyr, "contour": cont, "energy": energy})
    out["sections"] = clean
    return out

# ----------------------------------------------------------------------------- manual mode
SEMI_NOTE = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
GENRE_KEYS = {"synthwave":"A minor","pop":"C major","rock":"E minor","hiphop":"A minor",
              "edm":"F minor","lofi":"D minor","metal":"E minor","folk":"G major"}
# canonical progressions in A minor / C major, transposed to the song key
PROG_MIN = {"intro":["Am","F","C","G"],"verse":["Am","F","C","G"],"prechorus":["Dm","F","E","E"],
            "chorus":["F","G","Am","Am"],"bridge":["F","Em","Dm","E"],"solo":["Am","G","F","E"],
            "outro":["Am","F","C","G"],"hook":["F","G","Am","Am"]}
PROG_MAJ = {"intro":["C","G","Am","F"],"verse":["C","G","Am","F"],"prechorus":["Dm","Am","G","G"],
            "chorus":["F","G","C","Am"],"bridge":["Am","F","G","Em"],"solo":["C","F","G","G"],
            "outro":["C","G","Am","F"],"hook":["F","G","C","Am"]}

def transpose_sym(sym, semis):
    m = re.match(r"^([A-G][#b]?)(.*)$", sym)
    root = (NOTE_SEMI[m.group(1).upper()] + semis) % 12
    return SEMI_NOTE[root] + m.group(2)

TAG_RX = re.compile(r"^\s*[\[\(]\s*(intro|verse|pre[\s-]?chorus|chorus|bridge|solo|hook|outro|instrumental)[^\]\)]*[\]\)]\s*$", re.I)

def parse_tagged_lyrics(text):
    """Tagged lyric text -> [(type, [lines])]. Untagged text gets chunked into verse/chorus."""
    secs, cur, buf = [], None, []
    for raw in (text or "").splitlines():
        ln = raw.strip()
        m = TAG_RX.match(ln)
        if m:
            if cur is not None or buf: secs.append((cur or "verse", buf))
            t = re.sub(r"[\s-]", "", m.group(1).lower())
            cur = {"prechorus": "prechorus", "instrumental": "solo"}.get(t, t); buf = []
        elif ln:
            buf.append(ln[:90])
    if cur is not None or buf: secs.append((cur or "verse", buf))
    out = [(t, l[:8]) for t, l in secs if l or t in ("intro", "outro", "solo", "bridge")]
    if len(out) == 1 and out[0][0] == "verse" and len(secs[0][1]) > 4:  # untagged wall of text
        alll, out = secs[0][1][:32], []
        for i in range(0, len(alll), 4):
            out.append(("chorus" if (i // 4) % 2 else "verse", alll[i:i + 4]))
    return out or [("intro", []), ("verse", []), ("chorus", []), ("outro", [])]

def manual_local_spec(opts, manual):
    """Build a full render spec from user-edited title/lyrics/bpm/key — no LLM."""
    genre = opts["genre"]; g = GENRES[genre]
    key = (manual.get("key") or "").strip() or GENRE_KEYS.get(genre, "A minor")
    tonic, scale, mode = parse_key(key)
    prog = PROG_MIN if mode == "minor" else PROG_MAJ
    semis = (tonic - (9 if mode == "minor" else 0)) % 12
    try: bpm = int(str(manual.get("bpm") or "").strip() or 0)
    except Exception: bpm = 0
    if not bpm: bpm = sum(g["bpm"]) // 2
    parsed = parse_tagged_lyrics(manual.get("lyrics", ""))
    if parsed[0][0] != "intro": parsed.insert(0, ("intro", []))
    if parsed[-1][0] != "outro": parsed.append(("outro", []))
    energy = {"intro":4,"verse":6,"prechorus":7,"chorus":9,"bridge":6,"solo":8,"hook":9,"outro":3}
    cont = {"verse":["arc","rise","wave","fall"],"chorus":["hook","rise","hook","fall"],
            "prechorus":["rise","rise","rise","rise"],"bridge":["wave","arc","wave","fall"]}
    secs = []
    for typ, lines in parsed[:10]:
        if typ in ("intro", "outro", "solo"): lines = []
        n = max(1, len(lines))
        bars = 4 if typ in ("intro", "outro") else (4 if n <= 2 else 8 if n <= 4 else 16)
        base = prog.get(typ, prog["verse"])
        chords = [transpose_sym(base[i % 4], semis) for i in range(bars)]
        cc = cont.get(typ, cont["verse"])
        secs.append({"type": typ, "bars": bars, "chords": chords, "lyrics": lines,
                     "contour": [cc[i % 4] for i in range(len(lines))], "energy": energy.get(typ, 6)})
    spec = {"title": (manual.get("title") or "").strip()[:80] or "Untitled",
            "bpm": bpm, "key": key, "sections": secs}
    return validate_spec(spec, genre)

# ----------------------------------------------------------------------------- cloud engine (MiniMax music-2.6)
MINIMAX_ERR = {1002:"rate limit — wait a minute and retry",1004:"auth failed — check MiniMax API key",
               1008:"insufficient MiniMax balance",1026:"content flagged by MiniMax moderation",
               2013:"invalid parameters",2049:"invalid MiniMax API key"}

def draft_brief(opts, progress):
    """LLM song brief: (title, style_prompt, tagged lyrics). Raises on LLM failure."""
    g = GENRES[opts["genre"]]
    mood = MOOD_TAGS.get(opts.get("mood",""), opts.get("mood","")) if opts.get("mood") != "auto" else ""
    voice = {"male":"male vocals","female":"female vocals","instrumental":"instrumental"}.get(opts.get("voice","auto"), "")
    base_style = ", ".join(x for x in (g["tags"], mood, voice, opts.get("tempo","") + " tempo" if opts.get("tempo") not in (None,"auto") else "") if x)
    sys_p = "You are a hit songwriter. Reply with ONLY a single minified JSON object, no markdown."
    user_p = f"""Write an original song brief as JSON: {{"title":str,"style_prompt":str,"lyrics":str}}
- style_prompt: under 280 chars; describe genre, mood, instrumentation, tempo feel, vocal type. Build on: {base_style}
- lyrics: full song using structure tags [Intro] [Verse] [Pre Chorus] [Chorus] [Bridge] [Outro] each on its own line, lyric lines separated by newlines, under 2800 chars total. Choruses repeat the same hook. {'Keep it instrumental: lyrics should be just minimal tags.' if opts.get('voice')=='instrumental' else ''}
Song idea: {opts.get('idea') or 'your choice — surprise me'}"""
    brief = extract_json(call_llm([{"role":"system","content":sys_p},{"role":"user","content":user_p}], progress=progress))
    title = str(brief.get("title","Untitled"))[:80]
    style = str(brief.get("style_prompt", base_style))[:1900]
    lyrics = str(brief.get("lyrics",""))[:3400]
    if not lyrics.strip(): raise ValueError("empty lyrics")
    return title, style, lyrics

def llm_cloud_brief(opts, progress):
    g = GENRES[opts["genre"]]
    mood = MOOD_TAGS.get(opts.get("mood",""), opts.get("mood","")) if opts.get("mood") != "auto" else ""
    voice = {"male":"male vocals","female":"female vocals","instrumental":"instrumental"}.get(opts.get("voice","auto"), "")
    base_style = ", ".join(x for x in (g["tags"], mood, voice, opts.get("tempo","") + " tempo" if opts.get("tempo") not in (None,"auto") else "") if x)
    try:
        title, style, lyrics = draft_brief(opts, progress)
        return title, style, lyrics, False
    except Exception as e:
        progress(f"no LLM lyrics ({e}) — letting MiniMax write them")
        style = (base_style + ". " + (opts.get("idea") or ""))[:1900]
        return "Untitled", style, "", True

def generate_cloud(opts, progress):
    if not CONF.get("minimax_api_key"):
        raise RuntimeError("cloud engine needs a MiniMax API key — set it in settings (free tier works: model music-2.6-free)")
    manual = opts.get("manual") if opts.get("mode") == "manual" else None
    if manual is not None:
        g = GENRES[opts["genre"]]
        title = (manual.get("title") or "").strip()[:80] or "Untitled"
        style = (manual.get("style") or "").strip()[:1900] or g["tags"]
        lyrics = (manual.get("lyrics") or "").strip()[:3400]
        auto_lyrics = not lyrics
        progress("rendering your manual brief")
    else:
        title, style, lyrics, auto_lyrics = llm_cloud_brief(opts, progress)
    payload = {"model": CONF.get("minimax_model", "music-2.6-free"),
               "prompt": style,
               "output_format": "hex",
               "audio_setting": {"sample_rate": 44100, "bitrate": 256000, "format": "mp3"}}
    if opts.get("voice") == "instrumental":
        payload["is_instrumental"] = True
    elif auto_lyrics:
        payload["lyrics_optimizer"] = True
    else:
        payload["lyrics"] = lyrics
    progress("MiniMax is rendering the song (this takes 1-3 min)")
    data = _post_json("https://api.minimax.io/v1/music_generation", payload,
                      {"Authorization": f"Bearer {CONF['minimax_api_key']}"}, timeout=420)
    code = (data.get("base_resp") or {}).get("status_code", -1)
    if code != 0:
        raise RuntimeError(f"MiniMax error {code}: {MINIMAX_ERR.get(code, (data.get('base_resp') or {}).get('status_msg','unknown'))}")
    hexstr = (data.get("data") or {}).get("audio") or ""
    if not hexstr: raise RuntimeError("MiniMax returned no audio")
    audio = bytes.fromhex(hexstr)
    return title, lyrics, audio, style

# ----------------------------------------------------------------------------- hosted engine (free HF Space, no GPU, no key, real vocals)
_HOSTED_OK = {}
def hosted_available():
    """gradio_client present? (the only dep for the hosted free path)"""
    if "v" in _HOSTED_OK: return _HOSTED_OK["v"]
    try:
        import gradio_client  # noqa
        ok = True
    except Exception:
        ok = False
    _HOSTED_OK["v"] = ok
    return ok

def _hosted_pick_endpoint(client):
    """Find the text->music endpoint on whatever ACE-Step Space build is live."""
    try:
        info = client.view_api(return_format="dict") or {}
    except Exception:
        info = {}
    named = (info.get("named_endpoints") or {})
    # prefer endpoints whose name screams generation
    pref = [n for n in named if any(k in n.lower() for k in ("generate", "text2music", "text_to_music", "music", "run", "predict"))]
    order = pref + [n for n in named if n not in pref]
    return order

def generate_hosted(opts, progress):
    if not hosted_available():
        raise RuntimeError("hosted engine needs the gradio-client package — run:  python3 apollo.py --setup-hosted  "
                           "(tiny, ~1 min, no GPU). Then this works with no key.")
    from gradio_client import Client
    if opts.get("genre", "auto") == "auto":
        opts["genre"] = random.choice(list(GENRES))
    title, style, lyrics = neural_brief(opts, progress)
    voice = opts.get("voice", "auto")
    if voice == "instrumental":
        lyrics = "[inst]"
    elif voice in ("male", "female") and "vocal" not in style.lower():
        style = f"{style}, {voice} vocals"
    lyrics = normalize_lyric_tags(lyrics) or "[inst]"
    dur = float(NEURAL_SECONDS.get(opts.get("length", "standard"), 120))
    space = (CONF.get("hosted_space") or "ACE-Step/ACE-Step").strip()
    token = (CONF.get("hf_token") or "").strip() or None
    progress(f"connecting to free neural Space ({space})")
    client = None; cerr = None
    for kwargs in ([{"hf_token": token}] if token else []) + ([{"token": token}] if token else []) + [{}]:
        try:
            client = Client(space, verbose=False, **kwargs); break
        except TypeError:
            try: client = Client(space, **kwargs); break
            except Exception as e: cerr = e
        except Exception as e:
            cerr = e
    if client is None:
        raise RuntimeError(f"couldn't reach the free Space ({cerr}). It may be sleeping/busy — retry in a minute, "
                           f"or use the cloud engine (free MiniMax key) for reliability.")
    eps = _hosted_pick_endpoint(client)
    if not eps:
        raise RuntimeError("the free Space changed its API and no generation endpoint was found. "
                           "Use the cloud engine for now; I can point Apollo at a new Space later.")
    # ACE-Step Space signature: (format, audio_duration, prompt, lyrics, infer_step, guidance_scale, ...)
    attempts = [
        dict(format="wav", audio_duration=dur, prompt=style, lyrics=lyrics,
             infer_step=int(CONF.get("neural_steps", 27)), guidance_scale=float(CONF.get("neural_guidance", 12.0))),
        dict(prompt=style, lyrics=lyrics, audio_duration=dur),
        dict(prompt=style, lyrics=lyrics),
    ]
    last_err = None; result = None
    progress("queued on the free Space — first run can take a few minutes, hang tight")
    for ep in eps[:3]:
        for kw in attempts:
            try:
                job = client.submit(api_name=ep, **kw)
                # stream status while it runs
                while not job.done():
                    try:
                        st = job.status()
                        if getattr(st, "rank", None) is not None:
                            progress(f"queue position {st.rank}" + (f", ~{int(st.eta)}s" if getattr(st, "eta", None) else ""))
                    except Exception: pass
                    time.sleep(2.0)
                result = job.result()
                break
            except TypeError as e:
                last_err = e; continue   # wrong kwargs for this endpoint, try simpler
            except Exception as e:
                last_err = e; continue
        if result is not None: break
    if result is None:
        raise RuntimeError(f"the free Space didn't return audio ({last_err}). It's likely overloaded — "
                           f"retry shortly, or use the cloud engine (fast, free key).")
    # result may be a filepath, a dict with 'value'/'name', or a tuple containing one
    path = _hosted_extract_audio(result)
    if not path or not os.path.isfile(path):
        raise RuntimeError("the free Space returned an unexpected result. Try again, or use the cloud engine.")
    mp3 = maybe_mp3(path) if path.endswith(".wav") else path
    dst = os.path.join(TMP_DIR, f"{opts.get('_jid','hosted')}{os.path.splitext(mp3)[1]}")
    shutil.copyfile(mp3, dst)
    with open(dst, "rb") as f: audio = f.read()
    return title, lyrics, audio, style, os.path.basename(dst)

def _hosted_extract_audio(res):
    """Dig an audio filepath out of whatever shape the Space returned."""
    def from_one(x):
        if isinstance(x, str) and os.path.isfile(x): return x
        if isinstance(x, dict):
            for k in ("name", "value", "path", "audio"):
                v = x.get(k)
                if isinstance(v, str) and os.path.isfile(v): return v
        return None
    p = from_one(res)
    if p: return p
    if isinstance(res, (list, tuple)):
        for item in res:
            p = from_one(item) or (_hosted_extract_audio(item) if isinstance(item, (list, tuple, dict)) else None)
            if p: return p
    return None

# ----------------------------------------------------------------------------- neural engine (ACE-Step 1.5, fully local, real vocals)
_NEURAL_TAG_RX = re.compile(r"\[(intro|verse|pre[\s-]?chorus|chorus|bridge|solo|hook|outro|inst|instrumental|break)\b[^\]]*\]", re.I)

def normalize_lyric_tags(text):
    """ACE-Step wants lowercase [verse]/[chorus]/[inst] tags. Map ours onto its vocabulary."""
    def repl(m):
        t = re.sub(r"[\s-]", "", m.group(1).lower())
        t = {"prechorus": "chorus", "hook": "chorus", "solo": "inst",
             "instrumental": "inst", "break": "inst", "intro": "inst", "outro": "inst"}.get(t, t)
        return "[" + t + "]"
    return _NEURAL_TAG_RX.sub(repl, text or "")

def neural_python_path():
    """The interpreter that has ACE-Step installed: configured venv, else this interpreter."""
    p = (CONF.get("neural_python") or "").strip()
    if p and os.path.isfile(p): return p
    cand = os.path.join(NEURAL_DIR, "bin", "python")
    if os.path.isfile(cand): return cand
    return sys.executable

_NEURAL_OK = {}
def neural_available():
    py = neural_python_path()
    if py in _NEURAL_OK: return _NEURAL_OK[py]
    try:
        r = subprocess.run([py, "-c", "import acestep.pipeline_ace_step"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=40)
        ok = (r.returncode == 0)
    except Exception:
        ok = False
    _NEURAL_OK[py] = ok
    return ok

NEURAL_WORKER = r'''
import sys, json, inspect, os
args = json.load(open(sys.argv[1]))
def log(m): print("NEURAL: " + m, flush=True)
try:
    import torch
except Exception as e:
    print(json.dumps({"ok": False, "error": "torch not installed in neural env: %s" % e})); sys.exit(0)
try:
    from acestep.pipeline_ace_step import ACEStepPipeline
except Exception:
    try:
        from acestep import ACEStepPipeline
    except Exception as e:
        print(json.dumps({"ok": False, "error": "ACE-Step not importable: %s" % e})); sys.exit(0)

if torch.cuda.is_available():
    device, bf16 = "cuda", True
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device, bf16 = "mps", False
else:
    device, bf16 = "cpu", False
log("device=%s" % device)

def filt(fn, want):
    try: ok = set(inspect.signature(fn).parameters)
    except Exception: return want
    return {k: v for k, v in want.items() if k in ok}

ctor = {"checkpoint_path": args.get("checkpoint") or None, "checkpoint_dir": args.get("checkpoint") or None,
        "bf16": bf16, "torch_compile": False, "device_id": 0,
        "cpu_offload": (device == "cuda"), "overlapped_decode": (device == "cuda")}
log("loading model (first run downloads ~3.5GB)…")
pipe = ACEStepPipeline(**filt(ACEStepPipeline.__init__, ctor))

seed = int(args.get("seed", 0))
call = {"audio_duration": float(args.get("duration", 120)), "prompt": args["prompt"], "lyrics": args["lyrics"],
        "infer_step": int(args.get("steps", 27)), "guidance_scale": float(args.get("guidance", 12.0)),
        "save_path": args["out"], "format": "wav",
        "manual_seeds": str(seed), "actual_seeds": [seed], "manual_seed": seed}
call = filt(pipe.__call__, call)
log("generating %ss, %s steps…" % (int(call.get("audio_duration", 0)), call.get("infer_step")))
res = pipe(**call)

out = args["out"]
if not os.path.isfile(out):
    cand = None
    if isinstance(res, str) and os.path.isfile(res): cand = res
    elif isinstance(res, (list, tuple)) and res and isinstance(res[0], str) and os.path.isfile(res[0]): cand = res[0]
    if cand:
        out = cand
    else:
        try:
            import soundfile as sf, numpy as np
            arr = res[0] if isinstance(res, (list, tuple)) else res
            arr = np.asarray(arr)
            sr = getattr(getattr(pipe, "config", None), "sampling_rate", 44100)
            sf.write(out, arr.T if arr.ndim == 2 and arr.shape[0] < arr.shape[1] else arr, int(sr))
        except Exception as e:
            print(json.dumps({"ok": False, "error": "no output produced: %s" % e})); sys.exit(0)
print(json.dumps({"ok": True, "file": out}))
'''

def write_neural_worker():
    os.makedirs(CONF_DIR, exist_ok=True)
    path = os.path.join(CONF_DIR, "neural_worker.py")
    with open(path, "w") as f: f.write(NEURAL_WORKER)
    return path

def neural_brief(opts, progress):
    """(title, style_prompt, lyrics) for the neural engine."""
    g = GENRES[opts["genre"]]
    manual = opts.get("manual") if opts.get("mode") == "manual" else None
    if manual is not None:
        title = (manual.get("title") or "").strip()[:80] or "Untitled"
        style = (manual.get("style") or "").strip()[:600] or g["tags"]
        lyrics = (manual.get("lyrics") or "").strip()
        return title, style, lyrics
    if opts.get("voice") == "instrumental":
        mood = MOOD_TAGS.get(opts.get("mood", ""), "") if opts.get("mood") != "auto" else ""
        style = ", ".join(x for x in (g["tags"], mood, "instrumental") if x)
        return "Untitled", style, "[inst]"
    try:
        return draft_brief(opts, progress)
    except Exception as e:
        progress(f"no LLM ({e}) — using a built-in lyric template")
        spec = template_spec(opts["genre"])
        lyrics = "\n\n".join((f"[{s['type']}]\n" + "\n".join(s["lyrics"])) if s["lyrics"] else f"[{s['type']}]"
                             for s in spec["sections"])
        mood = MOOD_TAGS.get(opts.get("mood", ""), "") if opts.get("mood") != "auto" else ""
        voice = {"male": "male vocals", "female": "female vocals"}.get(opts.get("voice", ""), "")
        style = ", ".join(x for x in (g["tags"], mood, voice) if x)
        return spec["title"], style, lyrics

def generate_neural(opts, progress):
    if not neural_available():
        raise RuntimeError("neural engine isn't installed yet — run:  python3 apollo.py --setup-neural  "
                           "(one-time, downloads ACE-Step + PyTorch). Until then use the cloud or synth engine.")
    if opts.get("genre", "auto") == "auto":
        opts["genre"] = random.choice(list(GENRES))
    title, style, lyrics = neural_brief(opts, progress)
    voice = opts.get("voice", "auto")
    if voice == "instrumental":
        lyrics = "[inst]"
    elif voice in ("male", "female") and "vocal" not in style.lower():
        style = f"{style}, {voice} vocals"
    lyrics = normalize_lyric_tags(lyrics) or "[inst]"
    worker = write_neural_worker()
    out = os.path.join(TMP_DIR, f"{opts.get('_jid','neural')}.wav")
    payload = {"prompt": style[:600], "lyrics": lyrics[:3000], "out": out,
               "duration": NEURAL_SECONDS.get(opts.get("length", "standard"), 120),
               "steps": int(CONF.get("neural_steps", 27)), "guidance": float(CONF.get("neural_guidance", 12.0)),
               "seed": random.randrange(2**31), "checkpoint": (CONF.get("neural_checkpoint") or "").strip()}
    argf = os.path.join(TMP_DIR, f"{opts.get('_jid','neural')}_args.json")
    with open(argf, "w") as f: json.dump(payload, f)
    py = neural_python_path()
    progress("starting ACE-Step (first run downloads the model — be patient)")
    proc = subprocess.Popen([py, worker, argf], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    result = None
    for line in proc.stdout:
        line = line.rstrip()
        if not line: continue
        if line.startswith("NEURAL: "):
            progress(line[8:])
        elif line.startswith("{") and '"ok"' in line:
            try: result = json.loads(line)
            except Exception: pass
        elif any(k in line for k in ("%|", "Download", "Loading", "it/s", "s/it")):
            progress(line[:90])
    proc.wait(timeout=5)
    if not result or not result.get("ok"):
        raise RuntimeError((result or {}).get("error", "neural render failed — check the terminal log"))
    src = result["file"]
    mp3 = maybe_mp3(src) if src.endswith(".wav") else src
    with open(mp3, "rb") as f: audio = f.read()
    return title, lyrics, audio, style, os.path.basename(mp3)

# ----------------------------------------------------------------------------- jobs
JOBS = {}
JOBS_LOCK = threading.Lock()

def slugify(s):
    s = re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-").lower()
    return s[:48] or "song"

def run_job(jid, opts):
    job = JOBS[jid]
    def progress(msg):
        job["stage"] = msg; job["log"].append(f"[{time.strftime('%H:%M:%S')}] {msg}")
    try:
        engine = opts.get("engine", "local")
        opts["_jid"] = jid
        if opts.get("genre", "auto") == "auto":
            opts["genre"] = random.choice(list(GENRES))
        if engine == "hosted":
            title, lyrics, audio, style, fname = generate_hosted(opts, progress)
            with open(os.path.join(TMP_DIR, fname), "wb") as f: f.write(audio)
            job.update(title=title, lyrics=lyrics, file=f"/audio/{fname}", style=style,
                       meta=f"hosted neural · ACE-Step Space · {opts['genre']} · free, no key")
        elif engine == "neural":
            title, lyrics, audio, style, fname = generate_neural(opts, progress)
            with open(os.path.join(TMP_DIR, fname), "wb") as f: f.write(audio)
            job.update(title=title, lyrics=lyrics, file=f"/audio/{fname}", style=style,
                       meta=f"neural · ACE-Step · {opts['genre']} · local model")
        elif engine == "cloud":
            title, lyrics, audio, style = generate_cloud(opts, progress)
            fname = f"{jid}.mp3"
            with open(os.path.join(TMP_DIR, fname), "wb") as f: f.write(audio)
            job.update(title=title, lyrics=lyrics, file=f"/audio/{fname}", style=style,
                       meta=f"cloud · minimax {CONF.get('minimax_model','')}")
        else:
            if opts.get("mode") == "manual" and opts.get("manual"):
                progress("building spec from your manual edit")
                spec = manual_local_spec(opts, opts["manual"]); src = "you"
            else:
                try:
                    spec = llm_local_spec(opts, progress)
                    src = "deepseek" if CONF.get("siliconflow_api_key") else "groq"
                except LLMError as e:
                    progress(f"LLM unavailable ({e}) — using built-in template, set a SiliconFlow key for real AI composition")
                    spec = validate_spec(template_spec(opts["genre"]), opts["genre"])
                    spec["_genre"] = opts["genre"]; src = "template"
            opts["seed"] = random.randrange(2**31)
            L, R, dur = render_local(spec, opts, progress)
            wav = os.path.join(TMP_DIR, f"{jid}.wav")
            progress("writing audio")
            write_wav(wav, L, R)
            out = maybe_mp3(wav)
            lyrics = "\n\n".join(f"[{s['type'].title()}]\n" + "\n".join(s["lyrics"]) if s["lyrics"] else f"[{s['type'].title()}]"
                                 for s in spec["sections"])
            job.update(title=spec["title"], lyrics=lyrics, file=f"/audio/{os.path.basename(out)}",
                       meta=f"local · {spec['_genre']} · {spec['key']} · {spec['bpm']} bpm · {int(dur//60)}:{int(dur%60):02d} · composed by {src}",
                       spec=spec, bpm=spec["bpm"], key=spec["key"])
        job["stage"] = "done"; job["done"] = True
        progress("done")
    except Exception as e:
        job["error"] = str(e); job["done"] = True
        job["log"].append(f"[{time.strftime('%H:%M:%S')}] ERROR: {e}")
        traceback.print_exc()

def start_job(opts):
    jid = hashlib.md5(f"{time.time()}{random.random()}".encode()).hexdigest()[:10]
    JOBS[jid] = {"id": jid, "stage": "queued", "log": [], "done": False, "error": None,
                 "title": None, "lyrics": "", "file": None, "meta": "", "opts": opts}
    threading.Thread(target=run_job, args=(jid, opts), daemon=True).start()
    return jid

# ----------------------------------------------------------------------------- save / library
def save_job(jid):
    job = JOBS.get(jid)
    if not job or not job.get("file"): raise RuntimeError("nothing to save")
    os.makedirs(LIB_DIR, exist_ok=True)
    src = os.path.join(TMP_DIR, os.path.basename(job["file"]))
    ext = os.path.splitext(src)[1]
    base = f"{slugify(job['title'] or 'song')}__{jid}"
    dst = os.path.join(LIB_DIR, base + ext)
    shutil.copy2(src, dst)
    with open(os.path.join(LIB_DIR, base + ".txt"), "w") as f:
        f.write((job["title"] or "Untitled") + "\n" + (job.get("meta") or "") + "\n\n" + (job.get("lyrics") or ""))
    if job.get("spec"):
        with open(os.path.join(LIB_DIR, base + ".json"), "w") as f: json.dump(job["spec"], f, indent=2)
    return dst

def list_library():
    if not os.path.isdir(LIB_DIR): return []
    items = []
    for fn in sorted(os.listdir(LIB_DIR), key=lambda f: os.path.getmtime(os.path.join(LIB_DIR, f)), reverse=True):
        if not fn.lower().endswith((".mp3", ".wav")): continue
        p = os.path.join(LIB_DIR, fn)
        title = fn.rsplit("__", 1)[0].replace("-", " ").title()
        items.append({"file": "/library/" + fn, "name": title, "path": p,
                      "date": time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(p)))})
    return items[:60]

# ----------------------------------------------------------------------------- app shell (launch sting · icon · window)
def render_boot_sting():
    """~3.6s badass launch sting from the synth engine, cached in the config dir."""
    path = os.path.join(CONF_DIR, "boot.wav")
    if os.path.isfile(path): return path
    n = int(3.6 * SR); L = np.zeros(n, np.float32); R = np.zeros(n, np.float32)
    kick, clap = drum("kick"), drum("clap")
    for t in (0.0, 0.5, 1.0, 1.25):                                  # build roll
        place(L, kick, t, .9); place(R, kick, t, .9)
    rise = noise(1.5) * (np.linspace(0, 1, int(1.5 * SR), dtype=np.float32) ** 2)
    rise = fft_filter(rise, hp=900) * .35
    place(L, rise, 0.0, 1); place(R, rise, 0.02, 1)
    chord = [33, 45, 52, 57, 60, 64]                                 # A1 A2 E3 A3 C4 E4
    dur = 2.0
    stack = np.zeros(int(dur * SR), np.float32)
    for m in chord:
        stack += supersaw(midi_hz(m), dur, 11, 5) * (1.0 if m < 50 else .6)
    stack = soft_drive(fft_filter(stack, lp=7000), 2.2) * env_adsr(len(stack), .004, .25, .75, .6)
    duck = np.ones(len(stack), np.float32)                           # sidechain pump
    for t in (0.0, 0.5, 1.0, 1.5):
        i = int(t * SR); j = min(len(duck), i + int(.22 * SR))
        duck[i:j] *= np.linspace(.25, 1, j - i, dtype=np.float32) ** 1.5
    stack *= duck * .5
    place(L, stack, 1.5, 1)
    place(R, stack, 1.512, 1)                                        # haas width
    for t in (1.5, 2.0, 2.5, 3.0):
        place(L, kick, t, 1.0); place(R, kick, t, 1.0)
    place(L, clap, 1.5, .8); place(R, clap, 1.5, .8)
    crash = fft_filter(noise(1.6), hp=2500) * env_adsr(int(1.6 * SR), .002, .4, .0, 1.0) * .4
    place(L, crash, 1.5, 1); place(R, crash, 1.508, 1)
    for i, m in enumerate([57, 60, 64, 69, 72, 76]):                 # gliss into the drop
        tone = osc_saw(midi_hz(m), .14) * env_adsr(int(.14 * SR), .003, .05, .5, .06) * .22
        place(L if i % 2 else R, tone, 1.26 + i * .04, 1)
    boom = osc_sine(55, 1.4) * env_adsr(int(1.4 * SR), .002, .9, .0, .4) * .8
    place(L, boom, 1.5, 1); place(R, boom, 1.5, 1)
    irl, irr = make_reverb_ir(1.4, 11)
    L = L + fft_convolve(L, irl)[:n] * .22
    R = R + fft_convolve(R, irr)[:n] * .22
    peak = max(float(np.abs(L).max()), float(np.abs(R).max()), 1e-9)
    os.makedirs(CONF_DIR, exist_ok=True)
    write_wav(path, L / peak * .9, R / peak * .9)
    return path

def play_boot():
    """Fire the sting through whatever audio player exists. Silent no-op if none."""
    try: path = render_boot_sting()
    except Exception: return
    for player, args in (("paplay", []), ("pw-play", []), ("aplay", ["-q"]),
                         ("ffplay", ["-nodisp", "-autoexit", "-loglevel", "quiet"]), ("play", ["-q"])):
        exe = shutil.which(player)
        if exe:
            try:
                subprocess.Popen([exe, *args, path], stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL, start_new_session=True)
            except Exception: pass
            return

def _png(path, rgba):
    h, w = rgba.shape[:2]
    raw = b"".join(b"\x00" + rgba[y].tobytes() for y in range(h))
    def chunk(tag, data):
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xffffffff)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)))
        f.write(chunk(b"IDAT", zlib.compress(raw, 9)))
        f.write(chunk(b"IEND", b""))

def make_icon(path, size=512):
    """Apollo icon: amber sun + waveform on a dark rounded plate. Pure numpy + zlib."""
    s = size * 2
    yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)
    cx = cy = s / 2
    img = np.zeros((s, s, 4), np.float32)
    r = s * .21
    dx = np.maximum(np.abs(xx - cx) - (s / 2 - r), 0); dy = np.maximum(np.abs(yy - cy) - (s / 2 - r), 0)
    plate = np.clip(r - np.sqrt(dx ** 2 + dy ** 2) + 1, 0, 1)
    grad = (yy / s)[..., None]
    base = (np.array([0x1b, 0x1d, 0x27], np.float32) / 255 * (1 - grad)
            + np.array([0x10, 0x11, 0x18], np.float32) / 255 * grad)
    img[..., :3] = base
    img[..., 3] = plate
    amber = np.array([0xf0, 0xa8, 0x3a], np.float32) / 255
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    sun = np.clip(s * .30 - dist + 1, 0, 1)
    ang = np.arctan2(yy - cy, xx - cx)
    rays = ((np.cos(ang * 8) > .92) & (dist > s * .345) & (dist < s * .43)).astype(np.float32)
    glyph = np.clip(sun + rays, 0, 1) * plate
    wave_y = cy + s * .10 * np.sin((xx - cx) / (s * .30) * np.pi * 2.2)
    glyph[(np.abs(yy - wave_y) < s * .028) & (dist < s * .305)] = 0
    for c in range(3):
        img[..., c] = img[..., c] * (1 - glyph) + amber[c] * glyph
    img = img.reshape(size, 2, size, 2, 4).mean(axis=(1, 3))
    _png(path, (np.clip(img, 0, 1) * 255 + .5).astype(np.uint8))
    return path

def open_window(url):
    """Open Apollo as an app window (chromium app mode), falling back down the browser chain."""
    prof = os.path.join(CONF_DIR, "appwin")
    for b in ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable", "brave-browser", "microsoft-edge"):
        exe = shutil.which(b)
        if exe:
            subprocess.Popen([exe, f"--app={url}", "--window-size=860,1100", f"--user-data-dir={prof}",
                              "--no-first-run", "--no-default-browser-check"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            return f"app window ({b})"
    for b in ("epiphany", "firefox-esr", "firefox"):
        exe = shutil.which(b)
        if exe:
            subprocess.Popen([exe, url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            return b
    import webbrowser; webbrowser.open(url)
    return "default browser"

def install_desktop():
    """Write the icon + .desktop entry so Apollo shows up in the app launcher."""
    me = os.path.abspath(__file__)
    icon_dir = os.path.expanduser("~/.local/share/icons/hicolor/512x512/apps")
    os.makedirs(icon_dir, exist_ok=True)
    icon = make_icon(os.path.join(icon_dir, "apollo.png"))
    app_dir = os.path.expanduser("~/.local/share/applications")
    os.makedirs(app_dir, exist_ok=True)
    desk = os.path.join(app_dir, "apollo.desktop")
    with open(desk, "w") as f:
        f.write("[Desktop Entry]\nType=Application\nName=Apollo\nComment=AI song forge\n"
                f"Exec={shutil.which('python3') or 'python3'} {me}\nIcon={icon}\nTerminal=false\n"
                "Categories=AudioVideo;Audio;Music;\nStartupNotify=false\n")
    for cmd in (["update-desktop-database", app_dir],
                ["gtk-update-icon-cache", "-q", os.path.expanduser("~/.local/share/icons/hicolor")]):
        try: subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
        except Exception: pass
    print(f"[apollo] icon → {icon}")
    print(f"[apollo] launcher → {desk}")
    return desk

# ----------------------------------------------------------------------------- http server
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

def masked(k): return (k[:4] + "…" + k[-4:]) if len(k) > 10 else ("set" if k else "")

class Handler(BaseHTTPRequestHandler):
    server_version = "Apollo/" + VERSION

    def log_message(self, fmt, *args): pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _file(self, root, name):
        path = os.path.realpath(os.path.join(root, os.path.basename(name)))
        if not path.startswith(os.path.realpath(root)) or not os.path.isfile(path):
            return self._send(404, {"error": "not found"})
        ctype = "audio/mpeg" if path.endswith(".mp3") else "audio/wav"
        with open(path, "rb") as f: data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        try:
            path, _, query = self.path.partition("?")
            q = dict(p.split("=", 1) for p in query.split("&") if "=" in p)
            if path == "/": return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            if path == "/api/config":
                return self._send(200, {
                    "siliconflow": masked(CONF["siliconflow_api_key"]), "minimax": masked(CONF["minimax_api_key"]),
                    "groq": masked(CONF["groq_api_key"]), "sf_model": CONF["siliconflow_model"],
                    "mm_model": CONF["minimax_model"], "espeak": bool(ESPEAK), "lib": LIB_DIR,
                    "neural": neural_available(), "neural_python": neural_python_path(),
                    "neural_steps": CONF.get("neural_steps", 27), "hosted": hosted_available()})
            if path == "/api/status":
                job = JOBS.get(q.get("id", ""))
                if not job: return self._send(404, {"error": "unknown job"})
                out = {k: job[k] for k in ("id","stage","log","done","error","title","lyrics","file","meta")}
                out.update({k: job[k] for k in ("style","bpm","key") if k in job})
                return self._send(200, out)
            if path == "/api/library": return self._send(200, {"items": list_library()})
            if path.startswith("/audio/"): return self._file(TMP_DIR, path[7:])
            if path.startswith("/library/"): return self._file(LIB_DIR, path[9:])
            return self._send(404, {"error": "not found"})
        except (BrokenPipeError, ConnectionResetError): pass
        except Exception as e:
            try: self._send(500, {"error": str(e)})
            except Exception: pass

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0) or 0)
            body = json.loads(self.rfile.read(n).decode() or "{}")
            if self.path == "/api/generate":
                opts = {k: str(body.get(k, "auto")).lower() for k in ("engine","genre","mood","tempo","voice","length")}
                opts["idea"] = str(body.get("idea", ""))[:1200]
                if opts["genre"] not in GENRES and opts["genre"] != "auto": opts["genre"] = "auto"
                opts["mode"] = str(body.get("mode", "auto")).lower()
                man = body.get("manual")
                if opts["mode"] == "manual" and isinstance(man, dict):
                    opts["manual"] = {"title": str(man.get("title", ""))[:120], "style": str(man.get("style", ""))[:1900],
                                      "lyrics": str(man.get("lyrics", ""))[:3400], "bpm": str(man.get("bpm", ""))[:6],
                                      "key": str(man.get("key", ""))[:24]}
                return self._send(200, {"id": start_job(opts)})
            if self.path == "/api/draft":
                opts = {k: str(body.get(k, "auto")).lower() for k in ("engine","genre","mood","tempo","voice","length")}
                opts["idea"] = str(body.get("idea", ""))[:1200]
                if opts["genre"] not in GENRES: opts["genre"] = random.choice(list(GENRES))
                title, style, lyrics = draft_brief(opts, lambda m: None)
                return self._send(200, {"title": title, "style": style, "lyrics": lyrics})
            if self.path == "/api/save":
                return self._send(200, {"path": save_job(str(body.get("id", "")))})
            if self.path == "/api/config":
                for k in ("siliconflow_api_key","minimax_api_key","groq_api_key","siliconflow_model","minimax_model","neural_python","neural_checkpoint","hf_token","hosted_space"):
                    v = body.get(k)
                    if isinstance(v, str) and v.strip() and "…" not in v: CONF[k] = v.strip()
                if isinstance(body.get("neural_steps"), (int, float)): CONF["neural_steps"] = int(body["neural_steps"])
                _NEURAL_OK.clear(); _HOSTED_OK.clear()
                save_conf(CONF)
                return self._send(200, {"ok": True})
            return self._send(404, {"error": "not found"})
        except Exception as e:
            try: self._send(500, {"error": str(e)})
            except Exception: pass

# ----------------------------------------------------------------------------- UI
PAGE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Apollo — song forge</title>
<style>
:root{
  --bg:#14151c; --panel:#1b1d27; --panel2:#21232f; --line:#2c2f3d;
  --ink:#eae5d6; --dim:#8e92a6; --amber:#f0a83a; --amber2:#ffd07a; --red:#d96a6a;
  --mono:ui-monospace,'JetBrains Mono',Menlo,Consolas,monospace;
  --sans:system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;
}
*{box-sizing:border-box;margin:0}
body{background:var(--bg);color:var(--ink);font:15px/1.5 var(--sans);padding:0 16px 64px}
.wrap{max-width:720px;margin:0 auto}
header{display:flex;align-items:baseline;gap:12px;padding:22px 0 10px}
.mark{font:700 13px var(--mono);letter-spacing:.34em;color:var(--amber)}
.sub{font:11px var(--mono);letter-spacing:.18em;color:var(--dim)}
header button{margin-left:auto;background:none;border:1px solid var(--line);color:var(--dim);
  font:11px var(--mono);letter-spacing:.12em;padding:6px 12px;border-radius:8px;cursor:pointer}
header button:hover{color:var(--ink);border-color:var(--dim)}
#vu{display:block;width:100%;height:54px;border:1px solid var(--line);border-radius:10px;background:var(--panel)}
.lbl{font:10px var(--mono);letter-spacing:.26em;color:var(--dim);margin:20px 0 8px}
textarea{width:100%;background:var(--panel);border:1px solid var(--line);border-radius:10px;color:var(--ink);
  font:15px var(--sans);padding:12px 14px;min-height:74px;resize:vertical}
textarea:focus,input:focus,button:focus-visible{outline:2px solid var(--amber);outline-offset:1px}
.chips{display:flex;flex-wrap:wrap;gap:8px}
.chip{background:var(--panel);border:1px solid var(--line);color:var(--dim);border-radius:999px;
  font:11px var(--mono);letter-spacing:.08em;padding:7px 13px;cursor:pointer;text-transform:uppercase}
.chip.on{background:var(--amber);border-color:var(--amber);color:#161204;font-weight:700}
.chip.off{display:none}
#manual{display:none}
#manual.show{display:block}
.mf{width:100%;background:var(--panel);border:1px solid var(--line);border-radius:10px;color:var(--ink);
  font:14px var(--sans);padding:11px 13px}
#rowBK{display:flex;gap:10px}
#rowBK .mf{flex:1;min-width:140px;font-family:var(--mono);font-size:13px}
.dimnote{letter-spacing:0;text-transform:none;color:var(--dim)}
#go{width:100%;margin-top:26px;padding:15px;border:none;border-radius:12px;cursor:pointer;
  background:linear-gradient(180deg,var(--amber2),var(--amber));color:#161204;
  font:800 16px var(--sans);letter-spacing:.06em}
#go:disabled{filter:grayscale(.7) brightness(.7);cursor:wait}
#status{font:12px var(--mono);color:var(--dim);background:var(--panel);border:1px solid var(--line);
  border-radius:10px;padding:10px 13px;margin-top:12px;min-height:38px;white-space:pre-wrap;display:none}
#status.show{display:block}
#status .err{color:var(--red)}
.card{display:none;margin-top:26px;background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:18px}
.card.show{display:block}
.card h2{font:800 26px/1.15 var(--sans);letter-spacing:-.01em}
.meta{font:11px var(--mono);color:var(--dim);letter-spacing:.06em;margin:6px 0 14px}
audio{width:100%;margin-bottom:14px}
pre{font:12.5px/1.7 var(--mono);color:var(--ink);background:var(--panel2);border-radius:10px;
  padding:14px;overflow-x:auto;white-space:pre-wrap;margin-bottom:14px}
.row{display:flex;gap:10px;flex-wrap:wrap}
.btn{flex:1;min-width:120px;padding:11px;border-radius:10px;border:1px solid var(--line);cursor:pointer;
  background:var(--panel2);color:var(--ink);font:700 13px var(--sans)}
.btn.amber{background:var(--amber);border-color:var(--amber);color:#161204}
.btn:hover{border-color:var(--dim)}
.saved{font:11px var(--mono);color:var(--amber2);margin-top:10px;word-break:break-all;display:none}
#library .item{display:flex;align-items:center;gap:12px;padding:10px 4px;border-bottom:1px solid var(--line)}
#library .item a{color:var(--amber);text-decoration:none;font:700 14px var(--sans)}
#library .item span{font:10px var(--mono);color:var(--dim);margin-left:auto;text-align:right}
#drawer{display:none;margin-top:14px;background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px}
#drawer.show{display:block}
#drawer label{display:block;font:10px var(--mono);letter-spacing:.2em;color:var(--dim);margin:12px 0 4px}
#drawer input{width:100%;background:var(--panel2);border:1px solid var(--line);border-radius:8px;
  color:var(--ink);font:13px var(--mono);padding:9px 11px}
#drawer .hint{font:11px var(--mono);color:var(--dim);margin-top:10px;line-height:1.7}
@media (prefers-reduced-motion: reduce){ #vu{display:none} }
</style></head><body><div class="wrap">
<header>
  <span class="mark">APOLLO</span><span class="sub">song forge</span>
  <button id="gear" aria-label="settings">API KEYS</button>
</header>
<canvas id="vu" width="1400" height="108" aria-hidden="true"></canvas>

<div id="drawer">
  <label>SILICONFLOW API KEY — composes structure + lyrics (DeepSeek)</label><input id="k_sf" autocomplete="off" placeholder="sk-…">
  <label>MINIMAX API KEY — cloud engine, real sung vocals (free tier ok)</label><input id="k_mm" autocomplete="off" placeholder="…">
  <label>GROQ API KEY — fallback only, used if SiliconFlow fails</label><input id="k_gq" autocomplete="off" placeholder="gsk_…">
  <label>SILICONFLOW MODEL</label><input id="m_sf">
  <label>MINIMAX MODEL (music-2.6-free / music-2.6)</label><input id="m_mm">
  <div class="row" style="margin-top:14px"><button class="btn amber" id="saveKeys">Save keys</button></div>
  <div class="hint" id="confHint"></div>
</div>

<div class="lbl">THE IDEA</div>
<textarea id="idea" placeholder="Describe the song — story, vibe, anything. Or leave blank and roll the dice."></textarea>

<div class="lbl">ENGINE</div><div class="chips" id="g_engine"></div>
<div id="engineNote" style="font:11px var(--mono);color:var(--dim);margin-top:8px;line-height:1.6"></div>
<div class="lbl">MODE</div><div class="chips" id="g_mode"></div>
<div class="lbl">GENRE</div><div class="chips" id="g_genre"></div>
<div class="lbl">MOOD</div><div class="chips" id="g_mood"></div>
<div class="lbl">TEMPO</div><div class="chips" id="g_tempo"></div>
<div class="lbl">VOICE</div><div class="chips" id="g_voice"></div>
<div class="lbl">LENGTH</div><div class="chips" id="g_length"></div>

<div id="manual">
  <div class="lbl">TITLE</div>
  <input class="mf" id="mTitle" autocomplete="off" placeholder="Untitled">
  <div class="lbl" id="lblStyle">STYLE PROMPT <span class="dimnote">— what MiniMax hears</span></div>
  <textarea id="mStyle" placeholder="genre, mood, instrumentation, tempo feel, vocal type"></textarea>
  <div class="lbl">LYRICS <span class="dimnote">— tag sections: [Intro] [Verse] [Pre Chorus] [Chorus] [Bridge] [Outro]</span></div>
  <textarea id="mLyrics" style="min-height:210px" placeholder="[Verse]
first line here
second line here

[Chorus]
the big hook"></textarea>
  <div class="lbl" id="lblBK">BPM / KEY <span class="dimnote">— blank = genre default</span></div>
  <div id="rowBK">
    <input class="mf" id="mBpm" autocomplete="off" inputmode="numeric" placeholder="BPM e.g. 120">
    <input class="mf" id="mKey" autocomplete="off" placeholder="KEY e.g. A minor">
  </div>
  <button class="btn" id="draft" style="width:100%;margin-top:14px">✦ DRAFT WITH AI — THEN EDIT</button>
</div>

<button id="go">GENERATE</button>
<div id="status"></div>

<div class="card" id="card">
  <h2 id="title"></h2><div class="meta" id="meta"></div>
  <audio id="player" controls crossorigin="anonymous"></audio>
  <pre id="lyrics"></pre>
  <div class="row">
    <button class="btn amber" id="save">Save to library</button>
    <a class="btn" id="dl" style="text-align:center;text-decoration:none" download>Download</a>
    <button class="btn" id="edit">Edit &amp; re-render</button>
    <button class="btn" id="again">Regenerate</button>
  </div>
  <div class="saved" id="savedPath"></div>
</div>

<div class="lbl">LIBRARY <span id="libPath" style="letter-spacing:0;text-transform:none"></span></div>
<div id="library"><div style="font:12px var(--mono);color:var(--dim)">Nothing saved yet.</div></div>
</div>
<script>
const GROUPS={
 engine:[["hosted","HOSTED · FREE AI VOCALS"],["cloud","CLOUD · REAL VOCALS"],["neural","NEURAL · LOCAL (GPU)"],["local","SYNTH · OFFLINE"]],
 mode:[["auto","AUTO"],["manual","MANUAL · EDIT"]],
 genre:[["auto","AUTO"],["synthwave","SYNTHWAVE"],["pop","POP"],["rock","ROCK"],["hiphop","HIP-HOP"],["edm","EDM"],["lofi","LO-FI"],["metal","METAL"],["folk","FOLK"]],
 mood:[["auto","AUTO"],["dark","DARK"],["upbeat","UPBEAT"],["melancholy","MELANCHOLY"],["aggressive","AGGRESSIVE"],["chill","CHILL"],["epic","EPIC"],["romantic","ROMANTIC"]],
 tempo:[["auto","AUTO"],["slow","SLOW"],["mid","MID"],["fast","FAST"]],
 voice:[["auto","AUTO"],["male","MALE"],["female","FEMALE"],["instrumental","INSTRUMENTAL"],["croak","CROAK*"],["whisper","WHISPER*"]],
 length:[["standard","STANDARD"],["short","SHORT"],["full","FULL"]]};
const sel={engine:"local",mode:"auto",genre:"auto",mood:"auto",tempo:"auto",voice:"auto",length:"standard"};
const $=id=>document.getElementById(id);
for(const [grp,items] of Object.entries(GROUPS)){
  const box=$("g_"+grp);
  for(const [val,lab] of items){
    const b=document.createElement("button");
    b.className="chip"+(sel[grp]===val?" on":""); b.textContent=lab; b.dataset.v=val;
    b.onclick=()=>{sel[grp]=val;[...box.children].forEach(c=>c.classList.toggle("on",c===b));syncVoice();};
    box.appendChild(b);
  }
}
function syncVoice(){
  const localOnly=["croak","whisper"];
  [...$("g_voice").children].forEach(c=>{
    const hide=sel.engine==="cloud"&&localOnly.includes(c.dataset.v);
    c.classList.toggle("off",hide);
    if(hide&&sel.voice===c.dataset.v){sel.voice="auto";[...$("g_voice").children].forEach(x=>x.classList.toggle("on",x.dataset.v==="auto"));}
  });
  const man=sel.mode==="manual", synth=sel.engine==="local", styled=sel.engine!=="local";
  $("manual").classList.toggle("show",man);
  $("mStyle").style.display=styled?"":"none"; $("lblStyle").style.display=styled?"":"none";
  $("rowBK").style.display=synth?"flex":"none"; $("lblBK").style.display=synth?"":"none";
  $("go").textContent=man?"RENDER MY EDIT":"GENERATE";
  const note=$("engineNote");
  if(sel.engine==="hosted")note.textContent=window.__hosted?"✓ free AI vocals on a hosted GPU — no key, no local GPU needed. First run can take a few min (shared queue).":"⚠ one-time tiny setup:  python3 apollo.py --setup-hosted  — then free real AI vocals, no key, no GPU.";
  else if(sel.engine==="cloud")note.textContent="needs a free MiniMax key (gear). Real sung vocals, fast + reliable.";
  else if(sel.engine==="neural")note.textContent=window.__neural?"✓ local AI vocals — private + offline. SLOW without a GPU (minutes/song on a laptop).":"⚠ local model:  python3 apollo.py --setup-neural  — best on a GPU machine, slow on laptops.";
  else note.textContent="offline, instant, no key — synth music + robotic vocals. Always works.";
}
// ---- config
async function loadConf(){
  const c=await (await fetch("/api/config")).json();
  window.__neural=!!c.neural; window.__hosted=!!c.hosted;
  $("k_sf").placeholder=c.siliconflow||"sk-…"; $("k_mm").placeholder=c.minimax||"…"; $("k_gq").placeholder=c.groq||"gsk_…";
  $("m_sf").value=c.sf_model; $("m_mm").value=c.mm_model; $("libPath").textContent="· "+c.lib;
  $("confHint").innerHTML=(c.hosted?"Hosted engine: <b>ready</b> (free AI vocals, no GPU/key needed). ":"Hosted engine: run <b>python3 apollo.py --setup-hosted</b> for free AI vocals — no GPU, no key. ")
    +(c.minimax?"":"No MiniMax key: cloud engine off. ")+(c.siliconflow?"":"No SiliconFlow key: lyrics use a built-in template. ")
    +(c.espeak?"":"espeak-ng missing — synth songs are instrumental.");
  if(!localStorage.getItem("apollo_engine")){
    const def=c.hosted?"hosted":(c.minimax?"cloud":(c.neural?"neural":"local"));
    sel.engine=def;[...$("g_engine").children].forEach(x=>x.classList.toggle("on",x.dataset.v===def));
  }
  syncVoice();
}
$("gear").onclick=()=>$("drawer").classList.toggle("show");
$("saveKeys").onclick=async()=>{
  await fetch("/api/config",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({
    siliconflow_api_key:$("k_sf").value,minimax_api_key:$("k_mm").value,groq_api_key:$("k_gq").value,
    siliconflow_model:$("m_sf").value,minimax_model:$("m_mm").value})});
  $("k_sf").value=$("k_mm").value=$("k_gq").value="";
  loadConf(); status("keys saved");
};
// ---- status + generate
let curId=null,poll=null,lastJob=null;
function status(msg,err){const s=$("status");s.classList.add("show");
  s.innerHTML+=(err?'<span class="err">':"")+msg.replace(/</g,"&lt;")+(err?"</span>":"")+"\n";s.scrollTop=s.scrollHeight;}
function resetStatus(){$("status").innerHTML="";$("status").classList.add("show");}
async function generate(){
  $("go").disabled=true; $("card").classList.remove("show"); $("savedPath").style.display="none";
  resetStatus(); status("▸ queued");
  localStorage.setItem("apollo_engine",sel.engine);
  const body={...sel,idea:$("idea").value};
  if(sel.mode==="manual")body.manual={title:$("mTitle").value,style:$("mStyle").value,
    lyrics:$("mLyrics").value,bpm:$("mBpm").value,key:$("mKey").value};
  try{
    const r=await(await fetch("/api/generate",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)})).json();
    curId=r.id; let lastLog=0; vuMode="gen";
    poll=setInterval(async()=>{
      const j=await(await fetch("/api/status?id="+curId)).json();
      for(;lastLog<j.log.length;lastLog++)status(j.log[lastLog],j.log[lastLog].includes("ERROR"));
      if(j.done){clearInterval(poll);$("go").disabled=false;vuMode="idle";
        if(j.error){status("✗ "+j.error,true);return;}
        lastJob=j;
        $("title").textContent=j.title||"Untitled"; $("meta").textContent=j.meta||"";
        $("lyrics").textContent=j.lyrics||"(instrumental)";
        $("player").src=j.file; $("dl").href=j.file;
        $("dl").download=(j.title||"song").replace(/[^A-Za-z0-9]+/g,"-")+j.file.slice(j.file.lastIndexOf("."));
        $("card").classList.add("show"); hookVU(); $("player").play().catch(()=>{});
      }
    },900);
  }catch(e){status("✗ "+e,true);$("go").disabled=false;vuMode="idle";}
}
$("go").onclick=generate; $("again").onclick=generate;
$("draft").onclick=async()=>{
  $("draft").disabled=true; resetStatus(); status("▸ drafting with AI…");
  try{
    const r=await(await fetch("/api/draft",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({...sel,idea:$("idea").value})})).json();
    if(r.error)throw r.error;
    $("mTitle").value=r.title||""; $("mStyle").value=r.style||""; $("mLyrics").value=r.lyrics||"";
    status("✓ draft ready — edit anything, then hit RENDER MY EDIT");
  }catch(e){status("✗ "+e,true);}
  $("draft").disabled=false;
};
$("edit").onclick=()=>{
  if(!lastJob)return;
  sel.mode="manual";
  [...$("g_mode").children].forEach(c=>c.classList.toggle("on",c.dataset.v==="manual"));
  $("mTitle").value=lastJob.title||""; $("mLyrics").value=lastJob.lyrics||"";
  if(lastJob.style)$("mStyle").value=lastJob.style;
  if(lastJob.bpm)$("mBpm").value=lastJob.bpm;
  if(lastJob.key)$("mKey").value=lastJob.key;
  syncVoice();
  $("manual").scrollIntoView({behavior:"smooth"});
};
$("save").onclick=async()=>{
  const r=await(await fetch("/api/save",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({id:curId})})).json();
  if(r.path){const p=$("savedPath");p.style.display="block";p.textContent="saved → "+r.path;loadLib();}
  else status("✗ save failed: "+(r.error||"?"),true);
};
async function loadLib(){
  const d=await(await fetch("/api/library")).json();
  if(!d.items.length)return;
  $("library").innerHTML=d.items.map(i=>`<div class="item"><a href="${i.file}" target="_blank">${i.name}</a><span>${i.date}</span></div>`).join("");
}
// ---- VU canvas (the one flashy thing)
const cv=$("vu"),cx=cv.getContext("2d");let vuMode="idle",analyser=null,acx=null,hooked=false;
function hookVU(){
  if(hooked||window.matchMedia("(prefers-reduced-motion: reduce)").matches)return;
  try{acx=new (window.AudioContext||window.webkitAudioContext)();
    const srcNode=acx.createMediaElementSource($("player"));
    analyser=acx.createAnalyser();analyser.fftSize=256;
    srcNode.connect(analyser);analyser.connect(acx.destination);hooked=true;
  }catch(e){}
}
const BARS=56,data=new Uint8Array(128);
function draw(ts){
  const w=cv.width,h=cv.height;cx.clearRect(0,0,w,h);
  const bw=w/BARS;
  for(let i=0;i<BARS;i++){
    let v;
    if(analyser&&!$("player").paused){analyser.getByteFrequencyData(data);v=data[Math.floor(i*data.length/BARS)]/255;}
    else if(vuMode==="gen")v=.18+.5*Math.abs(Math.sin(ts/180+i*.6))*Math.random();
    else v=.10+.08*Math.sin(ts/900+i*.33);
    const bh=Math.max(3,v*h*.86);
    const grad=cx.createLinearGradient(0,h,0,h-bh);
    grad.addColorStop(0,"#f0a83a");grad.addColorStop(1,"#ffd07a");
    cx.fillStyle=grad;cx.globalAlpha=.92;
    cx.fillRect(i*bw+1.5,h-bh,bw-3,bh);
  }
  requestAnimationFrame(draw);
}
if(!window.matchMedia("(prefers-reduced-motion: reduce)").matches)requestAnimationFrame(draw);
$("player").addEventListener("play",()=>{hookVU();acx&&acx.resume();});
loadConf();loadLib();
</script></body></html>"""

def setup_hosted():
    """Tiny one-time install: gradio-client, for the free no-GPU hosted neural engine."""
    print("[apollo] installing gradio-client for the free hosted neural engine (no GPU, no key)…")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "gradio_client"])
    except subprocess.CalledProcessError:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--break-system-packages", "gradio_client"])
        except subprocess.CalledProcessError as e:
            print(f"[apollo] install failed: {e}\n[apollo] try:  pip install gradio_client")
            return
    _HOSTED_OK.clear()
    print(f"[apollo] hosted engine {'READY' if hosted_available() else 'installed — restart Apollo'}.")
    print("[apollo] launch Apollo, pick the HOSTED engine. Real AI vocals, free, runs on the Space's GPU.")

def setup_neural():
    """One-time: build an isolated venv with PyTorch + ACE-Step, wire it into config."""
    import venv as _venv
    print("[apollo] setting up the neural engine (ACE-Step). This is a big one-time install.\n")
    has_nv = shutil.which("nvidia-smi") is not None
    print(f"[apollo] GPU detected: {'NVIDIA CUDA' if has_nv else 'none — will install CPU PyTorch (songs render slowly)'}")
    print(f"[apollo] venv: {NEURAL_DIR}")
    os.makedirs(os.path.dirname(NEURAL_DIR), exist_ok=True)
    if not os.path.isfile(os.path.join(NEURAL_DIR, "bin", "python")):
        print("[apollo] creating venv…")
        _venv.EnvBuilder(with_pip=True).create(NEURAL_DIR)
    py = os.path.join(NEURAL_DIR, "bin", "python")
    def run(cmd):
        print("  $ " + " ".join(cmd)); subprocess.check_call(cmd)
    try:
        run([py, "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools"])
        if has_nv:
            run([py, "-m", "pip", "install", "torch", "torchaudio"])
        else:
            run([py, "-m", "pip", "install", "torch", "torchaudio",
                 "--index-url", "https://download.pytorch.org/whl/cpu"])
        run([py, "-m", "pip", "install", "soundfile", "librosa"])
        run([py, "-m", "pip", "install", "git+https://github.com/ace-step/ACE-Step.git"])
    except subprocess.CalledProcessError as e:
        print(f"\n[apollo] install failed: {e}\n[apollo] you can retry, or open an issue. The cloud/synth engines still work.")
        return
    CONF["neural_python"] = py
    save_conf(CONF)
    _NEURAL_OK.clear()
    ok = neural_available()
    print(f"\n[apollo] neural engine {'READY' if ok else 'installed but not importable — check the log above'}.")
    print("[apollo] first song will download the ACE-Step model (~3.5GB) to ~/.cache/ace-step. Launch Apollo and pick the NEURAL engine.")

# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Apollo — AI song forge")
    ap.add_argument("--port", type=int, default=int(CONF.get("port", 8585)))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--demo", action="store_true", help="render the built-in template offline to ./apollo_demo.wav and exit")
    ap.add_argument("--no-vocals", action="store_true")
    ap.add_argument("--no-window", action="store_true", help="don't open the app window")
    ap.add_argument("--no-sound", action="store_true", help="skip the launch sting")
    ap.add_argument("--install-desktop", action="store_true", help="write icon + .desktop launcher and exit")
    ap.add_argument("--setup-neural", action="store_true", help="install the local ACE-Step neural engine (free, real vocals, needs GPU to be fast) and exit")
    ap.add_argument("--setup-hosted", action="store_true", help="install gradio-client for the free hosted neural engine (no GPU, no key) and exit")
    args = ap.parse_args()

    if args.setup_hosted:
        setup_hosted()
        return

    if args.setup_neural:
        setup_neural()
        return

    if args.install_desktop:
        install_desktop()
        return

    if args.demo:
        spec = validate_spec(template_spec("synthwave"), "synthwave"); spec["_genre"] = "synthwave"
        t0 = time.time()
        L, R, dur = render_local(spec, {"voice": "male", "seed": 42, "no_vocals": args.no_vocals},
                                 progress=lambda s: print("  ·", s))
        write_wav("apollo_demo.wav", L, R)
        print(f"[apollo] demo rendered: apollo_demo.wav  ({dur:.1f}s of audio in {time.time()-t0:.1f}s)")
        return

    if not ESPEAK:
        print("[apollo] espeak-ng not found — local engine will be instrumental. Fix: sudo apt install espeak-ng")
    print(f"[apollo] v{VERSION} · library: {LIB_DIR}")
    print(f"[apollo] listening on http://{args.host}:{args.port}")
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '::') else args.host}:{args.port}"
    def boot():
        if not args.no_sound:
            try: play_boot()
            except Exception: pass
        if not args.no_window:
            try: print(f"[apollo] opening {open_window(url)}")
            except Exception: pass
    threading.Thread(target=boot, daemon=True).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[apollo] out.")

if __name__ == "__main__":
    main()
