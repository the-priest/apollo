# Apollo — AI song forge

Idea in → full song out (music + vocals) → one-click save. Or load your own MP3 and sing new lyrics over it. Runs as a desktop app: own window, app-menu icon, launch sting. Free, no API key required.

![engines: VOICE (Kokoro) · SYNTH (offline)]

## Install

```
curl -fsSL https://raw.githubusercontent.com/the-priest/apollo/main/install.sh | bash
```

Launch **Apollo** from your app menu. To also install the free AI singer in one go:

```
curl -fsSL https://raw.githubusercontent.com/the-priest/apollo/main/install.sh | bash -s -- --voice
```

## Two engines

**VOICE — free real AI singer (Kokoro).** A small open-source neural voice model (Apache-2.0) that runs on your CPU — no API key, no GPU, no account. Sings your lyrics over the generated music. One-time setup downloads the model (~310MB):

```
apollo --setup-voice
```

After that it's fully offline. First song in a session takes ~10–20s extra to load the model; then it's quick. Voices: male/female, US/UK.

**SYNTH — instant offline.** Pure numpy synthesis + a robotic espeak voice. No setup, no model, renders in seconds. The voice is robotic, but the music is warm and it always works with zero extra dependencies. Good for quick sketches or before the voice model is installed.

Lyrics/structure are written by SiliconFlow/DeepSeek if you add a key (gear menu); otherwise a built-in template is used. Groq is fallback only.

## Features

**Your own MP3 as a backing track.** Hit **LOAD MP3 / AUDIO**, drop in any audio file (mp3/wav/m4a/ogg/flac). Apollo decodes it, detects the tempo, and sings your lyrics over it. Set your words in MANUAL mode (or type an idea) and GENERATE.

**Vocal style — SUNG or SPOKEN.** SUNG retunes the voice toward the melody. SPOKEN reads your lyrics plainly — the clearest, most intelligible option.

**Genre-accurate.** Each genre renders at its true tempo and tone: lo-fi (~78bpm, warm/dusty), hip-hop (~87), synthwave (~109), pop (~113), EDM (~128, bright), rock (~137), metal (~165). 

**Auto vs Manual.** AUTO writes everything from your idea. MANUAL lets you set your own title/lyrics/BPM/key, tag sections (`[Intro] [Verse] [Chorus] [Bridge] [Outro]`), draft with AI then edit, and re-render any finished song.

## Use

Pick ENGINE / MODE / genre / mood / tempo / voice / length / vocal-style, type the idea (or load a track + set lyrics), GENERATE. Watch the log, play it, **QUIT** (top-right) stops the server cleanly. SAVE drops mp3 + lyrics .txt + spec .json into `~/Music/Apollo`. Library list replays saved songs.

Optional keys go in the gear drawer (chmod-600 `~/.config/apollo/config.json`) or env: `SILICONFLOW_API_KEY`, `GROQ_API_KEY`.

## Flags

```
apollo
```
```
apollo --setup-voice
```
```
apollo --install-desktop
```
```
apollo --demo
```
```
apollo --no-sound
```
```
apollo --no-window
```

`--demo` renders an offline song to ./apollo_demo.wav (optional genre, e.g. `--demo lofi`). `--host`/`--port` to move it (auto-finds a free port and won't start a second copy if one's already running).

## Requirements

- Linux (tested on Kali/Debian/Ubuntu), Python 3.10+
- `espeak-ng`, `python3-numpy`, `ffmpeg` (the installer handles these)
- For the VOICE engine: `kokoro-onnx` + model files (the `--setup-voice` command handles these)

## Genres

synthwave · pop · rock · hiphop · edm · lofi · metal · folk · auto

## License

MIT — see [LICENSE](LICENSE).
