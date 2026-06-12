# Apollo — AI song forge

Idea in → full song out (music + vocals) → one-click save. Runs as a desktop app: own window, app-menu icon, launch sting. Two engines, both free, no API key required.

## Install

```
curl -fsSL https://raw.githubusercontent.com/the-priest/apollo/main/install.sh | bash
```

Launch **Apollo** from your app menu. To also install the free AI singer in one go:

```
curl -fsSL https://raw.githubusercontent.com/the-priest/apollo/main/install.sh | bash -s -- --voice
```

## Two engines

**VOICE — free real AI singer (Kokoro).** A small open-source neural voice model (Apache-2.0) that runs on your CPU — no API key, no GPU, no account. It sings your lyrics over the generated music. One-time setup downloads the model (~310MB):

```
apollo --setup-voice
```

After that it's fully offline. First song in a session takes ~10–20s extra to load the model; then it's quick. Voices: male/female, US/UK.

**SYNTH — instant offline.** Pure numpy synthesis + a robotic espeak voice. No setup, no model, renders in seconds. The voice is robotic (vocaloid-ish), but the music is warm and the whole thing always works with zero dependencies. Good for quick sketches or if the voice model isn't installed.

Lyrics/structure are written by SiliconFlow/DeepSeek if you add a key (gear menu); otherwise a built-in template is used. Groq is fallback only.

## Auto vs Manual

**AUTO** — type an idea, GENERATE, it writes everything.

**MANUAL · EDIT** — control it:
- write/paste your own **title, lyrics, BPM, key**
- tag sections: `[Intro] [Verse] [Pre Chorus] [Chorus] [Bridge] [Outro]`
- **✦ DRAFT WITH AI** fills the fields from your idea so you can edit before rendering
- any finished song has **Edit & re-render**

## Use

Pick ENGINE / MODE / genre / mood / tempo / voice / length, type the idea, GENERATE. Watch the log, play it, hit **QUIT** (top-right) to stop the server cleanly. SAVE drops mp3 + lyrics .txt + spec .json into `~/Music/Apollo`. Library list replays saved songs.

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

`--demo` renders an offline song to ./apollo_demo.wav (optional genre, e.g. `--demo lofi`). `--host`/`--port` to move it (it auto-finds a free port and won't start a second copy if one's already running).

## Genres

synthwave · pop · rock · hiphop · edm · lofi · metal · folk · auto
