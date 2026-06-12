# Apollo — AI song forge

Idea in → full song out (music + real vocals) → one-click save. Runs as a desktop **app**: own window, app-menu icon, launch sting. **Free, no API key required.**

## Quick start

```
curl -fsSL https://raw.githubusercontent.com/the-priest/apollo/main/install.sh | bash
```

Launch **Apollo** from your app menu → pick the **HOSTED** engine → type an idea → GENERATE. Real AI vocals, no key, no GPU needed.

## Four engines

**hosted — FREE real AI vocals, no GPU, no key. ← use this on a laptop.** Runs ACE-Step (open-source, Suno-class) on a shared hosted GPU via Hugging Face. You need nothing but the tiny `gradio-client` package (the installer adds it). First song can take a few minutes because it's a shared free queue; after that it's quick. Optional: drop a free Hugging Face token in the gear drawer to get your own queue/quota.

```
apollo --setup-hosted
```

**cloud — real vocals, fast, most reliable.** MiniMax `music-2.6-free`. Needs a free key (https://platform.minimax.io → user center → API keys). Best if you want speed and consistency and don't mind one signup.

**neural — local AI vocals, private + offline.** Same ACE-Step model but on *your* machine. Great on a GPU box (seconds/song); **slow on a no-GPU laptop** (minutes/song). First run downloads ~3.5GB.

```
apollo --setup-neural
```

**synth — offline, instant, always works.** numpy synth + espeak-ng. No key, no network, renders in seconds — but the vocals are robotic vocaloid, not real singing. Good for quick sketches.

LLM for lyrics/structure: SiliconFlow / DeepSeek-V4-Flash primary, Groq fallback only. (Note: SiliconFlow & Groq only do text/speech — they can't generate full songs, which is why song vocals come from ACE-Step or MiniMax.)

## Auto vs Manual

**AUTO** — type an idea, GENERATE, it writes everything.

**MANUAL · EDIT** — control it yourself:
- write/paste your own **title, style prompt, lyrics** (hosted/cloud/neural) or **title, lyrics, BPM, key** (synth)
- tag sections: `[Intro] [Verse] [Pre Chorus] [Chorus] [Bridge] [Outro]` (auto-mapped per engine)
- **✦ DRAFT WITH AI** fills the fields from your idea so you can edit lines before rendering
- any finished song has **Edit & re-render**

## Use

Pick ENGINE / MODE / genre / mood / tempo / voice / length, type the idea, GENERATE. Watch the log, play it. SAVE drops mp3 + lyrics .txt + spec .json into `~/Music/Apollo`. Library list replays saved songs. Defaults to whichever engine is ready (hosted → cloud → neural → synth).

Keys/tokens go in the gear drawer (chmod-600 `~/.config/apollo/config.json`) or env: `SILICONFLOW_API_KEY`, `MINIMAX_API_KEY`, `GROQ_API_KEY`.

## Which engine?

- **No-GPU laptop, want it free + real vocals:** hosted.
- **Want fast + reliable, OK with one free signup:** cloud.
- **Have a GPU, want private/offline:** neural.
- **Just need something instant and offline:** synth.

## Flags

```
apollo
```
```
apollo --setup-hosted
```
```
apollo --setup-neural
```
```
apollo --install-desktop
```
```
apollo --no-sound
```
```
apollo --demo
```

`--demo` renders the offline synth template to ./apollo_demo.wav. `--host`/`--port` to move it.

## Genres

synthwave · pop · rock · hiphop · edm · lofi · metal · folk · auto
