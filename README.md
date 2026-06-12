# Apollo — AI song forge

Idea in → full song out (music + real vocals) → one-click save. Runs as a desktop **app**: own window, app-menu icon, launch sting. Three engines, two of them make genuine sung vocals, **one is 100% free and local with no API key**.

## Install

```
curl -fsSL https://raw.githubusercontent.com/the-priest/apollo/main/install.sh | bash
```

Launch **Apollo** from your app menu. To also install the free local AI-vocal engine in one go (heavy — pulls PyTorch + a ~3.5GB model):

```
curl -fsSL https://raw.githubusercontent.com/the-priest/apollo/main/install.sh | bash -s -- --neural
```

## Three engines

**neural — ACE-Step 1.5, fully local, FREE, no key.** Open-source diffusion music model that generates real songs with sung vocals and lyrics. Benchmarks sit between Suno v4.5 and v5. This is the "free for anyone" path: no account, no API key, runs offline. One-time setup:

```
apollo --setup-neural
```

Speed depends on your hardware. With an NVIDIA/AMD/Apple-silicon GPU it's fast (seconds to a couple minutes a song). On a **CPU-only laptop (like the X395) it works but is slow** — expect several minutes per song, and the first song downloads the model (~3.5GB to `~/.cache/ace-step`). If you have any CUDA box on hand, run Apollo there for the neural engine and it'll fly. (Hosted free fallback if you don't: acemusic.ai.)

**cloud — MiniMax `music-2.6-free`.** Real sung vocals, fast, needs a free key (https://platform.minimax.io → user center → API keys). DeepSeek (SiliconFlow) writes the lyrics/style; MiniMax renders the song. Works with only a MiniMax key too.

**synth — offline numpy + espeak-ng.** Instant, zero deps, no key, but the vocals are robotic vocaloid, not Suno. Good for quick sketches and fully offline use.

LLM for lyrics/structure: SiliconFlow / DeepSeek-V4-Flash primary, Groq fallback only.

## Auto vs Manual

**AUTO** — type an idea, hit GENERATE, it writes everything.

**MANUAL · EDIT** — you control it:
- write/paste your own **title, style prompt, lyrics** (neural & cloud) or **title, lyrics, BPM, key** (synth)
- tag sections: `[Intro] [Verse] [Pre Chorus] [Chorus] [Bridge] [Outro]` (auto-mapped to each engine's format)
- **✦ DRAFT WITH AI** fills the fields from your idea so you can edit lines before rendering
- any finished song has **Edit & re-render** to load it back in and tweak

## Use

Pick ENGINE / MODE / genre / mood / tempo / voice / length, type the idea (or fill the editor), GENERATE. Watch the log, play it. SAVE drops mp3 + lyrics .txt + spec .json into `~/Music/Apollo`. Library list replays saved songs. Defaults to whichever engine is ready (neural if installed, else cloud if you have a key, else synth).

Keys go in the gear drawer (chmod-600 `~/.config/apollo/config.json`) or env: `SILICONFLOW_API_KEY`, `MINIMAX_API_KEY`, `GROQ_API_KEY`.

## Flags

```
apollo
```

```
apollo --setup-neural
```

```
apollo --install-desktop
```

```
apollo --no-window
```

```
apollo --no-sound
```

```
apollo --demo
```

`--demo` renders the offline synth template to ./apollo_demo.wav. `--host`/`--port` to move it.

## Which engine should I use?

- **Want real, professional-sounding songs for free with no signup, and you have a GPU (or patience on CPU):** neural.
- **Want real songs fast and don't mind a free API key:** cloud.
- **Want something instant and offline for a quick idea:** synth.

## Genres

synthwave · pop · rock · hiphop · edm · lofi · metal · folk · auto
