#!/usr/bin/env bash
# Apollo — AI song forge · installer
#   curl -fsSL https://raw.githubusercontent.com/the-priest/apollo/main/install.sh | bash
# the installer sets up the FREE hosted AI-vocal engine by default (no GPU, no key).
# add --neural to also install the heavy LOCAL model (only worth it on a GPU machine):
#   ... | bash -s -- --neural
set -euo pipefail

RAW_URL="https://raw.githubusercontent.com/the-priest/apollo/main/apollo.py"
APP_DIR="$HOME/.local/share/apollo"
BIN_DIR="$HOME/.local/bin"
WANT_NEURAL=0
[ "${1:-}" = "--neural" ] && WANT_NEURAL=1

say(){ printf '\033[1;33m[apollo]\033[0m %s\n' "$*"; }

say "installing deps (espeak-ng, python3-numpy, ffmpeg, python3-venv)…"
sudo apt-get install -y --no-install-recommends espeak-ng python3-numpy ffmpeg python3-venv python3-pip

if ! command -v chromium >/dev/null && ! command -v chromium-browser >/dev/null \
   && ! command -v brave-browser >/dev/null && ! command -v google-chrome >/dev/null; then
  say "NOTE: no chromium/brave/chrome found — Apollo opens in your default browser instead of an app window."
fi

mkdir -p "$APP_DIR" "$BIN_DIR"

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || true)"
if [ -n "$SRC_DIR" ] && [ -f "$SRC_DIR/apollo.py" ]; then
  say "using local apollo.py"; cp "$SRC_DIR/apollo.py" "$APP_DIR/apollo.py"
else
  say "fetching apollo.py…"; curl -fsSL "$RAW_URL" -o "$APP_DIR/apollo.py"
fi
chmod +x "$APP_DIR/apollo.py"

cat > "$BIN_DIR/apollo" <<EOF
#!/bin/sh
exec python3 "$APP_DIR/apollo.py" "\$@"
EOF
chmod +x "$BIN_DIR/apollo"

say "installing app icon + launcher…"
python3 "$APP_DIR/apollo.py" --install-desktop

say "setting up the FREE hosted AI-vocal engine (no GPU, no key)…"
python3 "$APP_DIR/apollo.py" --setup-hosted || say "hosted setup hiccup — you can run it later: apollo --setup-hosted"

if [ "$WANT_NEURAL" = "1" ]; then
  say "also setting up the LOCAL neural model (heavy — PyTorch + weights)…"
  python3 "$APP_DIR/apollo.py" --setup-neural
fi

case ":$PATH:" in *":$BIN_DIR:"*) ;; *) say "NOTE: $BIN_DIR not in PATH — launch from your app menu, or run: $BIN_DIR/apollo";; esac

say "done. launch 'Apollo' from your app menu."
say "pick the HOSTED engine for free real AI vocals (no key). songs save to ~/Music/Apollo"
