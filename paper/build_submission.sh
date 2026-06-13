#!/usr/bin/env bash
set -euo pipefail

PAPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_NAME="when_plausible_videos_lie_iclr_submission.pdf"
cd "$PAPER_DIR"

if command -v latexmk >/dev/null 2>&1 && command -v perl >/dev/null 2>&1 && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex; then
  :
else
  echo "latexmk unavailable, missing Perl, or failed; falling back to pdflatex/bibtex." >&2
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
  bibtex main
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
fi

cp main.pdf "$OUT_NAME"

if [[ -n "${USERPROFILE:-}" ]]; then
  cp main.pdf "$USERPROFILE/Downloads/$OUT_NAME"
  mkdir -p "$USERPROFILE/OneDrive/Desktop"
  cp main.pdf "$USERPROFILE/OneDrive/Desktop/best of n video transformer world model-v2.pdf"
elif [[ -n "${HOME:-}" ]]; then
  cp main.pdf "$HOME/Downloads/$OUT_NAME"
fi

echo "Built $PAPER_DIR/$OUT_NAME"
