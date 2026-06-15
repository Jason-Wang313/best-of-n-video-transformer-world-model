#!/usr/bin/env bash
set -euo pipefail

PAPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FINAL_DIR="$PAPER_DIR/final"
OUT_NAME="best of n video transformer world model-v4.pdf"
cd "$PAPER_DIR"
mkdir -p "$FINAL_DIR"

if command -v latexmk >/dev/null 2>&1 && command -v perl >/dev/null 2>&1 && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex; then
  :
else
  echo "latexmk unavailable, missing Perl, or failed; falling back to pdflatex/bibtex." >&2
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
  bibtex main
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
fi

cp main.pdf "$FINAL_DIR/$OUT_NAME"

echo "Built $FINAL_DIR/$OUT_NAME"
