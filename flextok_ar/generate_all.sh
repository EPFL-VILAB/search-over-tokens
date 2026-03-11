#!/bin/bash
# Run from repo root: bash flextok_ar/generate_all.sh
# Requires: pip install -e . (from release/)

set -e

# ── Text-to-Image (T2I) ─────────────────────────────────────────────────────

# d12 (smallest)
python generate.py --mode t2i \
    --model-id EPFL-VILAB/FlexAR-113M-T2I \
    --prompt "A serene lake at sunset" \
    --seed 42 --debug-tokens \
    --output test_d12.png

# d18
python generate.py --mode t2i \
    --model-id EPFL-VILAB/FlexAR-382M-T2I \
    --prompt "A serene lake at sunset" \
    --seed 42 --debug-tokens \
    --output test_d18.png

# d26
python generate.py --mode t2i \
    --model-id EPFL-VILAB/FlexAR-1B-T2I \
    --prompt "A serene lake at sunset" \
    --seed 42 --debug-tokens \
    --output test_d26.png

# d36
python generate.py --mode t2i \
    --model-id EPFL-VILAB/FlexAR-3B-T2I \
    --prompt "A serene lake at sunset" \
    --seed 42 --debug-tokens \
    --output test_d36.png

# 2D GridTok d36
python generate.py --mode t2i \
    --model-id EPFL-VILAB/GridAR-3B-T2I \
    --prompt "A serene lake at sunset" \
    --seed 42 --debug-tokens \
    --output test_gridtok_d36.png

# ── Class-to-Image (C2I) ────────────────────────────────────────────────────

# FlexAR-1B-C2I (ImageNet class 285 = cat)
python generate.py --mode c2i \
    --model-id EPFL-VILAB/FlexAR-1B-C2I \
    --class-label 285 \
    --seed 42 --debug-tokens \
    --output test_c2i_cat.png

# Multiple C2I samples (class 281 = tabby cat)
python generate.py --mode c2i \
    --model-id EPFL-VILAB/FlexAR-1B-C2I \
    --class-label 281 \
    --num-samples 4 \
    --output test_c2i_tabby_{i}.png
