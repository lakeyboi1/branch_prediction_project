#!/usr/bin/env bash

set -euo pipefail

PROJ="$HOME/Desktop/project markov"
GEM5_SRC="$PROJ/gem5-fresh"
GEM5_BIN="${GEM5_BIN:-$GEM5_SRC/build/X86/gem5.opt}"
WORKERS="${WORKERS:-8}"
DEPTH_RESULTS="results/X86/lvp_depth_sweep"
CONF_RESULTS="results/X86/lvp_conf_sweep"
PLOT_DIR="plots/lvp"

echo " gem5 source : $GEM5_SRC"
echo " gem5 binary : $GEM5_BIN"
echo " Workers     : $WORKERS"
echo ""

echo "--- Step 1/3: Building gem5 ---"
cd "$GEM5_SRC"
scons build/X86/gem5.opt -j"$WORKERS"
echo "Build complete."
echo ""

echo "--- Step 2/3: Running LVP parameter sweep ---"
cd "$PROJ"
export GEM5_BIN
python3 sweep_lvp_params.py "$WORKERS"
echo "Sweep complete."
echo ""

echo "--- Step 3/3: Generating plots ---"
python3 plot_lvp_accuracy.py \
    --depth-results "$DEPTH_RESULTS" \
    --conf-results  "$CONF_RESULTS" \
    --out-dir       "$PLOT_DIR"

echo ""
echo "=============================================="
echo " Done! Plots saved to $PLOT_DIR/"
echo "=============================================="