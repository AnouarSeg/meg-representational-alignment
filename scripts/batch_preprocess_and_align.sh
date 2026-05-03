#!/usr/bin/env bash
# Preprocess all downloaded EEG1 subjects then run alignment complexity.
# Run after download_things_eeg1.py finishes.
# Usage: bash scripts/batch_preprocess_and_align.sh

set -e
PYTHON=/opt/anaconda3/envs/things-meg/bin/python
EEG1=/Volumes/MEG/things-eeg1
RESULTS=results/eeg1

mkdir -p "$RESULTS"

echo "=== Batch preprocessing THINGS-EEG1 ==="
for sub_dir in "$EEG1"/sub-*/; do
    sub=$(basename "$sub_dir")
    means="$EEG1/derivatives/preprocessed/$sub/${sub}_condition_means.npy"
    if [ -f "$means" ]; then
        echo "$sub already preprocessed — skipping"
    else
        echo "Preprocessing $sub ..."
        $PYTHON scripts/preprocess_eeg1.py --subject "$sub"
    fi
done

echo ""
echo "=== Running alignment complexity ==="
$PYTHON scripts/run_alignment_eeg1.py

echo ""
echo "=== Done ==="
