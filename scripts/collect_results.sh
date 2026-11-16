#!/bin/bash
# Summarise every finished run: mean test win rate over the last 10 evaluation
# points, which is the statistic used throughout this project.
# Usage (on krull):  bash /projects/users/$USER/marl-cluster/scripts/collect_results.sh
set -euo pipefail
logs="/projects/users/${USER}/marl-cluster/logs"

printf "%-28s %-10s %-10s %-10s %s\n" "RUN" "LAST-10" "FINAL" "BEST" "STATE"
for f in "${logs}"/mappo-marl-*.out; do
    [ -e "$f" ] || continue
    arm=$(grep -oE 'arm=[a-z]+ map=[a-z0-9_]+ slice=[^ ]+ steps=[0-9]+ seed=[0-9]+' "$f" | head -1 \
          | sed -E 's/arm=([a-z]+) map=([a-z0-9_]+).*seed=([0-9]+)/\2_\1_s\3/')
    [ -n "$arm" ] || continue
    vals=$(grep -oE 'test_battle_won_mean: *[0-9.]+' "$f" | awk '{print $2}')
    [ -n "$vals" ] || { printf "%-28s %-10s %-10s %-10s %s\n" "$arm" "-" "-" "-" "no evals yet"; continue; }
    l10=$(echo "$vals" | tail -10 | awk '{s+=$1} END {printf "%.4f", s/NR}')
    fin=$(echo "$vals" | tail -1)
    bst=$(echo "$vals" | sort -rn | head -1)
    state=$(grep -q "finished arm=" "$f" && echo done || echo running)
    printf "%-28s %-10s %-10s %-10s %s\n" "$arm" "$l10" "$fin" "$bst" "$state"
done
