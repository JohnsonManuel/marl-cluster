#!/bin/bash
# Summarise the QMIX seed runs: per-arm mean and sample standard deviation of the
# last-10-evaluation-point mean, across seeds.
#
# Usage: bash scripts/collect_qmix_seeds.sh [log-dir]
# Default log dir is the newest logs/qmix_seeds_*/
set -uo pipefail
project="/projects/users/${USER}/marl-cluster"
dir="${1:-$(ls -dt ${project}/logs/qmix_seeds_* 2>/dev/null | head -1)}"
[ -d "${dir:-}" ] || { echo "no qmix_seeds log dir found"; exit 1; }
echo "log dir: ${dir}"
echo

printf "%-11s %-11s %-8s %-9s %-9s %s\n" MAP ARM SEEDS MEAN STD "per-seed"
for map in 5m_vs_6m 3s_vs_5z; do
  for arm in baseline allfeat chosenfeat; do
    vals=""
    for s in 1 2 3 4 5; do
      f="${dir}/${map}_${arm}_s${s}.log"
      [ -f "$f" ] || continue
      v=$(grep -oE "test_battle_won_mean: *[0-9.]+" "$f" | grep -oE "[0-9.]+$" \
          | tail -10 | awk '{t+=$1;c++} END{if(c)printf "%.4f",t/c}')
      [ -n "$v" ] && vals="$vals $v"
    done
    [ -z "$vals" ] && { printf "%-11s %-11s %-8s %s\n" "$map" "$arm" 0 "(no data)"; continue; }
    echo "$vals" | awk -v m="$map" -v a="$arm" '{
      n=NF; s=0; for(i=1;i<=n;i++) s+=$i; mu=s/n;
      q=0; for(i=1;i<=n;i++) q+=($i-mu)^2;
      sd=(n>1)? sqrt(q/(n-1)) : 0;
      printf "%-11s %-11s %-8d %-9.4f %-9.4f %s\n", m, a, n, mu, sd, $0
    }'
  done
done
echo
echo "completed: $(grep -l 'Finished after' ${dir}/*.log 2>/dev/null | wc -l)/30"
echo "tracebacks: $(grep -l Traceback ${dir}/*.log 2>/dev/null | wc -l)"
