#!/usr/bin/env bash
# Everything that needs a GPU, in one command, in dependency order.
#
# Written because the rental is metered and the failure mode is discovering at
# step 5 that step 2 needed rerunning. Each stage logs to outputs/ and the
# summary at the end is the thing to read.
#
# Usage:  bash scripts/run_gpu_suite.sh
set -uo pipefail
cd "$(dirname "$0")/.."
PY=./.venv/bin/python
mkdir -p outputs
declare -a NAMES=() RESULTS=()

run() {                       # run <name> <logfile> <command...>
  local name="$1" log="$2"; shift 2
  echo ""
  echo "=============================================================="
  echo ">>> $name"
  echo "=============================================================="
  if "$@" 2>&1 | tee "outputs/$log"; then
    local rc=${PIPESTATUS[0]}
  else
    local rc=${PIPESTATUS[0]}
  fi
  NAMES+=("$name")
  if [ "$rc" -eq 0 ]; then RESULTS+=("PASS"); else RESULTS+=("FAIL (exit $rc)"); fi
  return 0                    # never abort the suite; a later stage may still inform
}

echo "CourtVision GPU suite — started $(date '+%Y-%m-%d %H:%M:%S')"
$PY -c "import torch; print(f'torch {torch.__version__}  cuda={torch.cuda.is_available()}  devices={torch.cuda.device_count()}')"
$PY -c "from courtvision.device import resolve_device; print('resolve_device() ->', resolve_device())"

# 1. Confirm the corpus rebalance travelled. Cheap, and if the clips did not
#    all rsync across, every number after this is meaningless.
run "audit: source/label confound" audit.log \
    $PY -m scripts.audit_source_cue

# 2. The actual training run. This is the long one.
run "V7: action classifier" v7_run.log \
    $PY -u scripts/validate_v7.py

# 3. The pass criterion. Read the cross-source generalisation block, not the
#    headline accuracy.
run "V7: per-class + cross-source report" confusion.log \
    $PY -m scripts.report_action_confusion

# 4. End-to-end pipeline on the retrained 7-class model.
run "V9: end-to-end" v9.log \
    $PY -m scripts.validate_v9

# 5. Host-side kernel check. Runs without a GPU and is seconds long; if the
#    arithmetic is wrong here, nvcc will not save it.
run "kernel: host numerics + reduction" kernel_host.log \
    $PY -m scripts.verify_kernel_numerics

# 6. v2 §7.1/§7.2 — kernel compiles, matches the oracle, is faster, and the
#    split pipeline runs. Needs 2 devices for the disaggregation step.
run "v2: CUDA kernel + disaggregated serving" verify_v2.log \
    $PY -m scripts.verify_v2_gpu

echo ""
echo "=============================================================="
echo "SUMMARY — $(date '+%H:%M:%S')"
echo "=============================================================="
for i in "${!NAMES[@]}"; do printf "  %-42s %s\n" "${NAMES[$i]}" "${RESULTS[$i]}"; done
echo ""
echo "  Full logs in outputs/. The number that decides V7 is the"
echo "  cross-source generalisation block in outputs/confusion.log,"
echo "  not the headline accuracy."
echo ""
echo "  Copy results back, then DESTROY the pod — stopping still bills."
