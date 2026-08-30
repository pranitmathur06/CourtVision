#!/usr/bin/env bash
# Provision a Runpod GPU pod, ship CourtVision to it, and run the GPU suite.
#
#   bash scripts/launch_gpu_run.sh            # plan only, creates nothing
#   bash scripts/launch_gpu_run.sh --confirm  # actually provision (BILLABLE)
#
# Follows Runpod golden path 04 (batch training run on a pod): no --ports, a
# network volume that outlives the pod, the job launched DETACHED so it survives
# an SSH drop, and progress read by tailing a log on the volume.
set -uo pipefail
cd "$(dirname "$0")/.."

GPU="${GPU:-NVIDIA GeForce RTX 4090}"
DC="${DC:-EU-RO-1}"
TEMPLATE="${TEMPLATE:-runpod-torch-v280}"
VOLUME_GB="${VOLUME_GB:-60}"
DISK_GB="${DISK_GB:-60}"     # container disk; the default 20 will not hold the frame cache
GPU_COUNT="${GPU_COUNT:-1}"  # 2 for v2 §7.2 disaggregation across cuda:0/cuda:1
HOURS="${HOURS:-3}"          # hard deadline enforced by a local watchdog, see below
NAME="${NAME:-courtvision}"

say() { printf "\n\033[1m>>> %s\033[0m\n" "$*"; }
die() { printf "\n\033[31mFAILED: %s\033[0m\n" "$*"; exit 1; }

DEADLINE=$(python3 -c "
import datetime as d
print((d.datetime.now(d.timezone.utc)+d.timedelta(hours=$HOURS)).strftime('%Y-%m-%d %H:%M UTC'))")

cat <<PLAN
Plan
  GPU              $GPU   (x1 — V7 needs one; §7.2 disaggregation needs two)
  data centre      $DC
  template         $TEMPLATE  (ships a CUDA torch; do NOT create a venv over it)
  network volume   ${VOLUME_GB} GB, survives the pod
  container disk   ${DISK_GB} GB  (the default 20 will NOT hold the ~8 GB frame cache)
  gpus             ${GPU_COUNT}
  hard deadline    $DEADLINE  (~${HOURS}h) — enforced locally, see below
  ships            ~$(du -sh data/labeled/actions 2>/dev/null | cut -f1) clips,
                   $(du -sh checkpoints 2>/dev/null | cut -f1) checkpoints, plus the source tree
  runs             scripts/run_gpu_suite.sh (audit, V7, confusion, V9, v2 CUDA)

  COST GUARD. Runpod golden path 04 uses --terminate-after, but runpodctl
  2.12.0 has no such flag and no TTL of any kind: pod-create offers only
  --wait-timeout, which is how long to wait for SSH, not a lifetime. So
  nothing on Runpod's side will stop the meter. This script starts a local
  watchdog that deletes the pod at the deadline, and prints the delete command.
  Neither is bulletproof if this machine sleeps — check the console yourself.
  A stopped pod still bills for its volume; a deleted one does not.
PLAN

[ "${1:-}" = "--confirm" ] || { echo; echo "Dry run. Nothing created. Re-run with --confirm."; exit 0; }

command -v runpodctl >/dev/null || die "runpodctl not installed"
runpodctl user >/dev/null 2>&1 || die "no RUNPOD_API_KEY — get one at https://console.runpod.io/user/settings"

say "Registering an SSH key (must exist BEFORE the pod is created)"
runpodctl ssh list-keys 2>/dev/null | head -5

say "Creating the network volume"
VOL=$(runpodctl network-volume create --name "${NAME}-vol" --size "$VOLUME_GB" \
        --data-center-id "$DC" 2>&1 | tee /dev/stderr | grep -oE '[0-9a-z]{20,}' | head -1)
[ -n "$VOL" ] || die "no volume id returned"

say "Creating the pod (no --ports: this is a batch job, it serves nothing)"
# --wait blocks until ssh answers, which is what the hand-rolled polling loop
# here used to do worse. --ports is deliberately absent: a batch job serves
# nothing.
POD=$(runpodctl pod create --name "$NAME" --template-id "$TEMPLATE" \
        --gpu-id "$GPU" --gpu-count "$GPU_COUNT" --data-center-ids "$DC" \
        --container-disk-in-gb "$DISK_GB" \
        --network-volume-id "$VOL" --volume-mount-path /workspace \
        --ssh --wait --wait-timeout 10m 2>&1 | tee /dev/stderr \
      | grep -oE '[0-9a-z]{20,}' | head -1)
[ -n "$POD" ] || die \
"pod create failed or timed out. A brand-new pod can draw a machine whose
runtime never becomes ready (golden path 07 hit this). If an id was printed
above, delete it and try again rather than waiting:
  runpodctl pod delete <id>"

say "Starting the local cost watchdog (${HOURS}h)"
setsid bash -c "sleep $((HOURS*3600)); runpodctl pod delete '$POD'" \
  >/tmp/courtvision-watchdog.log 2>&1 </dev/null &
echo "watchdog pid $! — deletes pod $POD at $DEADLINE"

eval "$(runpodctl ssh info "$POD" | python3 -c \
  'import sys,json; d=json.load(sys.stdin); print(f"IP={d[\"ip\"]} PORT={d[\"port\"]} KEY={d[\"ssh_key\"][\"path\"]}")')"
SSHQ=(ssh -i "$KEY" -o StrictHostKeyChecking=no -p "$PORT" "root@$IP")
echo "pod $POD at $IP:$PORT"

say "Shipping the tree (source + clips + checkpoints, skipping the 9.3 GB detector set)"
# INCLUDE what the suite reads, rather than excluding what it does not. An
# exclude list silently grows stale: data/labeled holds 15 GB across
# bard_meta/clips, detector, handler_harvest, spacejam and roboflow, none of
# which any stage on the pod opens, and each new one would have to be
# remembered. This list is short because the suite's real appetite is small.
#
#   data/labeled/actions            3,455 clips — audit, V7, confusion report
#   data/raw_clips                  sample + holdout — V9, disaggregation
#   checkpoints                     detector + current classifier
#   data/labeled/bard_meta/*.csv    the official play list V9 names players from
#   data/ground_truth               hand-made answer keys
NEEDED=(src scripts tests pyproject.toml
        data/labeled/actions data/raw_clips data/ground_truth
        checkpoints
        data/labeled/bard_meta/dataset.csv)
for path in "${NEEDED[@]}"; do
  [ -e "$path" ] || die "missing $path — the suite needs it"
done
tar czf /tmp/courtvision.tgz --exclude='__pycache__' "${NEEDED[@]}" \
  || die "tar failed"
ls -lh /tmp/courtvision.tgz | awk '{print "  bundle:", $5}'
"${SSHQ[@]}" 'mkdir -p /workspace/CourtVision' || die "ssh failed"
scp -i "$KEY" -o StrictHostKeyChecking=no -P "$PORT" /tmp/courtvision.tgz \
  "root@$IP:/workspace/" || die "scp failed"
"${SSHQ[@]}" 'cd /workspace/CourtVision && tar xzf ../courtvision.tgz && ls data/labeled/actions | wc -l'

say "Launching the suite DETACHED (setsid + </dev/null survive an SSH drop)"
"${SSHQ[@]}" 'cd /workspace/CourtVision && export HF_HOME=/workspace/hf-cache && \
  setsid bash -c "bash scripts/bootstrap_pod.sh && bash scripts/run_gpu_suite.sh" \
  > /workspace/suite.log 2>&1 </dev/null & echo LAUNCHED'

cat <<NEXT

Running. Monitor from here:
  ssh -i "$KEY" -p $PORT root@$IP 'tail -n 40 /workspace/suite.log'
  ssh -i "$KEY" -p $PORT root@$IP 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv'

Retrieve results:
  scp -i "$KEY" -P $PORT -r root@$IP:/workspace/CourtVision/outputs ./outputs-gpu

Runpod has no auto-terminate in runpodctl 2.12.0, so the meter runs until the
pod is deleted. A local watchdog will delete it at $DEADLINE, but that dies if
this machine sleeps. Delete it yourself as soon as the results are copied:
  runpodctl pod delete $POD
  runpodctl network-volume delete $VOL     # the volume bills separately
NEXT
