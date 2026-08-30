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
HOURS="${HOURS:-3}"
NAME="${NAME:-courtvision}"

say() { printf "\n\033[1m>>> %s\033[0m\n" "$*"; }
die() { printf "\n\033[31mFAILED: %s\033[0m\n" "$*"; exit 1; }

TERMINATE_AT=$(python3 -c "
import datetime as d
print((d.datetime.now(d.timezone.utc)+d.timedelta(hours=$HOURS)).strftime('%Y-%m-%dT%H:%M:%SZ'))")

cat <<PLAN
Plan
  GPU              $GPU   (x1 — V7 needs one; §7.2 disaggregation needs two)
  data centre      $DC
  template         $TEMPLATE  (ships a CUDA torch; do NOT create a venv over it)
  network volume   ${VOLUME_GB} GB, survives the pod
  auto-terminate   $TERMINATE_AT  (~${HOURS}h from now)
  ships            ~$(du -sh data/labeled/actions 2>/dev/null | cut -f1) clips,
                   $(du -sh checkpoints 2>/dev/null | cut -f1) checkpoints, plus the source tree
  runs             scripts/run_gpu_suite.sh (audit, V7, confusion, V9, v2 CUDA)

  --terminate-after is the real cost guard: the pod DELETES itself at that time.
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
POD=$(runpodctl pod create --name "$NAME" --template-id "$TEMPLATE" \
        --gpu-id "$GPU" --data-center-ids "$DC" \
        --network-volume-id "$VOL" --volume-mount-path /workspace \
        --ssh --terminate-after "$TERMINATE_AT" 2>&1 | tee /dev/stderr \
      | grep -oE '[0-9a-z]{20,}' | head -1)
[ -n "$POD" ] || die "no pod id returned"

say "Waiting for the runtime"
for i in $(seq 1 40); do
  if runpodctl ssh info "$POD" >/dev/null 2>&1; then echo "ready"; break; fi
  printf "."; sleep 15
done
runpodctl ssh info "$POD" >/dev/null 2>&1 || die \
"runtime never came up. Golden path 07 hit this: a bad machine never becomes ready.
Delete it and create a fresh one rather than waiting:
  runpodctl pod delete $POD"

eval "$(runpodctl ssh info "$POD" | python3 -c \
  'import sys,json; d=json.load(sys.stdin); print(f"IP={d[\"ip\"]} PORT={d[\"port\"]} KEY={d[\"ssh_key\"][\"path\"]}")')"
SSHQ=(ssh -i "$KEY" -o StrictHostKeyChecking=no -p "$PORT" "root@$IP")
echo "pod $POD at $IP:$PORT"

say "Shipping the tree (source + clips + checkpoints, skipping the 9.3 GB detector set)"
tar czf /tmp/courtvision.tgz \
  --exclude='.venv' --exclude='.git' --exclude='data/labeled/detector' \
  --exclude='outputs' --exclude='__pycache__' . || die "tar failed"
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

Pod $POD self-deletes at $TERMINATE_AT. To stop paying sooner:
  runpodctl pod delete $POD
NEXT
