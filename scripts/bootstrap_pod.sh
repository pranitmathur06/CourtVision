#!/usr/bin/env bash
# One command to paste into a fresh RunPod web terminal.
#
#   bash <(curl -sSL https://raw.githubusercontent.com/pronton1234/CourtVision/feat/v1-pipeline/scripts/bootstrap_pod.sh)
#
# ...or, since the repo may be private, clone first and run it locally:
#
#   git clone -b feat/v1-pipeline https://github.com/pronton1234/CourtVision.git
#   cd CourtVision && bash scripts/bootstrap_pod.sh
#
# Installs into the image's EXISTING Python rather than a fresh venv. RunPod's
# PyTorch images ship a CUDA-enabled torch built against the driver on the box;
# a venv would hide it and pip would helpfully install a CPU-only wheel over the
# top, at which point everything runs and nothing uses the GPU you are paying
# for. So this checks CUDA before and after, and stops if it disappears.
set -uo pipefail

REPO_BRANCH="feat/v1-pipeline"
say() { printf "\n\033[1m>>> %s\033[0m\n" "$*"; }
die() { printf "\n\033[31mFAILED: %s\033[0m\n" "$*"; exit 1; }

say "Environment"
python3 -c "import sys; print('python', sys.version.split()[0])" || die "no python3"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader \
  || die "nvidia-smi not found — this is not a GPU pod"

CUDA_BEFORE=$(python3 -c "
import torch
print(f'{torch.__version__}|{torch.cuda.is_available()}|{torch.cuda.device_count()}')
" 2>/dev/null) || die "torch not importable in the image's python"
echo "torch before install: $CUDA_BEFORE"
case "$CUDA_BEFORE" in
  *"|True|"*) ;;
  *) die "the image's torch cannot see CUDA; pick a PyTorch/CUDA template" ;;
esac

if [ ! -f pyproject.toml ]; then
  say "Cloning"
  git clone -b "$REPO_BRANCH" https://github.com/pronton1234/CourtVision.git \
    || die "clone failed (private repo? use a token or rsync the tree across)"
  cd CourtVision || die "cannot enter CourtVision"
fi

say "Installing (torch is already satisfied, so pip must not replace it)"
python3 -m pip install -q --upgrade pip
python3 -m pip install -q -e ".[dev]" || die "pip install failed"

CUDA_AFTER=$(python3 -c "
import torch
print(f'{torch.__version__}|{torch.cuda.is_available()}|{torch.cuda.device_count()}')
")
echo "torch after install:  $CUDA_AFTER"
case "$CUDA_AFTER" in
  *"|True|"*) ;;
  *) die "installing dependencies replaced torch with a build that cannot see CUDA.
Reinstall the CUDA wheel for this image before going further:
  pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu121" ;;
esac

say "nvcc (needed to compile the fused kernel for v2 §7.1)"
if command -v nvcc >/dev/null; then nvcc --version | tail -2
else echo "WARNING: nvcc missing. The v2 kernel step will fail; everything else runs.
  Fix with: apt-get update && apt-get install -y cuda-toolkit-12-1"
fi

say "Tests"
python3 -m pytest -q || die "test suite failed before any GPU work"

say "Device resolution"
python3 -c "from courtvision.device import resolve_device; print('resolve_device() ->', resolve_device())"

say "Data check"
CLIPS=$(find data/labeled/actions -name '*.mp4' 2>/dev/null | wc -l | tr -d ' ')
echo "action clips present: ${CLIPS:-0} (expect 3455)"
if [ "${CLIPS:-0}" -lt 3000 ]; then
cat <<'MSG'

The clips are gitignored, so they have to be copied across. From the Mac,
using the host and port under RunPod's Connect -> SSH:

  rsync -avz -e "ssh -p <PORT>" data/labeled/actions \
    root@<HOST>:/workspace/CourtVision/data/labeled/
  rsync -avz -e "ssh -p <PORT>" data/raw_clips \
    root@<HOST>:/workspace/CourtVision/data/
  rsync -avz -e "ssh -p <PORT>" checkpoints \
    root@<HOST>:/workspace/CourtVision/

Note the destinations: rsync appends the source directory name, so sending
data/labeled/actions to data/ would land it at data/actions and every script
would report no clips.

That is ~277 MB of clips plus the detector checkpoint. Do NOT send
data/labeled/detector — it is 9.3 GB and nothing here needs it.

Then run:  bash scripts/run_gpu_suite.sh
MSG
else
  say "Ready. Run: bash scripts/run_gpu_suite.sh"
fi

cat <<'MSG'

Reminder: DESTROY the pod when you are done. Stopping it still bills for the
volume, and an idle pod is the only real cost risk here.
MSG
