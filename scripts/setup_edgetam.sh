#!/usr/bin/env bash
# Install EdgeTAM into ./third_party/EdgeTAM and download the checkpoint.
#
# On Jetson Orin AGX:
#   1) Install JetPack-matched torch/torchvision wheels BEFORE running this script:
#      https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/
#   2) Then: bash scripts/setup_edgetam.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY="${ROOT}/third_party"
EDGETAM_DIR="${THIRD_PARTY}/EdgeTAM"

# Resolve a working pip. Many systems (incl. Jetson) expose pip only as
# `python3 -m pip`, not a bare `pip` on PATH. Override with PIP=... if needed.
detect_pip() {
  if [[ -n "${PIP:-}" ]]; then echo "${PIP}"; return; fi
  for c in "python3 -m pip" "python -m pip" "pip3" "pip"; do
    if ${c} --version >/dev/null 2>&1; then echo "${c}"; return; fi
  done
  echo ""
}
PIP="$(detect_pip)"
if [[ -z "${PIP}" ]]; then
  echo "ERROR: no working pip found. Install pip (e.g. 'sudo apt install python3-pip')" >&2
  echo "       or set PIP, e.g.  PIP='python3 -m pip' bash scripts/setup_edgetam.sh" >&2
  exit 1
fi
echo ">> Using pip: ${PIP}"

mkdir -p "${THIRD_PARTY}"

if [[ ! -d "${EDGETAM_DIR}" ]]; then
  echo ">> Cloning EdgeTAM into ${EDGETAM_DIR}"
  git clone --depth 1 https://github.com/facebookresearch/EdgeTAM.git "${EDGETAM_DIR}"
else
  echo ">> EdgeTAM already cloned at ${EDGETAM_DIR}"
fi

echo ">> Installing EdgeTAM (editable)"
${PIP} install -e "${EDGETAM_DIR}"

CKPT="${EDGETAM_DIR}/checkpoints/edgetam.pt"
if [[ ! -f "${CKPT}" ]]; then
  echo ">> Checkpoint ${CKPT} not present."
  echo "   The EdgeTAM repo ships it via Git LFS / a download script."
  if [[ -f "${EDGETAM_DIR}/checkpoints/download_ckpts.sh" ]]; then
    (cd "${EDGETAM_DIR}/checkpoints" && bash download_ckpts.sh) || true
  fi
fi

echo ">> Done. Project root: ${ROOT}"
echo "   Run: python cli.py --help"
