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

mkdir -p "${THIRD_PARTY}"

if [[ ! -d "${EDGETAM_DIR}" ]]; then
  echo ">> Cloning EdgeTAM into ${EDGETAM_DIR}"
  git clone --depth 1 https://github.com/facebookresearch/EdgeTAM.git "${EDGETAM_DIR}"
else
  echo ">> EdgeTAM already cloned at ${EDGETAM_DIR}"
fi

echo ">> Installing EdgeTAM (editable)"
pip install -e "${EDGETAM_DIR}"

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
