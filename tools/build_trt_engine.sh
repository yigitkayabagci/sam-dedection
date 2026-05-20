#!/usr/bin/env bash
# Build a TensorRT engine from an ONNX file using trtexec (ships with JetPack).
#
# Usage:
#   tools/build_trt_engine.sh <input.onnx> <output.engine> [extra trtexec args]
#
# Defaults: FP16, static shape, 4 GiB workspace. On Orin AGX an FP16 build of
# the EdgeTAM image encoder typically takes 1-5 min and produces a ~25-40 MB
# engine. The engine is hardware-specific: build it ON THE ORIN, do not copy
# from another machine.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <input.onnx> <output.engine> [extra trtexec args]" >&2
  exit 2
fi

ONNX="$1"
ENGINE="$2"
shift 2

if ! command -v trtexec >/dev/null 2>&1; then
  # On Jetson trtexec lives in /usr/src/tensorrt/bin
  if [[ -x /usr/src/tensorrt/bin/trtexec ]]; then
    TRTEXEC=/usr/src/tensorrt/bin/trtexec
  else
    echo "trtexec not found. Install TensorRT (JetPack provides it)." >&2
    exit 1
  fi
else
  TRTEXEC=trtexec
fi

mkdir -p "$(dirname "${ENGINE}")"

echo ">> Building TRT engine"
echo "   onnx:    ${ONNX}"
echo "   engine:  ${ENGINE}"

"${TRTEXEC}" \
  --onnx="${ONNX}" \
  --saveEngine="${ENGINE}" \
  --fp16 \
  --memPoolSize=workspace:4096 \
  --useCudaGraph \
  "$@"

echo ">> Engine built: ${ENGINE}"
ls -lh "${ENGINE}"
