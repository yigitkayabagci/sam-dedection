#!/usr/bin/env bash
# trtexec-based engine build. Convenience wrapper for the float precisions.
#
#   tools/build_trt_engine.sh <input.onnx> <output.engine> [precision] [extra trtexec args]
#
# precision: fp32 | tf32 | fp16 | bf16   (default: fp16)
#
# For INT8 use tools/build_trt_engine.py instead: trtexec can only *read* a
# calibration cache, it cannot produce one from your frames. The Python builder
# also writes a <engine>.json sidecar recording how the engine was built.
#
# The engine is hardware- and TensorRT-version-specific: build it ON THE ORIN,
# never copy one in from another machine.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <input.onnx> <output.engine> [fp32|tf32|fp16|bf16] [extra trtexec args]" >&2
  exit 2
fi

ONNX="$1"
ENGINE="$2"
PRECISION="${3:-fp16}"
if [[ $# -ge 3 ]]; then shift 3; else shift 2; fi

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

# TF32 is enabled by default on Ampere/Orin, so a plain build is really "tf32".
# --noTF32 is what makes the fp32 baseline honest.
case "${PRECISION}" in
  fp32) PRECISION_ARGS=(--noTF32) ;;
  tf32) PRECISION_ARGS=() ;;
  fp16) PRECISION_ARGS=(--fp16) ;;
  bf16) PRECISION_ARGS=(--bf16) ;;
  int8|int8-fp16|best)
    echo "ERROR: ${PRECISION} needs a calibrator. Use:" >&2
    echo "  python3 tools/build_trt_engine.py --onnx ${ONNX} --engine ${ENGINE} \\" >&2
    echo "      --precision ${PRECISION} --calib-video <clip.mp4> --calib-cache <file>" >&2
    exit 2 ;;
  *) echo "Unknown precision: ${PRECISION}" >&2; exit 2 ;;
esac

mkdir -p "$(dirname "${ENGINE}")"

echo ">> Building TRT engine"
echo "   onnx:      ${ONNX}"
echo "   engine:    ${ENGINE}"
echo "   precision: ${PRECISION} ${PRECISION_ARGS[*]:-}"

# --useCudaGraph is deliberately NOT passed here: it affects how trtexec
# *measures*, not the serialized engine. Runtime CUDA graph capture is a
# tracker option (use_cuda_graph in configs/*.yaml).
"${TRTEXEC}" \
  --onnx="${ONNX}" \
  --saveEngine="${ENGINE}" \
  "${PRECISION_ARGS[@]}" \
  --memPoolSize=workspace:4096 \
  --timingCacheFile=models/timing.cache \
  "$@"

echo ">> Engine built: ${ENGINE}"
ls -lh "${ENGINE}"
