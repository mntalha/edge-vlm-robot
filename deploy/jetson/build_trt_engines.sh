#!/usr/bin/env bash
# Build TensorRT engines from the ONNX files produced by scripts/export_onnx.py.
# Run ON THE JETSON (engines are specific to the GPU + TensorRT version).
#
#   bash deploy/jetson/build_trt_engines.sh exports/
#
set -euo pipefail
ONNX_DIR=${1:-exports}
OUT_DIR=${2:-engines}
TRTEXEC=${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}   # default location on JetPack
mkdir -p "$OUT_DIR"

# Our TinyVLM: 16 image tokens; text prompt ~20-40 tokens.
MIN_SEQ=17; OPT_SEQ=48; MAX_SEQ=128

echo "== vision encoder (FP16) =="
"$TRTEXEC" --onnx="$ONNX_DIR/vision.onnx" --saveEngine="$OUT_DIR/vision_fp16.engine" \
  --fp16 --minShapes=pixel_values:1x3x64x64 --optShapes=pixel_values:1x3x64x64 --maxShapes=pixel_values:4x3x64x64

echo "== decoder (FP16) =="
"$TRTEXEC" --onnx="$ONNX_DIR/decoder.onnx" --saveEngine="$OUT_DIR/decoder_fp16.engine" --fp16 \
  --minShapes=input_ids:1x${MIN_SEQ},image_embeds:1x16x192 \
  --optShapes=input_ids:1x${OPT_SEQ},image_embeds:1x16x192 \
  --maxShapes=input_ids:1x${MAX_SEQ},image_embeds:1x16x192

# trtexec prints latency/throughput summaries; re-run just the timing with:
#   $TRTEXEC --loadEngine=$OUT_DIR/vision_fp16.engine --shapes=pixel_values:1x3x64x64
echo "engines written to $OUT_DIR/"
