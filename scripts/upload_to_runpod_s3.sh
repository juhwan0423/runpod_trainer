#!/usr/bin/env bash
# Upload raw images + SFT JSONL + code bundle to a RunPod Network Volume via
# its S3-compatible API, without needing a running (billed) GPU pod.
#
# NOTE: rclone's default S3 request signing was rejected by RunPod's gateway
# (403 SignatureDoesNotMatch) even with valid S3 API keys; aws-cli worked
# against the same credentials/endpoint, so this script uses aws-cli.
#
# Prereqs:
#   1. aws-cli installed (this session installed it to ~/.local/bin/aws)
#   2. An AWS CLI profile configured with the S3 API key from
#      RunPod console -> Settings -> S3 API Keys (NOT the general account
#      API key):
#        aws configure set aws_access_key_id     <user_...>    --profile runpod
#        aws configure set aws_secret_access_key <rps_...>     --profile runpod
#        aws configure set region                <DATACENTER>  --profile runpod   # e.g. US-CA-2, uppercase
#   3. RUNPOD_BUCKET and RUNPOD_ENDPOINT set (bucket = network volume ID)
#
# Usage:
#   RUNPOD_BUCKET=igbwvhsrj2 \
#   RUNPOD_ENDPOINT=https://s3api-us-ca-2.runpod.io/ \
#   ./scripts/upload_to_runpod_s3.sh
set -euo pipefail

AWS=${AWS_BIN:-aws}
PROFILE=${AWS_PROFILE_NAME:-runpod}
BUCKET=${RUNPOD_BUCKET:?export RUNPOD_BUCKET=<network volume ID from RunPod console>}
ENDPOINT=${RUNPOD_ENDPOINT:?export RUNPOD_ENDPOINT=https://s3api-<datacenter>.runpod.io/}

RAW_IMAGES=${RAW_IMAGES:-/data/oem4/IndustSimBenchmark/benchmark_v1_3}
SFT_DATA=${SFT_DATA:-/data/oem4/_sft_gemma4}
CODE_BUNDLE_DIR=${CODE_BUNDLE_DIR:-/data/oem4/_runpod_bundle}

S3=("$AWS" --profile "$PROFILE" --endpoint-url "$ENDPOINT" s3)

echo "[1/3] raw images: $RAW_IMAGES -> s3://$BUCKET/raw/benchmark_v1_3"
"${S3[@]}" sync "$RAW_IMAGES" "s3://$BUCKET/raw/benchmark_v1_3" \
  --only-show-errors

echo "[2/3] SFT JSONL: $SFT_DATA -> s3://$BUCKET/sft"
"${S3[@]}" sync "$SFT_DATA" "s3://$BUCKET/sft" \
  --only-show-errors

echo "[3/3] code bundle: $CODE_BUNDLE_DIR -> s3://$BUCKET/code"
"${S3[@]}" sync "$CODE_BUNDLE_DIR" "s3://$BUCKET/code" \
  --only-show-errors

echo "done. verify with: ${S3[*]} ls s3://$BUCKET/raw/benchmark_v1_3/ --recursive --summarize | tail -2"
