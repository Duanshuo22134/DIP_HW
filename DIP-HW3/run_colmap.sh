#!/bin/bash
# COLMAP 3D reconstruction pipeline
# Usage:
#   bash run_colmap.sh          # sparse + dense reconstruction
#   bash run_colmap.sh --sparse-only

set -e

DATASET_PATH="data"
IMAGE_PATH="$DATASET_PATH/images"
COLMAP_PATH="$DATASET_PATH/colmap"
SPARSE_ONLY=0

if [ "${1:-}" = "--sparse-only" ]; then
    SPARSE_ONLY=1
fi

if ! command -v colmap >/dev/null 2>&1; then
    echo "COLMAP is not available in PATH. Install COLMAP and add it to PATH first."
    exit 1
fi

mkdir -p "$COLMAP_PATH/sparse"
mkdir -p "$COLMAP_PATH/dense"

echo "=== Step 1: Feature Extraction ==="
colmap feature_extractor \
    --database_path "$COLMAP_PATH/database.db" \
    --image_path "$IMAGE_PATH" \
    --ImageReader.camera_model PINHOLE \
    --ImageReader.single_camera 1

echo "=== Step 2: Feature Matching ==="
colmap exhaustive_matcher \
    --database_path "$COLMAP_PATH/database.db"

echo "=== Step 3: Sparse Reconstruction (Bundle Adjustment) ==="
colmap mapper \
    --database_path "$COLMAP_PATH/database.db" \
    --image_path "$IMAGE_PATH" \
    --output_path "$COLMAP_PATH/sparse"

if [ "$SPARSE_ONLY" = "1" ]; then
    echo "=== Sparse-only mode: skip dense reconstruction ==="
    echo "Sparse result: $COLMAP_PATH/sparse/0/"
    exit 0
fi

echo "=== Step 4: Image Undistortion ==="
colmap image_undistorter \
    --image_path "$IMAGE_PATH" \
    --input_path "$COLMAP_PATH/sparse/0" \
    --output_path "$COLMAP_PATH/dense"

echo "=== Step 5: Dense Reconstruction (Patch Match Stereo) ==="
colmap patch_match_stereo \
    --workspace_path "$COLMAP_PATH/dense"

echo "=== Step 6: Stereo Fusion ==="
colmap stereo_fusion \
    --workspace_path "$COLMAP_PATH/dense" \
    --output_path "$COLMAP_PATH/dense/fused.ply"


echo "=== Done! ==="
echo "Results:"
echo "  Sparse: $COLMAP_PATH/sparse/0/"
echo "  Dense:  $COLMAP_PATH/dense/fused.ply"
