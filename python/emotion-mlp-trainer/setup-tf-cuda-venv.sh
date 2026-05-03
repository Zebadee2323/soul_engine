#!/usr/bin/env bash
set -e

ENV_NAME=${1:-.venv}

echo ">>> Creating venv: $ENV_NAME"

# Create venv with Python 3.12
uv venv --python 3.12 "$ENV_NAME"

# Activate it
source "$ENV_NAME/bin/activate"

echo ">>> Installing TensorFlow + CUDA (this may take a while first time)..."
uv pip install "tensorflow[and-cuda]"

echo ">>> Configuring LD_LIBRARY_PATH in activate script..."

cat >> "$ENV_NAME/bin/activate" <<'EOF'

# --- TensorFlow CUDA setup ---
SITE=$(python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:\
$SITE/nvidia/cuda_runtime/lib:\
$SITE/nvidia/cublas/lib:\
$SITE/nvidia/cudnn/lib:\
$SITE/nvidia/cufft/lib:\
$SITE/nvidia/curand/lib:\
$SITE/nvidia/cusolver/lib:\
$SITE/nvidia/cusparse/lib:\
$SITE/nvidia/nccl/lib:\
$SITE/nvidia/cuda_cupti/lib:\
$SITE/nvidia/cuda_nvrtc/lib:\
$SITE/nvidia/nvjitlink/lib:\
$LD_LIBRARY_PATH
# --- end TensorFlow CUDA setup ---
EOF

echo ">>> Done!"
echo
echo "Activate with:"
echo "  source $ENV_NAME/bin/activate"
echo
echo "Test GPU with:"
echo '  python -c "import tensorflow as tf; print(tf.config.list_physical_devices(\"GPU\"))"'
