# Recipe only: a container with the measured configuration. No image is distributed (some dependencies are
# under non-commercial or copyleft licences; see THIRD_PARTY_LICENSES.md). Build on a machine with an NVIDIA GPU driver for
# CUDA 12.8; the CUDA extensions are compiled for TORCH_CUDA_ARCH_LIST (default 9.0, Hopper).
#
#   docker build -f env/Dockerfile -t fastpixal3d .
#   docker run --gpus all -v /path/to/weights:/weights fastpixal3d \
#       python scripts/infer.py --image /weights/example.png --output /weights/out.glb --students /weights/students ...
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ARG TORCH_CUDA_ARCH_LIST=9.0
ARG MAX_JOBS=4
ENV DEBIAN_FRONTEND=noninteractive TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} MAX_JOBS=${MAX_JOBS} \
    CONDA_DIR=/opt/conda PATH=/opt/conda/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends git wget ca-certificates build-essential \
        libgl1 libglib2.0-0 libegl1 && rm -rf /var/lib/apt/lists/*
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/m.sh \
    && bash /tmp/m.sh -b -p $CONDA_DIR && rm /tmp/m.sh

WORKDIR /opt/fastpixal3d
COPY . /opt/fastpixal3d
# conda environment, pinned packages, source builds, FlashAttention-3, patched extensions, this package
RUN bash env/install.sh
ENV PATH=/opt/conda/envs/fastpixal3d/bin:$PATH
