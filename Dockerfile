FROM nvidia/cuda:12.9.0-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common git build-essential curl wget ca-certificates \
    libglib2.0-0 libxrender-dev libsm6 libxext6 tmux \
 && add-apt-repository ppa:deadsnakes/ppa -y \
 && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-venv \
 && rm -rf /var/lib/apt/lists/*

RUN curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py \
 && python3.11 /tmp/get-pip.py \
 && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
 && ln -sf /usr/local/bin/pip3.11 /usr/local/bin/pip \
 && rm -f /tmp/get-pip.py

RUN pip install -U pip setuptools wheel ninja packaging \
    "transformers>=4.45,<4.49"

RUN pip install --no-cache-dir \
    torch==2.7.1 torchvision==0.22.* \
    --index-url https://download.pytorch.org/whl/cu128

RUN pip install einops optree regex omegaconf numpy==1.23.1 \
    pyfaidx pandas datasets tqdm matplotlib

WORKDIR /tmp/wheels
COPY wheels/*.whl ./
RUN pip install --no-index --find-links=. \
    flash_attn-2.8.0.post2+cu12torch2.7cxx11abiFALSE-cp311-cp311-linux_x86_64.whl \
    mamba_ssm-2.2.5+cu12torch2.7cxx11abiFALSE-cp311-cp311-linux_x86_64.whl \
    causal_conv1d-1.5.2+cu12torch2.7cxx11abiFALSE-cp311-cp311-linux_x86_64.whl \
 && rm -rf /tmp/wheels

WORKDIR /workspace/ldar
COPY pyproject.toml .
COPY ldar/ ldar/
COPY pretrain.py .

ENV PYTHONPATH=/workspace/ldar

RUN pip install -e . --no-deps

RUN printf '%s\n' \
    'set-option -g prefix C-a' \
    'unbind-key C-b' \
    'bind-key C-a send-prefix' \
    > /root/.tmux.conf
