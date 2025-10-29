# NVIDIA CUDA + cuDNN + Ubuntu 22.04 ベース (Python3.10)
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# タイムゾーンなどの設定
ENV TZ=Asia/Tokyo DEBIAN_FRONTEND=noninteractive
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Python3.10 & 基本ツール
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3-pip python3.10-venv \
    git wget curl vim build-essential \
    libglib2.0-0 libgl1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# pip upgrade
RUN python3.10 -m pip install --no-cache-dir --upgrade pip

# 作業ディレクトリ
WORKDIR /root/share

# requirements.txt をコピーしてインストール
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# 開発用ツール
RUN pip install --no-cache-dir \
    twine==6.1.0 build==1.2.2.post1 pytest>=8.3.5

# jupyter lab
RUN pip install --no-cache-dir jupyterlab>=4.0

# Azure CLI
RUN apt-get update && apt-get install -y curl gnupg \
    && curl -sL https://aka.ms/InstallAzureCLIDeb | bash \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN az --version

RUN pip install  torchvision bitsandbytes
RUN pip install "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo"
RUN pip install "unsloth[base] @ git+https://github.com/unslothai/unsloth"
RUN pip install "transformers>=4.43.0,<5.0.0"
RUN pip install triton
RUN pip uninstall -y keras tensorflow tf-keras || true
# ソースをシンボリックリンク
RUN ln -s /root/share /external_code

# === 追加推奨 ENV ===
# Hugging Face のキャッシュを /root/share/hub に
ENV HUGGINGFACE_HUB_CACHE=/root/share/hub

# Transformers に TF / Flax を読ませない（Keras3問題の回避・軽量化）
ENV TRANSFORMERS_NO_TF=1
ENV TRANSFORMERS_NO_FLAX=1

# PyTorch の新しいメモリアロケータ設定（旧 PYTORCH_CUDA_ALLOC_CONF は非推奨）
ENV PYTORCH_ALLOC_CONF="max_split_size_mb:128,expandable_segments:true"

CMD ["bash"]

