FROM nvcr.io/nvidia/cuda:11.8.0-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# パッケージキャッシュの修復
# RUN rm -f /var/lib/dpkg/status && touch /var/lib/dpkg/status
# RUN rm -f /var/lib/apt/extended_states
# RUN apt-get clean
# RUN rm -rf /var/lib/apt/lists/*
# RUN apt-get update && apt-get -f install

# 必要なパッケージのインストール
# RUN apt-get update && \
#     apt-get upgrade -y && \
#     apt-get install -y \
#     software-properties-common \
#     tzdata \
#     wget \
#     ninja-build \
#     language-pack-ja 
#     # ca-certificates \
#     # curl \
#     # clang \
#     # apt-get clean && \
#     # rm -rf /var/lib/apt/lists/*

# RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y software-properties-common tzdata wget ninja-build
ENV TZ=Asia/Tokyo
RUN apt-get update && apt-get install -y language-pack-ja && \
    update-locale LANG=ja_JP.UTF-8 && rm -rf /var/lib/apt/lists/*
ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:ja
ENV LC_ALL ja_JP.UTF-8
# # タイムゾーンとロケールの設定
# ENV TZ=Asia/Tokyo
# RUN update-locale LANG=ja_JP.UTF-8
# ENV LANG=ja_JP.UTF-8
# ENV LANGUAGE=ja_JP:ja
# ENV LC_ALL=ja_JP.UTF-8

WORKDIR /opt

# RYEのインストール
ENV RYE_HOME="/opt/rye"
ENV PATH="$RYE_HOME/shims:$PATH"
ENV CUDA_HOME=/usr/local/cuda-11.8
ENV PATH=/usr/local/cuda-11.8/bin:${PATH}

SHELL [ "/bin/bash", "-o", "pipefail", "-c" ]

RUN curl -sSf https://rye.astral.sh/get | RYE_INSTALL_OPTION="--yes" bash && \
    rye config --set-bool behavior.global-python=true && \
    rye config --set-bool behavior.use-uv=true && \
    chown -R vscode $RYE_HOME

ENV PYTHONUNBUFFERED True
