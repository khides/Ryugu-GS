FROM nvcr.io/nvidia/cuda:11.8.0-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y software-properties-common tzdata wget ninja-build
ENV TZ=Asia/Tokyo
RUN apt-get update && apt-get install -y language-pack-ja && \
    update-locale LANG=ja_JP.UTF-8 && rm -rf /var/lib/apt/lists/*
ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:ja
ENV LC_ALL ja_JP.UTF-8

ENV CUDA_HOME /usr/local/cuda-11.8
ENV PATH /usr/local/cuda-11.8/bin:${PATH}


RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    clang && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /opt

ENV RYE_HOME="/opt/rye"
ENV PATH="$RYE_HOME/shims:$PATH"

SHELL [ "/bin/bash", "-o", "pipefail", "-c" ]

RUN curl -sSf https://rye.astral.sh/get | RYE_INSTALL_OPTION="--yes" bash && \
    rye config --set-bool behavior.global-python=true && \
    rye config --set-bool behavior.use-uv=true && \
    chown -R vscode $RYE_HOME

ENV PYTHONUNBUFFERED True
