FROM python:3.10

USER root
ENV DEBIAN_FRONTEND=noninteractive

# 1) 必要なビルドツール・ライブラリを一括インストール
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential git cmake \
      ca-certificates software-properties-common \
      libpng-dev libjpeg-dev libtiff-dev libglu1-mesa-dev \
      libxi-dev libxrandr-dev libxxf86vm-dev \
      libboost-iostreams-dev libboost-program-options-dev libboost-system-dev libboost-serialization-dev \
      libopencv-dev libeigen3-dev \
      libcgal-dev libcgal-qt5-dev \
      libatlas-base-dev libsuitesparse-dev libceres-dev \
      freeglut3-dev libglew-dev libglfw3-dev libjxl-dev && \
    pip install --upgrade pip && \
    rm -rf /var/lib/apt/lists/*

# 2) VCG_ROOT を環境変数として指定
ENV VCG_ROOT=/openMVS/libs/VCG

# 3) OpenMVS をサブモジュール込みでクローン→ビルド→インストール
RUN rm -rf /openMVS && \
    git clone --recurse-submodules https://github.com/cdcseacave/openMVS.git /openMVS && \
    cd /openMVS && \
    mkdir -p build && cd build && \
    cmake .. \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr/local && \
    make -j$(nproc) && \
    make install

WORKDIR /root/workspace
