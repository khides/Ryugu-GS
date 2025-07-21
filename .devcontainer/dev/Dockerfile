FROM python:3.10

USER root
ENV DEBIAN_FRONTEND=noninteractive
ENV VCG_ROOT=/vcglib

# 1) 必要なビルドツール・ライブラリをまとめてインストール
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

# 2) VCGLib をヘッダーのみクローン
RUN git clone https://github.com/cdcseacave/VCG.git /vcglib

# 3) OpenMVS をクローン→ソースにパッチ→ビルド→インストール
RUN git clone https://github.com/cdcseacave/openMVS.git /openMVS && \
    # JPEG XL 定数名を OpenCV 標準に差し替え
    sed -i 's/IMWRITE_JPEGXL_QUALITY/IMWRITE_JPEG_QUALITY/g' /openMVS/libs/Common/Types.inl && \
    # CGAL ヘッダー名を修正（Traits と Triangle Primitive の両方）
    sed -i -e 's#<CGAL/AABB_traits_3.h>#<CGAL/AABB_traits.h>#g' \
           -e 's#<CGAL/AABB_triangle_primitive_3.h>#<CGAL/AABB_triangle_primitive.h>#g' \
           /openMVS/libs/MVS/SceneReconstruct.cpp && \
    mkdir -p /openMVS/build && \
    cd /openMVS/build && \
    cmake .. \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr/local && \
    make -j$(nproc) && \
    make install

WORKDIR /root/workspace
