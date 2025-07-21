FROM python:3.10

USER root

RUN apt-get update && \
    apt-get -y install --reinstall ca-certificates && \
    apt-get -y install software-properties-common && \
    pip install --upgrade pip && \
    apt-get install -y git cmake libpng-dev libjpeg-dev libtiff-dev libglu1-mesa-dev && \
    apt-get install -y libboost-iostreams-dev libboost-program-options-dev libboost-system-dev libboost-serialization-dev && \
    apt-get install -y libopencv-dev libcgal-dev libceres-dev  libjxl-dev 

RUN rm -rf openMVS && \
    git clone https://github.com/cdcseacave/openMVS.git && cd openMVS && \
    mkdir -p build && \
    cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && \
    make install

WORKDIR /root/workspace