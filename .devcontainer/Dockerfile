FROM nvcr.io/nvidia/cuda:11.8.0-devel-ubuntu20.04

# Set the working directory
WORKDIR /opt

# Install necessary packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.7 \
    python3-pip \
    python3-dev \
    python3-setuptools \
    python3-wheel \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.7 as the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1
