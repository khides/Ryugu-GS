FROM nvcr.io/nvidia/cuda:11.8.0-devel-ubuntu20.04

# Install necessary packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    clang \
    python3.7 \
    python3-pip \
    python3-dev \
    python3-setuptools \
    python3-wheel \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set Python 3.7 as the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1

# Set working directory
WORKDIR /opt

# Install rye
ENV RYE_HOME="/opt/rye"
ENV PATH="$RYE_HOME/shims:$PATH"
SHELL [ "/bin/bash", "-o", "pipefail", "-c" ]
RUN curl -sSf https://rye.astral.sh/get | RYE_INSTALL_OPTION="--yes" bash && \
    rye config --set-bool behavior.global-python=true && \
    rye config --set-bool behavior.use-uv=true && \
    chown -R vscode $RYE_HOME

# Copy requirements file and install dependencies
WORKDIR /workspace
COPY requirements.txt /workspace/requirements.txt
RUN rye sync requirements.txt

ENV PYTHONUNBUFFERED True
