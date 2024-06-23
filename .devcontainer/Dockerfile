FROM nvcr.io/nvidia/cuda:11.8.0-devel-ubuntu20.04


# Install necessary packages
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
