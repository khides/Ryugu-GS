# 基本イメージを指定
# FROM nvidia/cuda:11.8-base-ubuntu20.04
FROM nvcr.io/nvidia/cuda:11.8.0-devel-ubuntu20.04

# 必要なパッケージのインストール

# RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    # libglib2.0-0 libxext6 libsm6 libxrender1 \
    # git mercurial subversion sudo
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y software-properties-common tzdata wget ninja-build
ENV TZ=Asia/Tokyo
RUN apt-get update && apt-get install -y language-pack-ja && \
    update-locale LANG=ja_JP.UTF-8 && rm -rf /var/lib/apt/lists/*
ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:ja
ENV LC_ALL ja_JP.UTF-8

# ユーザーの追加
ARG USER=developer
RUN useradd -m -s /bin/bash ${USER} && \
    echo "${USER}:${USER}" | chpasswd && \
    usermod -aG sudo ${USER}

# ユーザーを変更
# USER root
# RUN chmod -R 777 /home/${USER}/3D-gaussian-splatting
USER ${USER}


# 作業ディレクトリの設定
WORKDIR /home/${USER}/3D-gaussian-splatting

# ソースコードをコンテナにコピーする
COPY . .

#  ユーザーを変更
USER root
RUN chmod -R 777 /home/${USER}/3D-gaussian-splatting
USER ${USER}

# 環境変数の設定
ENV PATH /home/${USER}/conda/bin:$PATH
ENV CUDA_HOME /usr/local/cuda-11.8
ENV PATH /usr/local/cuda-11.8/bin:${PATH}

# Anacondaのインストール
RUN wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh -O ~/anaconda.sh && \
    chmod +x ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p ~/conda && \
    rm ~/anaconda.sh && \
    ~/conda/bin/conda clean --all -y && \
    echo ". ~/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# 環境ファイルからConda環境の作成
RUN . ~/conda/etc/profile.d/conda.sh
#  && \
    # conda env create -f /home/${USER}/3D-gaussian-splatting/gaussian-splatting/environment.yml
# 作成した環境をアクティブにする
# SHELL ["conda", "run", "-n", "gaussian_splatting", "/bin/bash", "-c"]
# Docker コンテナのデフォルトコマンド
CMD [ "/bin/bash" ]
