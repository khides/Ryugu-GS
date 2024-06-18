# 3D-gaussian-splatting
## コンテナ名
- name: `intelligent_solomon`
- id: `5ccb89a1a9ea`

### Dockerコンテナ作成
- `docker build -t gaussian_splatting .`
- `docker run --gpus all -it gaussian_splatting`

### コンテナ内でAnaconda環境の作成
- `cd gaussian-splatting`
- `conda env create --file environment.yml`
- `conda activate gaussian_splatting`

### 学習の実行
- `python train.py -s ../nerf_blender_qiita`

### 結果をコピー
- `ls -tl`で更新が最も新しいものをコピーする

- `docker cp 5ccb89a1a9ea:/home/developer/3D-gaussian-splatting/output /home/hideshima/3D-gaussian-splatting`

または
- `docker cp 5ccb89a1a9ea:/home/developer/3D-gaussian-splatting/gaussian-splatting/output /home/hideshima/3D-gaussian-splatting`

### 結果を表示
- `.\viewers\bin\SIBR_gaussianViewer_app.exe -m .\output\[新しいファイル]`

### コンテナ外の変更をコンテナ内に反映
- `docker cp /home/hideshima/3D-gaussian-splatting [コンテナid]:/home/developer`

### コンテナIDを取得する
- `docker info -e`

### コンテナをストップする
- `docker stop [コンテナid]`

### コンテナを再起動する
- `docker start [コンテナid]`
- `docker exec -it [コンテナid] /bin/bash`

### コンテナを削除する
- `docker rm [コンテナid]`

### データを生成
- `python convert.py --colmap_executable "D:\GoogleDrive\ProgramFiles\Colmap\COLMAP.bat" -s E:\Git\lileaLab\GaussianSplatting\Data`