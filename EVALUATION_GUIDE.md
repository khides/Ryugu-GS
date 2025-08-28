# Rendering Methods Evaluation Guide

このガイドでは、`evaluate_rendering_methods.py`スクリプトを使用して、Blenderメッシュベースレンダリングと3D Gaussian Splattingの性能比較を行う方法を説明します。

## 📋 概要

このスクリプトは以下の評価を実行します：

- **Blender（メッシュベース）**: レンダリング時間と画質精度の評価
- **Gaussian Splatting（ニューラル）**: 学習時間、レンダリング時間、画質精度の評価
- **公平な比較**: 学習用と評価用データセットの適切な分離

## 🏗️ 必要なディレクトリ構造

### 基本構造
```
project_root/
├── evaluate_rendering_methods.py      # メイン評価スクリプト
├── gaussian-splatting/                # Gaussian Splatting実装
│   ├── train.py
│   ├── render.py
│   └── utils/
├── data_input/
│   ├── BOX-A_train/                   # 学習用データセット（GSのみ）
│   │   ├── images/                    # 学習用画像
│   │   ├── sparse/                    # COLMAP再構成
│   │   └── transforms_train.json      # カメラポーズ（NeRF形式の場合）
│   └── BOX-A_test/                    # 評価用データセット（両手法）
│       ├── images/                    # 評価用画像（Ground Truth）
│       └── ...
├── blender_data/
│   ├── ryugu_render_0001.png          # Blenderレンダリング画像
│   ├── ryugu_render_0002.png
│   ├── ...
│   └── render_times.csv               # レンダリング時間記録
└── evaluation_results/                # 出力ディレクトリ（自動作成）
    ├── comparison_results.csv
    ├── evaluation_stats.json
    └── gs_model/                      # 学習済みGSモデル
```

### データセット準備

#### 1. 学習用・評価用データセットの分割

既存の統合データセットがある場合、以下のスクリプトで分割できます：

```python
# utils/train_test_split.py を参考に
from sklearn.model_selection import train_test_split
import shutil
import os

source_dir = 'data_input/BOX-A_full/images'
train_dir = 'data_input/BOX-A_train/images' 
test_dir = 'data_input/BOX-A_test/images'

# 80:20の割合で分割
file_paths = [f for f in os.listdir(source_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
train_files, test_files = train_test_split(file_paths, test_size=0.2, random_state=42)

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

for file in train_files:
    shutil.copy(os.path.join(source_dir, file), train_dir)
for file in test_files:
    shutil.copy(os.path.join(source_dir, file), test_dir)
```

#### 2. Blenderレンダリング時間CSVの形式

`blender_data/render_times.csv`：

```csv
frame,time_sec
ryugu_render_0001,1.23
ryugu_render_0002,1.45
ryugu_render_0003,1.12
...
```

または：

```csv
filename,time
ryugu_render_0001.png,1.23
ryugu_render_0002.png,1.45
...
```

## 🚀 実行方法

### 基本実行

```bash
# デフォルト設定で実行
python evaluate_rendering_methods.py
```

### カスタム設定での実行

```bash
# 全パラメータを指定
python evaluate_rendering_methods.py \
    --train-data data_input/BOX-A_train \
    --test-data data_input/BOX-A_test \
    --blender-data blender_data \
    --gs-dir gaussian-splatting \
    --output evaluation_results
```

### パラメータ説明

| パラメータ | デフォルト値 | 説明 |
|-----------|-------------|------|
| `--train-data` | `data_input/BOX-A_train` | GS学習用データセット |
| `--test-data` | `data_input/BOX-A_test` | 評価用データセット |
| `--blender-data` | `blender_data` | Blenderレンダリングデータ |
| `--gs-dir` | `gaussian-splatting` | GS実装ディレクトリ |
| `--output` | `evaluation_results` | 結果出力ディレクトリ |

## 📊 出力結果

### 1. CSV結果ファイル（`comparison_results.csv`）

```csv
frame_filename,blender_render_time_sec,blender_psnr,blender_ssim,blender_lpips,gs_render_time_sec,gs_psnr,gs_ssim,gs_lpips
image_001.png,1.23,28.45,0.8123,0.1234,0.1,31.20,0.8765,0.0987
image_002.png,1.45,27.89,0.8045,0.1345,0.1,30.87,0.8654,0.1012
...
```

### 2. 統計情報（`evaluation_stats.json`）

```json
{
  "gs_training_time_sec": 3600.5,
  "blender_results_count": 150,
  "gs_results_count": 150,
  "total_evaluation_frames": 150
}
```

### 3. コンソール出力例

```
================================================================================
RENDERING METHODS EVALUATION SUMMARY
================================================================================

BLENDER (Mesh-based) RESULTS:
  Frames evaluated: 150
  Average render time: 1.234 sec/frame
  Average PSNR: 28.45 dB
  Average SSIM: 0.8123
  Average LPIPS: 0.1234

GAUSSIAN SPLATTING RESULTS:
  Total training time: 3600.5 sec
  Frames evaluated: 150
  Average render time: 0.100 sec/frame
  Average PSNR: 31.20 dB
  Average SSIM: 0.8765
  Average LPIPS: 0.0987

================================================================================
```

## ⚙️ 設定のカスタマイズ

### 画像前処理パラメータの調整

```python
# スクリプト内のImagePreprocessorクラス
class ImagePreprocessor:
    def __init__(self, target_bg_ratio: float = 0.3):  # 背景黒以外の目標割合
        self.target_bg_ratio = target_bg_ratio
        # ...
```

### メトリクス計算のカスタマイズ

```python
# 独自のメトリクス追加例
def calculate_custom_metric(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
    # カスタムメトリクスの実装
    return metric_value
```

## 🔧 依存関係

### 必須パッケージ

```bash
pip install torch torchvision opencv-python pillow numpy tqdm scikit-learn
```

### Gaussian Splatting固有の依存関係

```bash
# Gaussian Splattingディレクトリで実行
cd gaussian-splatting
pip install -e submodules/diff-gaussian-rasterization
pip install -e submodules/simple-knn
```

## 🐛 トラブルシューティング

### よくある問題と解決方法

#### 1. CUDA メモリ不足
```bash
# GPU メモリを節約する設定
python evaluate_rendering_methods.py --gs-settings "--resolution 4 --data_device cpu"
```

#### 2. レンダリング時間CSVが見つからない
- `blender_data/render_times.csv`が存在することを確認
- CSVのヘッダーが正しい形式であることを確認

#### 3. データセットの形式エラー
- COLMAPとNeRF形式の両方をサポート
- `sparse/`ディレクトリまたは`transforms_*.json`の存在を確認

#### 4. メトリクス計算エラー
- 画像サイズの不一致：自動リサイズで対応
- 形式の違い：PIL経由でRGB変換

### ログファイル

実行時に`evaluation.log`が生成され、詳細な実行ログが記録されます。

```bash
# ログの確認
tail -f evaluation.log
```

## 📈 結果の解釈

### メトリクス指標の意味

- **PSNR (dB)**: 高いほど良い（30+ dB が良好）
- **SSIM**: 0-1の範囲、1に近いほど良い（0.8+ が良好）  
- **LPIPS**: 低いほど良い（0.1以下が良好）

### 性能比較のポイント

1. **速度**: Blenderのレンダリング時間 vs GSの学習時間+レンダリング時間
2. **品質**: 3つのメトリクスでの総合評価
3. **用途**: リアルタイム性 vs 最終品質

## 🔄 継続的な評価

### バッチ処理での複数データセット評価

```bash
# 複数のデータセットで評価
for dataset in BOX-A BOX-B BOX-C; do
    python evaluate_rendering_methods.py \
        --train-data data_input/${dataset}_train \
        --test-data data_input/${dataset}_test \
        --output evaluation_results_${dataset}
done
```

### 結果の集計分析

```python
# 複数の結果を集計するスクリプト例
import pandas as pd
import glob

# 全結果ファイルを読み込み
results = []
for file in glob.glob("evaluation_results_*/comparison_results.csv"):
    df = pd.read_csv(file)
    dataset_name = file.split('_')[-2]
    df['dataset'] = dataset_name
    results.append(df)

combined = pd.concat(results)
summary = combined.groupby('dataset').mean()
print(summary)
```

このガイドに従って、Ryugu-GSプロジェクトでの3Dレンダリング手法の包括的な比較評価を実行できます。