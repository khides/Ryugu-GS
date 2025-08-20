# Ryugu-GS Utils Directory

このディレクトリには、Ryugu-GS プロジェクト用のデータ処理・画像変換・解析用ユーティリティスクリプトが含まれています。

## 📋 目次

1. [Hayabusa2 FITSデータダウンローダー](#1-hayabusa2-fitsデータダウンローダー)
2. [FITS to JPEG 変換ツール](#2-fits-to-jpeg-変換ツール)
3. [ヒストグラム調整ツール (CLAHE)](#3-ヒストグラム調整ツール-clahe)
4. [高度な画像マスキングツール](#4-高度な画像マスキングツール)
5. [その他のスクリプト](#5-その他のスクリプト)

---

## 1. Hayabusa2 FITSデータダウンローダー

**ファイル**: `download_hayabusa2_fits.py`

JAXA DARTS リポジトリから Hayabusa2 ONC FITS データを自動ダウンロードするスクリプトです。

### 依存パッケージ
```bash
pip install requests
```

### 基本的な使用方法

```bash
# 2005年11月2日～19日のデータをダウンロード（デフォルト）
python utils/download_hayabusa2_fits.py

# カスタム期間指定
python utils/download_hayabusa2_fits.py --start-date 2005-11-02 --end-date 2005-11-10

# 保存先ディレクトリ指定
python utils/download_hayabusa2_fits.py --output-dir ./my_hayabusa2_data
```

### 高度な使用例

```bash
# ドライラン（実際にはダウンロードせずファイル確認のみ）
python utils/download_hayabusa2_fits.py --dry-run

# 待機時間調整（サーバー負荷軽減）
python utils/download_hayabusa2_fits.py --delay 2.0 --output-dir ./data

# 特定の1日のみ
python utils/download_hayabusa2_fits.py --start-date 2005-11-02 --end-date 2005-11-02
```

### 主要オプション

| オプション | デフォルト | 説明 |
|-----------|------------|------|
| `--start-date` | 2005-11-02 | 開始日 (YYYY-MM-DD) |
| `--end-date` | 2005-11-19 | 終了日 (YYYY-MM-DD) |
| `--output-dir` | ./hayabusa2_data | 保存先ディレクトリ |
| `--delay` | 1.0 | リクエスト間待機時間（秒） |
| `--dry-run` | False | ファイル確認のみ（ダウンロードしない） |

---

## 2. FITS to JPEG 変換ツール

**ファイル**: `fits_to_jpeg.py`

天文FITS画像をJPEG形式に変換し、適切な画像ストレッチを適用するスクリプトです。

### 依存パッケージ
```bash
pip install astropy Pillow numpy
```

### 基本的な使用方法

```bash
# ディレクトリ一括変換
python utils/fits_to_jpeg.py --input-dir ./hayabusa2_data --output-dir ./jpeg_images

# 単一ファイル変換
python utils/fits_to_jpeg.py --input-file data.fits --output-file output.jpg
```

### 画像ストレッチ方法

```bash
# MinMax正規化（推奨、Hayabusa2データ用）
python utils/fits_to_jpeg.py --input-dir ./data --output-dir ./images --stretch minmax

# ZScale正規化（天文画像標準）
python utils/fits_to_jpeg.py --input-dir ./data --output-dir ./images --stretch zscale

# パーセンタイル正規化（ノイズ除去効果）
python utils/fits_to_jpeg.py --input-dir ./data --output-dir ./images --stretch percentile
```

### 高度な使用例

```bash
# Hayabusa2用フィルタリング（'tvf'を含むファイルのみ）
python utils/fits_to_jpeg.py --input-dir ./data --output-dir ./images --filter-tvf

# 高品質JPEG出力
python utils/fits_to_jpeg.py --input-dir ./data --output-dir ./images --quality 100

# フラット出力（ディレクトリ構造なし）
python utils/fits_to_jpeg.py --input-dir ./nested_data --output-dir ./flat_images --flat-output
```

### 主要オプション

| オプション | デフォルト | 説明 |
|-----------|------------|------|
| `--input-dir` | - | 入力ディレクトリ |
| `--input-file` | - | 入力FITSファイル（単一ファイル用） |
| `--output-dir` | ./jpeg_output | 出力ディレクトリ |
| `--output-file` | - | 出力JPEGファイル（単一ファイル用） |
| `--stretch` | minmax | 画像ストレッチ方法 (minmax/zscale/percentile) |
| `--quality` | 95 | JPEG品質 (1-100) |
| `--filter-tvf` | False | 'tvf'を含むファイルのみ処理 |
| `--flat-output` | False | フラット出力構造 |

---

## 3. ヒストグラム調整ツール (CLAHE)

**ファイル**: `adjust_histogram.py`

CLAHE（Contrast Limited Adaptive Histogram Equalization）を用いて画像のコントラストを適応的に調整するスクリプトです。

### 依存パッケージ
```bash
pip install opencv-python numpy
```

### 基本的な使用方法

```bash
# ディレクトリ一括処理
python utils/adjust_histogram.py --input-dir ./jpeg_images --output-dir ./enhanced_images

# 単一ファイル処理
python utils/adjust_histogram.py --input-file image.jpg --output-file enhanced.jpg
```

### パラメータ調整

```bash
# 強いコントラスト調整
python utils/adjust_histogram.py --input-dir ./images --output-dir ./enhanced --clip-limit 40.0

# 細かいタイル分割（ローカル調整強化）
python utils/adjust_histogram.py --input-dir ./images --output-dir ./enhanced --tile-size 8

# パラメータ組み合わせ
python utils/adjust_histogram.py --input-dir ./images --output-dir ./enhanced --clip-limit 30.0 --tile-size 12
```

### 高度な使用例

```bash
# フラット出力構造
python utils/adjust_histogram.py --input-dir ./nested_images --output-dir ./flat_enhanced --flat-output

# 微調整（ソフトな調整）
python utils/adjust_histogram.py --input-dir ./images --output-dir ./soft_enhanced --clip-limit 10.0 --tile-size 32
```

### 主要オプション

| オプション | デフォルト | 説明 |
|-----------|------------|------|
| `--input-dir` | - | 入力ディレクトリ |
| `--input-file` | - | 入力画像ファイル（単一ファイル用） |
| `--output-dir` | ./enhanced_images | 出力ディレクトリ |
| `--output-file` | - | 出力画像ファイル（単一ファイル用） |
| `--clip-limit` | 20.0 | CLAHEクリップ制限値（高い値=強いコントラスト） |
| `--tile-size` | 16 | タイルグリッドサイズ（NxN、小さい値=ローカル調整強化） |
| `--flat-output` | False | フラット出力構造 |

---

## 4. 高度な画像マスキングツール

**ファイル**: `image_masking.py`

複数のマスキング手法を提供する統合画像マスキングツールです。AI自動マスキングから手動マスクまで対応。

### 依存パッケージ

```bash
# 基本機能
pip install opencv-python numpy

# AI マスキング用（オプション）
# Detectron2のインストール: https://detectron2.readthedocs.io/en/latest/tutorials/install.html
pip install torch torchvision
# その後、Detectron2をインストール
```

### マスキング手法

#### 1. AI自動マスキング (detectron2)

```bash
# 基本的なAI自動マスキング
python utils/image_masking.py --input-dir ./images --output-dir ./masked --method detectron2

# 透明背景で出力
python utils/image_masking.py --input-dir ./images --output-dir ./masked --method detectron2 --output-mode transparent

# 検出閾値調整（より厳しい検出）
python utils/image_masking.py --input-dir ./images --output-dir ./masked --method detectron2 --detection-threshold 0.7
```

#### 2. 色範囲マスキング (color-range)

```bash
# 黒背景除去（デフォルト）
python utils/image_masking.py --input-dir ./images --output-dir ./masked --method color-range

# 白背景除去
python utils/image_masking.py --input-dir ./images --output-dir ./masked --method color-range --color-lower 200,200,200 --color-upper 255,255,255

# HSV色空間での緑色除去
python utils/image_masking.py --input-dir ./images --output-dir ./masked --method color-range --color-space HSV --color-lower 40,40,40 --color-upper 80,255,255
```

#### 3. 輪郭検出マスキング (contour)

```bash
# 標準的な輪郭検出
python utils/image_masking.py --input-dir ./images --output-dir ./masked --method contour

# 小さな物体除去（面積フィルタ）
python utils/image_masking.py --input-dir ./images --output-dir ./masked --method contour --min-area 5000 --max-area 50000

# 二値化閾値調整
python utils/image_masking.py --input-dir ./images --output-dir ./masked --method contour --threshold-val 100
```

#### 4. 手動マスクファイル適用 (manual)

```bash
# 単一ファイル
python utils/image_masking.py --input-file image.jpg --output-file masked.png --method manual --mask-file mask.png

# ディレクトリ一括（対応するマスクファイルが必要）
python utils/image_masking.py --input-dir ./images --output-dir ./masked --method manual --mask-dir ./masks
```

### 高度な使用例

```bash
# 色範囲マスクの反転（指定色以外をマスク）
python utils/image_masking.py --input-dir ./images --output-dir ./masked --method color-range --invert-mask

# フラット出力構造
python utils/image_masking.py --input-dir ./nested_images --output-dir ./flat_masked --method detectron2 --flat-output

# AIマスキング（黒背景モード）
python utils/image_masking.py --input-dir ./images --output-dir ./masked --method detectron2 --output-mode black_background
```

### 主要オプション

| オプション | デフォルト | 説明 |
|-----------|------------|------|
| `--method` | color-range | マスキング手法 |
| `--detection-threshold` | 0.05 | AI検出閾値（低い値=より多く検出） |
| `--output-mode` | transparent | AI出力モード（transparent/black_background/original_background） |
| `--color-lower` | 0,0,0 | 色範囲下限値 |
| `--color-upper` | 50,50,50 | 色範囲上限値 |
| `--color-space` | BGR | 色空間（BGR/HSV/LAB） |
| `--invert-mask` | False | 色範囲マスクを反転 |
| `--min-area` | 1000 | 輪郭最小面積 |
| `--max-area` | 100000 | 輪郭最大面積 |
| `--threshold-val` | 127 | 二値化閾値 |

---

## 5. その他のスクリプト

### create_appendix.py
プロジェクト用の補助ファイル作成スクリプト（既存）

### mask.py / mask_images.py
画像マスキング処理用スクリプト（既存）

### train_terst_split.py
トレーニングデータ分割スクリプト（既存）

---

## 🔄 典型的なワークフロー

### 1. Hayabusa2データの準備・処理

```bash
# Step 1: FITSデータダウンロード
python utils/download_hayabusa2_fits.py --start-date 2005-11-02 --end-date 2005-11-19 --output-dir ./raw_fits

# Step 2: FITS → JPEG変換
python utils/fits_to_jpeg.py --input-dir ./raw_fits --output-dir ./jpeg_images --filter-tvf --stretch minmax

# Step 3: 画像マスキング（必要に応じて）
python utils/image_masking.py --input-dir ./jpeg_images --output-dir ./masked_images --method color-range

# Step 4: ヒストグラム調整（必要に応じて）
python utils/adjust_histogram.py --input-dir ./masked_images --output-dir ./enhanced_images --clip-limit 25.0
```

### 2. 一般的な天文画像処理

```bash
# Step 1: FITS → JPEG変換（ZScale正規化）
python utils/fits_to_jpeg.py --input-dir ./fits_data --output-dir ./processed_images --stretch zscale --quality 100

# Step 2: コントラスト調整
python utils/adjust_histogram.py --input-dir ./processed_images --output-dir ./final_images --clip-limit 15.0 --tile-size 24
```

### 3. 単一ファイル処理パイプライン

```bash
# 単一FITSファイルの完全処理
python utils/fits_to_jpeg.py --input-file observation.fits --output-file temp.jpg --stretch minmax
python utils/image_masking.py --input-file temp.jpg --output-file temp_masked.png --method color-range
python utils/adjust_histogram.py --input-file temp_masked.png --output-file final_processed.jpg --clip-limit 30.0
```

---

## 📊 パラメータ推奨値

### FITS to JPEG変換

| データ種別 | stretch | quality | 備考 |
|-----------|---------|---------|------|
| Hayabusa2 | minmax | 95 | `--filter-tvf`推奨 |
| 一般天文画像 | zscale | 90-100 | 高ダイナミックレンジ |
| ノイズ多 | percentile | 85-95 | 外れ値除去効果 |

### 画像マスキング

| データ種別 | method | パラメータ | 備考 |
|-----------|--------|------------|------|
| 天文画像 | color-range | color-upper: 30,30,30 | 暗い背景除去 |
| 一般写真 | detectron2 | threshold: 0.05 | AI自動検出 |
| 高コントラスト | contour | threshold-val: 100 | 輪郭明確 |
| 手動制御 | manual | - | 精密マスク |

### CLAHE調整

| 用途 | clip-limit | tile-size | 備考 |
|-----|------------|-----------|------|
| 標準調整 | 20.0 | 16 | バランス良好 |
| 強調強化 | 30.0-40.0 | 8-12 | 細部強調 |
| ソフト調整 | 10.0-15.0 | 24-32 | 自然な仕上がり |
| 極端調整 | 50.0+ | 4-8 | 実験的用途 |

---

## ⚠️ 注意事項

1. **大量データ処理**: 処理時間とディスク容量に注意
2. **FITS形式**: HDU構造がデータによって異なる場合があります
3. **メモリ使用量**: 大きな画像ファイル処理時は十分なRAMを確保
4. **バックアップ**: 重要なデータは処理前にバックアップ推奨

---

## 🐛 トラブルシューティング

### よくある問題

**Q: FITSファイルが正しく読み込めない**
```bash
# HDU構造確認
python -c "from astropy.io import fits; print(fits.info('file.fits'))"
```

**Q: JPEG変換でノイズだらけの画像になる**
- `--stretch minmax` を試す
- 特定のHDUインデックスを確認
- データ範囲を手動確認

**Q: CLAHE処理が効果的でない**
- `--clip-limit` を 30.0-40.0 に増加
- `--tile-size` を 8-12 に減少
- 元画像のヒストグラム分布を確認

---

## 📞 サポート

問題や改善要望がある場合は、プロジェクトのIssue tracker または開発者に連絡してください。

---

*Last updated: 2025-08-20*