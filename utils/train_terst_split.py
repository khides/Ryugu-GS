import os
import shutil
from sklearn.model_selection import train_test_split

# 元のディレクトリと目的のディレクトリのパス
source_dir = '../data_input/BOX-A_full/Input'
train_dir = '../data_input/BOX-A_train/Input'
test_dir = '../data_input/BOX-A_query/Input'

# ディレクトリが存在しない場合は作成
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# JPEG画像のファイルパスを全て収集
file_paths = []
for root, dirs, files in os.walk(source_dir):
    for file in files:
        if file.endswith('.jpeg'):
            file_paths.append(os.path.join(root, file))

# 訓練データとテストデータに分割（ここでは訓練:テスト = 80:20の割合）
train_files, test_files = train_test_split(file_paths, test_size=0.2, random_state=42)

# 訓練データを訓練ディレクトリにコピー
for file in train_files:
    shutil.copy(file, train_dir)

# テストデータをテストディレクトリにコピー
for file in test_files:
    shutil.copy(file, test_dir)
