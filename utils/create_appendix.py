from PIL import Image, ImageDraw, ImageFont
import os
import glob
import math

def create_image_grids(directory, output_dir, grid_size=(6, 5), font_path=None, font_size=None):
    # ディレクトリ内の画像ファイルを取得
    image_files = glob.glob(os.path.join(directory, "*.jpeg"))
    if not image_files:
        print("画像が見つかりません。")
        return

    # フォントの設定（デフォルトのフォントを使用）
    if font_path is None:
        if font_size is None:
            font = ImageFont.load_default()
        else:
            font = ImageFont.truetype("arial.ttf", size=font_size) # フォントサイズを指定
    else:
        font = ImageFont.truetype(font_path, size=font_size)

    # 最初の画像でサイズを確認
    first_image = Image.open(image_files[0])
    img_width, img_height = first_image.size
    
    padding_y = 200  # ファイル名表示用の高さ
    padding_x = 50  # ファイル名表示用の幅
    img_total_height = img_height + padding_y
    img_total_width = img_width + padding_x

    # グリッド全体のサイズを計算
    grid_width = img_total_width * grid_size[0]
    grid_height = img_total_height * grid_size[1]

    # 必要なグリッド数を計算
    num_images = len(image_files)
    num_grids = math.ceil(num_images / (grid_size[0] * grid_size[1]))

    # グリッド画像を作成
    for grid_idx in range(num_grids):
        grid_image = Image.new("RGB", (grid_width, grid_height), color=(255, 255, 255))
        draw = ImageDraw.Draw(grid_image)

        # グリッドに画像を配置
        for idx in range(grid_size[0] * grid_size[1]):
            img_idx = grid_idx * grid_size[0] * grid_size[1] + idx
            if img_idx >= num_images:
                break
            img = Image.open(image_files[img_idx])
            x = (idx % grid_size[0]) * img_total_width
            y = (idx // grid_size[0]) * img_total_height
            grid_image.paste(img, (x, y))

            # ファイル名を描画
            filename = os.path.basename(image_files[img_idx])
            text_x = x + 5  # 左寄せで少し余白をつける
            text_y = y + img_height + 5  # 画像下部に余白を持たせる
            draw.text((text_x, text_y), filename, fill="black", font=font)

        # 出力ファイル名を設定して保存
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file = f"{output_dir}/{grid_idx + 1}.jpeg"
        grid_image.save(output_file)
        print(f"グリッド画像を作成しました: {output_file}")

# 使用例
create_image_grids(
    directory="./data_input/pole-18-8-24/images",
    output_dir="image_grid/BOX-B",
    grid_size=(5, 6),
    font_size=50  # 必要に応じてフォントファイルのパスを指定
)
