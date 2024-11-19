import struct
import pandas as pd 
import tools
import os

def read_colmap_cameras_bin_debug(file_path):
    cameras = {}
    
    with open(file_path, 'rb') as f:
        try:
            # カメラの数を読み取る
            num_cameras = struct.unpack('<Q', f.read(8))[0]
            print(f"Number of cameras: {num_cameras}")
            
            for _ in range(num_cameras):
                # カメラID
                camera_id = struct.unpack('<I', f.read(4))[0]
                
                # モデルID（PINHOLEモデルを仮定）
                model_id = struct.unpack('<I', f.read(4))[0]
                models = ["SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV", "OPENCV_FISHEYE"]
                model = models[model_id] if model_id < len(models) else f"Unknown({model_id})"
                
                # 幅と高さ
                width = struct.unpack('<I', f.read(4))[0]
                _ = struct.unpack('<I', f.read(4))[0]  # 無視される値
                height = struct.unpack('<I', f.read(4))[0]
                _ = struct.unpack('<I', f.read(4))[0]  # 無視される値
                
                # パラメータ（PINHOLEモデルの場合、fx, fy, cx, cyの4つを仮定）
                num_params = 4
                param_data = f.read(8 * num_params)
                params = struct.unpack('<' + 'd' * num_params, param_data)
                
                # 結果を保存
                cameras[camera_id] = {
                    'model': model,
                    'width': width,
                    'height': height,
                    'parameters': params,
                }
        
        except Exception as e:
            print(f"Error occurred: {e}")
            print(f"Current file offset: {f.tell()}")
            return {"error": str(e), "current_file_offset": f.tell()}
    
    return cameras


def write_colmap_cameras_bin(cameras, file_path):
    """
    辞書形式のカメラデータをCOLMAPのcameras.bin形式で書き込む

    Parameters:
        cameras (dict): カメラ情報の辞書
            {
                camera_id: {
                    'model': "PINHOLE",
                    'width': int,
                    'height': int,
                    'parameters': tuple (fx, fy, cx, cy)
                }
            }
        file_path (str): 書き込み先のファイルパス
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # モデル名からモデルIDへのマッピング
    models = {
        "SIMPLE_PINHOLE": 0,
        "PINHOLE": 1,
        "SIMPLE_RADIAL": 2,
        "RADIAL": 3,
        "OPENCV": 4,
        "OPENCV_FISHEYE": 5
    }

    with open(file_path, "wb") as f:
        # カメラの数を書き込む（64ビット符号なし整数）
        f.write(struct.pack('<Q', len(cameras)))

        for camera_id, camera_data in cameras.items():
            # カメラID（32ビット符号なし整数）
            f.write(struct.pack('<I', camera_id))
            
            # モデルID（32ビット符号なし整数）
            model_id = models.get(camera_data['model'], -1)
            if model_id == -1:
                raise ValueError(f"Unknown camera model: {camera_data['model']}")
            f.write(struct.pack('<I', model_id))
            
            # 幅と高さ（32ビット符号なし整数）
            f.write(struct.pack('<I', camera_data['width']))
            f.write(struct.pack('<I', 0))  # 予約領域（常に0）
            f.write(struct.pack('<I', camera_data['height']))
            f.write(struct.pack('<I', 0))  # 予約領域（常に0）
            
            # カメラパラメータ（64ビット浮動小数点数の配列）
            params = camera_data['parameters']
            num_params = len(params)
            f.write(struct.pack('<' + 'd' * num_params, *params))

cameras = {
    1: {
        'model': "PINHOLE",
        'width': 1024,
        'height': 768,
        'parameters': (500.0, 500.0, 512.0, 384.0)
    },
    2: {
        'model': "SIMPLE_RADIAL",
        'width': 1920,
        'height': 1080,
        'parameters': (1200.0, 960.0, 540.0, 540.0)
    }
}
write_colmap_cameras_bin(cameras, "./data_input/BOX-A/sparse/cameras.bin")
# Attempt parsing the file with debugging

cameras_data_debug = read_colmap_cameras_bin_debug("./data_input/BOX-A/sparse/cameras.bin")



# If the result is an error, display it; otherwise, process the data

if "error" in cameras_data_debug:

    print(cameras_data_debug)  # Display error details for debugging
    

else:

    print(cameras_data_debug)

    print("success")
