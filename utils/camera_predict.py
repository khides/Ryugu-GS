import cv2
import numpy as np
import sqlite3
import json
import os
import logging
import shutil
from sklearn.model_selection import train_test_split
from PIL import Image

class Log():
    def __init__(self, filename) -> None:
        # ログファイルの設定
        logging.basicConfig(filename=filename, level=logging.DEBUG, 
                            format='%(asctime)s %(levelname)s:%(message)s', filemode='w')
    
    def info(self, message: str) -> None:
        print(message)
        logging.info(message)
    
    def warning(self, message: str) -> None:
        print(message)
        logging.warning(message)
    
    def error(self, message: str) -> None:
        print(message)
        logging.error(message)

class Pose():
    def __init__(self, work_dir, fx, fy, cx, cy, logfile) -> None:
        self.work_dir = work_dir
        self.K = np.array([[fx, 0, cx],
                           [0, fy, cy],
                           [0,  0,  1]])
        self.logger = Log(logfile)
        self.database_path = os.path.join(work_dir, 'database.db').replace('\\', '/')
        self.image_path =  os.path.join(work_dir, 'Input').replace('\\', '/')
        self.test_path =  os.path.join(work_dir,'blender/test').replace('\\', '/')
        self.train_path =  os.path.join(work_dir, 'blender/train').replace('\\', '/')
        self.output_path = os.path.join(work_dir, 'blender/output.json').replace('\\', '/')
        self.output_train_path = os.path.join(work_dir, 'blender/transforms_train.json').replace('\\', '/')
        self.output_test_path = os.path.join(work_dir, 'blender/transforms_test.json').replace('\\', '/')
        self.image_width = None
        self.test_files = None
        self.train_files = None
        
    def read_matches_from_colmap(self):
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        # 画像のIDとパスを取得
        cursor.execute("SELECT image_id, name FROM images")
        image_paths = {row[0]: row[1] for row in cursor.fetchall()}
        
        # 特徴点を取得
        cursor.execute("SELECT image_id, rows, cols, data FROM keypoints")
        keypoints = {}
        for image_id, rows, cols, data in cursor.fetchall():
            keypoints[image_id] = np.frombuffer(data, dtype=np.float32).reshape((rows, cols))
        
        # マッチング情報を取得
        cursor.execute("SELECT pair_id, data FROM matches")
        matches = []
        for pair_id, data in cursor.fetchall():
            if data is not None:
                image_id1 = (pair_id // 2147483647) 
                image_id2 = (pair_id % 2147483647) 
                match_data = np.frombuffer(data, dtype=np.uint32).reshape(-1, 2)
                matches.append((image_id1, image_id2, match_data))
        
        conn.close()
        return image_paths, keypoints, matches

    def extract_matched_points(self, image_id1, image_id2, keypoints, matches):
        self.logger.info(f"Trying to match points between images {image_id1} and {image_id2}")
        for img1_id, img2_id, match_data in matches:
            if (img1_id == image_id1 and img2_id == image_id2) or (img1_id == image_id2 and img2_id == image_id1):
                if image_id1 in keypoints and image_id2 in keypoints:
                    kp1 = keypoints[image_id1]
                    kp2 = keypoints[image_id2]
                    
                    valid_matches = [(m1, m2) for m1, m2 in match_data if m1 < len(kp1) and m2 < len(kp2)]
                    
                    if valid_matches:
                        pts1 = np.array([kp1[m1, :2] for m1, m2 in valid_matches])
                        pts2 = np.array([kp2[m2, :2] for m1, m2 in valid_matches])
                        self.logger.info(f"Extracted {len(valid_matches)} matched points between images {image_id1} and {image_id2}")
                        return pts1, pts2
                    else:
                        self.logger.warning(f"No valid matches found for images {image_id1} and {image_id2}")
                else:
                    self.logger.warning(f"Keypoints for images {image_id1} and/or {image_id2} not found in keypoints dictionary")
        self.logger.error(f"Failed to extract matched points between images {image_id1} and {image_id2}")
        return None, None

    def estimate_camera_pose(self, pts1, pts2):
        self.logger.info(f"Estimate camera pose")
        E, mask = cv2.findEssentialMat(pts1, pts2, self.K)
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.K)
        num_inliers = np.sum(mask)
        return R, t, num_inliers

    def create_transform_matrix(self, R, t):
        self.logger.info("Transform to matrix")
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = R
        transform_matrix[:3, 3] = t.ravel()
        return transform_matrix

    def calculate_camera_angle_x(self, image_width):
        self.logger.info("Calculate camera angle")
        fx = self.K[0, 0]
        camera_angle_x = 2 * np.arctan(image_width / (2 * fx))
        return camera_angle_x

    def extract_camera_poses(self):
        image_paths, keypoints, matches = self.read_matches_from_colmap()
        
        best_poses = {}
        global_transforms = {}
        
        # 初期カメラポーズを設定（グローバル座標系の原点として）
        first_image_id = next(iter(image_paths.keys()))
        global_transforms[first_image_id] = np.eye(4)
        
        for (image_id1, image_id2, _) in matches:
            self.logger.info("========== Start extract matched points ==========")
            pts1, pts2 = self.extract_matched_points(image_id1, image_id2, keypoints, matches)
            
            if pts1 is not None and pts2 is not None:
                self.logger.info("========== Start estimate camera pose ==========")
                R, t, num_inliers = self.estimate_camera_pose(pts1, pts2)
                transform_matrix = self.create_transform_matrix(R, t)
                
                # Check and update best pose for image_id1
                if image_id1 not in best_poses or best_poses[image_id1]['num_inliers'] < num_inliers:
                    best_poses[image_id1] = {
                        "file_path": os.path.join(self.image_path, image_paths[image_id1]).replace('\\', '/'),
                        "transform_matrix": transform_matrix.tolist(),
                        "num_inliers": num_inliers
                    }
                
                # Check and update best pose for image_id2
                if image_id2 not in best_poses or best_poses[image_id2]['num_inliers'] < num_inliers:
                    best_poses[image_id2] = {
                        "file_path": os.path.join(self.image_path, image_paths[image_id2]).replace('\\', '/'),
                        "transform_matrix": transform_matrix.tolist(),
                        "num_inliers": num_inliers
                    }
                
                # グローバル変換行列を更新
                if image_id1 not in global_transforms:
                    self.logger.info(f"global transform image {image_id1} ")
                    global_transforms[image_id1] = global_transforms[first_image_id] @ transform_matrix
                if image_id2 not in global_transforms:
                    self.logger.info(f"global transform image {image_id2}")
                    global_transforms[image_id2] = global_transforms[image_id1] @ transform_matrix
        
        if best_poses:
            first_image_path = best_poses[first_image_id]["file_path"]
            image = cv2.imread(first_image_path.replace('\\', '/'), 0)
            image_width = image.shape[1]
            camera_angle_x = self.calculate_camera_angle_x(image_width)
        else:
            camera_angle_x = 0.0
        
        frames = [{"file_path": pose["file_path"], "transform_matrix": global_transforms[image_id].tolist()} for image_id, pose in best_poses.items()]
        
        test_frames = [
            {"file_path": './test/' + os.path.splitext(os.path.basename(pose["file_path"]))[0], "transform_matrix": global_transforms[image_id].tolist()}
            for image_id, pose in best_poses.items()
            if os.path.basename(pose["file_path"]) in self.test_files
        ]
        train_frames = [
            {"file_path": './train/' + os.path.splitext(os.path.basename(pose["file_path"]))[0], "transform_matrix": global_transforms[image_id].tolist()}
            for image_id, pose in best_poses.items()
            if os.path.basename(pose["file_path"]) in self.train_files
        ]
        
        output = {
            "camera_angle_x": camera_angle_x,
            "frames": frames
        }
        test_output = {
            "camera_angle_x": camera_angle_x,
            "frames": test_frames
        }
        train_output = {
            "camera_angle_x": camera_angle_x,
            "frames": train_frames
        }
        return output, test_output, train_output

    def get_image_ids(self):
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        # 画像のIDを取得
        cursor.execute("SELECT image_id FROM images")
        image_ids = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        self.logger.info(f"Image IDs: {sorted(image_ids)}")
        # return image_ids
        
    def convert_jpeg_to_png(self,jpeg_path, png_path):
        # 画像を開く
        with Image.open(jpeg_path) as img:
            # PNG形式で保存
            img.save(png_path, 'PNG')
        os.remove(jpeg_path)


    def split(self):
        # ディレクトリが存在しない場合は作成
        os.makedirs(self.train_path, exist_ok=True)
        os.makedirs(self.test_path, exist_ok=True)

        # JPEG画像のファイルパスを全て収集
        file_paths = []
        for root, dirs, files in os.walk(self.image_path):
            for file in files:
                if file.endswith('.jpeg'):
                    file_paths.append(os.path.join(root, file))

        # 訓練データとテストデータに分割（ここでは訓練:テスト = 80:20の割合）
        train_files, test_files = train_test_split(file_paths, test_size=0.2, random_state=42)

        # 訓練データを訓練ディレクトリにコピー
        for file in train_files:
            shutil.copy(file, self.train_path)

        # テストデータをテストディレクトリにコピー
        for file in test_files:
            shutil.copy(file, self.test_path)
            
        self.test_files = os.listdir(self.test_path)
        self.train_files =  os.listdir(self.train_path)

        for file_name in self.test_files:
            if file_name.lower().endswith('.jpg') or file_name.lower().endswith('.jpeg'):
                jpeg_path = os.path.join(self.test_path, file_name)
                png_path = os.path.join(self.test_path, os.path.splitext(file_name)[0] + '.png')
                self.convert_jpeg_to_png(jpeg_path, png_path)
                print(f"Converted {jpeg_path} to {png_path}")
        for file_name in self.train_files:
            if file_name.lower().endswith('.jpg') or file_name.lower().endswith('.jpeg'):
                jpeg_path = os.path.join(self.train_path, file_name)
                png_path = os.path.join(self.train_path, os.path.splitext(file_name)[0] + '.png')
                self.convert_jpeg_to_png(jpeg_path, png_path)
                print(f"Converted {jpeg_path} to {png_path}")
    
    def write_result(self, camera_poses, train_camera_poses, test_camera_poses):
        with open(self.output_path, 'w') as f:
            json.dump(camera_poses, f, indent=4)
        with open(self.output_train_path, 'w') as f:
            json.dump(train_camera_poses, f, indent=4)
        with open(self.output_test_path, 'w') as f:
            json.dump(test_camera_poses, f, indent=4)
        self.logger.info("Completed")

if __name__ == "__main__":
    # カメラ内部パラメータ (例として焦点距離fx, fyと光学中心cx, cy)
    fx, fy, cx, cy = 9231, 9231, 512, 512

    pose = Pose(work_dir='./Ryugu_Data/Ryugu_camera_2',
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                logfile='camera_pose_estimation.log')

    pose.split()
    pose.get_image_ids()
    camera_poses, test_camera_poses, train_camera_poses = pose.extract_camera_poses()
    pose.write_result(camera_poses=camera_poses,
                      train_camera_poses=train_camera_poses,
                      test_camera_poses=test_camera_poses)
