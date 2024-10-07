import struct

def read_points3D(file_path):
    with open(file_path, "rb") as f:
        num_points = struct.unpack("<Q", f.read(8))[0]
        points3D = {}
        for _ in range(num_points):
            point_id = struct.unpack("<Q", f.read(8))[0]
            xyz = struct.unpack("<3d", f.read(24))
            rgb = struct.unpack("<3B", f.read(3))
            error = struct.unpack("<d", f.read(8))[0]
            track_length = struct.unpack("<Q", f.read(8))[0]
            track = []
            for _ in range(track_length):
                image_id = struct.unpack("<i", f.read(4))[0]
                point2D_idx = struct.unpack("<i", f.read(4))[0]
                track.append((image_id, point2D_idx))
            points3D[point_id] = {
                "xyz": xyz,
                "rgb": rgb,
                "error": error,
                "track": track,
            }
    return points3D

points3D = read_points3D("./Ryugu_Data/Ryugu_mask_3-1/sparse/0/points3D.bin")

# points3D データをテキスト形式で保存する
with open("./Ryugu_Data/Ryugu_mask_3-1/sparse/0/points3D.txt", "w") as f:
    for point_id, point_data in points3D.items():
        xyz = " ".join(map(str, point_data["xyz"]))
        rgb = " ".join(map(str, point_data["rgb"]))
        error = str(point_data["error"])
        track = " ".join([f"{image_id},{point2D_idx}" for image_id, point2D_idx in point_data["track"]])
        f.write(f"{point_id} {xyz} {rgb} {error} {track}\n")
