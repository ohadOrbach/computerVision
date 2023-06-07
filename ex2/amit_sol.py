import cv2
import numpy as np


def read_intrinsic_matrix(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    intrinsic_matrix = np.zeros((3, 3))
    for i in range(3):
        values = lines[i].split()
        intrinsic_matrix[i] = [float(value) for value in values]

    return intrinsic_matrix


def generate_novel_views(left_image, left_depth_map, right_depth_map, intrinsics_matrix, baseline,
                         num_views=11):
    height, width = left_image.shape[:2]
    novel_views = []

    baseline_interval = 0.01  # 1 cm intervals on the baseline
    for i in range(num_views):
        # Compute the translation vector for each camera position
        translation = np.array([baseline - (num_views - i) * baseline_interval, 0, 0])

        # Compute the extrinsics matrix for each camera position
        extrinsics = np.eye(4)
        extrinsics[:3, 3] = translation

        # Compute the inverse of the extrinsics matrix
        extrinsics_inv = np.linalg.inv(extrinsics)

        # Calculate reprojection of all image coordinates to 3D
        points_3d_left = np.zeros((height, width, 3))
        points_3d_right = np.zeros((height, width, 3))
        valid_indices = np.where(~np.isnan(left_depth_map))

        for y, x in zip(valid_indices[0], valid_indices[1]):
            left_depth = left_depth_map[y, x]
            right_depth = right_depth_map[y, x]

            # Compute the 3D point in the left camera coordinate system
            point_2d = np.array([x, y, 1])
            point_3d_left = np.dot(np.linalg.inv(intrinsics_matrix), point_2d) * left_depth
            points_3d_left[y, x] = point_3d_left

            # Compute the 3D point in the right camera coordinate system
            point_3d_right = np.dot(np.linalg.inv(intrinsics_matrix), point_2d) * right_depth
            points_3d_right[y, x] = point_3d_right

        # Drop the 3D points back onto the left camera plane
        novel_view = np.zeros_like(left_image)
        valid_indices_3d = np.where(points_3d_left[:, :, 2] > 0)

        for y, x in zip(valid_indices_3d[0], valid_indices_3d[1]):
            point_3d_left = points_3d_left[y, x]

            # Transform the 3D point to the left camera frame
            point_3d_left = np.dot(extrinsics_inv, np.append(point_3d_left, 1))[:3]

            # Project the 3D point onto the left camera plane
            left_pixel = np.dot(intrinsics_matrix, point_3d_left)
            left_pixel /= left_pixel[2]

            left_x = int(round(left_pixel[0]))
            left_y = int(round(left_pixel[1]))

            if 0 <= left_x < width and 0 <= left_y < height:
                novel_view[left_y, left_x] = left_image[y, x]

        novel_views.append(novel_view)

    return novel_views



left_image = cv2.imread('example/im_left.jpg', cv2.IMREAD_COLOR)
intrinsics_matrix = read_intrinsic_matrix('example/K.txt')
left_depth_map = np.loadtxt('example/depth_left.txt', delimiter=",")
right_depth_map = np.loadtxt('example/depth_right.txt', delimiter=",")
baseline = 0.1  # Baseline in meters

# Generate new views
novel_views = generate_novel_views(left_image, left_depth_map, right_depth_map, intrinsics_matrix, baseline)

# Save synthesized images
for i, novel_view in enumerate(novel_views):
    cv2.imwrite(f'synth_{i+1}.jpg', novel_view)
