import numpy as np
from tqdm import tqdm
import cv2

def read_intrinsic_matrix(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    intrinsic_matrix = np.zeros((3, 3))
    for i in range(3):
        values = lines[i].split()
        intrinsic_matrix[i] = [float(value) for value in values]

    return intrinsic_matrix

def generate_novel_views_bon(left_image, left_depth_map, right_image, right_depth_map, intrinsics_matrix, baseline, num_views=1):
    height, width = left_image.shape[:2]
    baseline_interval = 0.01
    novel_views = []

    for i in tqdm(range(num_views)):
        # Compute the translation vector for each camera position
        translation = np.array([baseline - (num_views - i) * baseline_interval, 0, 0])
        extrinsics = np.eye(4)
        extrinsics[:3, 3] = translation
        extrinsics_inv = np.linalg.inv(extrinsics)
        points_3d_left = np.zeros((height, width, 3))
        points_3d_right = np.zeros((height, width, 3))
        valid_indices_left = np.where(~np.isnan(left_depth_map))
        valid_indices_right = np.where(~np.isnan(right_depth_map))

        # Process valid indices in the left image
        for y, x in zip(valid_indices_left[0], valid_indices_left[1]):
            point_2d = np.array([x, y, 1])
            left_depth = left_depth_map[y, x]
            point_3d_left = np.dot(np.linalg.inv(intrinsics_matrix), point_2d) * left_depth
            points_3d_left[y, x] = point_3d_left
            # Compute the 3D point in the right camera coordinate system
            right_depth = right_depth_map[y, x]
            point_3d_right = np.dot(np.linalg.inv(intrinsics_matrix), point_2d) * right_depth
            points_3d_right[y, x] = point_3d_right

        # Process valid indices in the right image
        for y, x in zip(valid_indices_right[0], valid_indices_right[1]):
            point_2d = np.array([x, y, 1])
            right_depth = right_depth_map[y, x]
            point_3d_right = np.dot(np.linalg.inv(intrinsics_matrix), point_2d) * right_depth
            points_3d_right[y, x] = point_3d_right
            # Compute the 3D point in the left camera coordinate system
            left_depth = left_depth_map[y, x]
            point_3d_left = np.dot(np.linalg.inv(intrinsics_matrix), point_2d) * left_depth
            points_3d_left[y, x] = point_3d_left

        # Drop the 3D points back onto the left camera plane
        novel_view = np.zeros_like(left_image)
        valid_indices_3d = np.where(points_3d_left[:, :, 2] > 0)

        for y, x in zip(valid_indices_3d[0], valid_indices_3d[1]):
            point_3d_left = points_3d_left[y, x]
            point_3d_right = points_3d_right[y, x]
            # Transform the 3D points to the left and right camera frames
            point_3d_left = np.dot(extrinsics_inv, np.append(point_3d_left, 1))[:3]
            point_3d_right = np.dot(extrinsics_inv, np.append(point_3d_right, 1))[:3]
            right_pixel = np.dot(intrinsics_matrix, point_3d_right)
            if right_pixel[2] != 0:
                right_pixel /= right_pixel[2]
                right_pixel[0] += baseline * 100  # Adjust the x-coordinate with the baseline distance
            # Project the 3D points onto the left and right camera planes
            left_pixel = np.dot(intrinsics_matrix, point_3d_left)
            left_pixel /= left_pixel[2]
            right_pixel = np.dot(intrinsics_matrix, point_3d_right)
            if right_pixel[2] != 0:
                right_pixel /= right_pixel[2]

            left_x = int(round(left_pixel[0]))
            left_y = int(round(left_pixel[1]))
            right_x = int(round(right_pixel[0]))
            right_y = int(round(right_pixel[1]))

            if 0 <= left_x < width and 0 <= left_y < height:
                novel_view[left_y, left_x] = left_image[y, x]

            if 0 <= right_x < width and 0 <= right_y < height:
                if np.isnan(novel_view[right_y, right_x]):
                    novel_view[right_y, right_x] = right_image[y, x]

        novel_views.append(novel_view)

    return novel_views


left_image = cv2.imread('example/im_left.jpg', cv2.IMREAD_COLOR)
right_image = cv2.imread('example/im_right.jpg', cv2.IMREAD_COLOR)
intrinsics_matrix = read_intrinsic_matrix('example/K.txt')
left_depth_map = np.loadtxt('example/depth_left.txt', delimiter=",")
right_depth_map = np.loadtxt('example/depth_right.txt', delimiter=",")
baseline = 0.1  # Baseline in meters
num_views = 1  # Number of novel views to generate

# Initialize the synthesized views
novel_views = []
for _ in range(num_views):
    novel_views.append(np.zeros_like(left_image))

# Generate new views
novel_views = generate_novel_views_bon(left_image, left_depth_map, right_image, right_depth_map,
                                       intrinsics_matrix, baseline, num_views)

# Save synthesized images
for i, novel_view in enumerate(novel_views):
    cv2.imwrite(f'NovelView_{i+1}.jpg', novel_view)
