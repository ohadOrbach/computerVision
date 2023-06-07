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


def generate_novel_views(left_image, right_image, disparity_map, depth_map, intrinsics_matrix, num_views=11):
    height, width = left_image.shape[:2]
    novel_views = []

    baseline_interval = 0.01  # 1 cm intervals on the baseline

    for i in range(num_views):
        baseline = (i + 1) * baseline_interval

        # Compute the translation vector for each camera position
        translation = np.array([-baseline, 0, 0])

        # Compute the extrinsics matrix for each camera position
        extrinsics = np.eye(4)
        extrinsics[:3, 3] = translation

        # Compute the inverse of the extrinsics matrix
        extrinsics_inv = np.linalg.inv(extrinsics)

        # Initialize the novel view for each extrinsics
        novel_view = np.zeros_like(left_image)

        for y in range(height):
            for x in range(width):
                disparity = disparity_map[y, x]
                depth = depth_map[y, x]

                # Skip if the depth or disparity value is invalid or zero
                if np.isnan(depth) or np.isnan(disparity) or depth == 0 or disparity == 0:
                    continue

                # Compute the 3D point in the left camera frame using the depth and disparity
                point_3d = np.dot(intrinsics_matrix, np.array([x, y, 1])) * (baseline / disparity)

                # Compute the 3D point in the right camera frame by applying the extrinsics transformation
                point_3d_right = np.dot(extrinsics_inv, np.append(point_3d, 1))[:3]

                # Compute the corresponding pixel location in the right image using the inverse intrinsics matrix
                right_pixel = np.dot(intrinsics_matrix, point_3d_right)
                right_pixel /= right_pixel[2]

                right_x = int(right_pixel[0])
                right_y = int(right_pixel[1])

                # Copy the pixel values from the right image to the novel view
                if 0 <= right_x < width and 0 <= right_y < height:
                    novel_view[y, x] = right_image[right_y, right_x]

        novel_views.append(novel_view)

    return novel_views


left_image = cv2.imread('example/im_left.jpg', cv2.IMREAD_COLOR)
intrinsics_matrix = read_intrinsic_matrix('example/K.txt')
left_depth_map = np.loadtxt('example/depth_left.txt', delimiter=",")
baseline = 0.1  # Baseline in meters
num_views = 11  # Number of desired views

# Generate new views
novel_views = generate_novel_views(left_image, left_depth_map, intrinsics_matrix, num_views)

# Save synthesized images
for i, novel_view in enumerate(novel_views):
    cv2.imwrite(f'synthesized_image_{i}.jpg', novel_view)
