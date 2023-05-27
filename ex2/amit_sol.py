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


def read_disparity_map(file_path):
    return cv2.imread(file_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0


def read_max_disparity(file_path):
    with open(file_path, 'r') as f:
        max_disparity = float(f.read())
    return max_disparity


def compute_depth_map(disparity_map, max_disparity):
    depth_map = (max_disparity / (disparity_map + 1e-6))
    return depth_map


def generate_new_views(left_image, right_image, left_disparity_map, right_disparity_map, intrinsics_matrix, max_disparity):
    num_views = 11
    baseline_interval = 0.1
    new_views = []
    for i in range(num_views):
        baseline = (i + 1) * baseline_interval

        translation = np.array([baseline, 0, 0])  # Adjust translation for leftward movement
        extrinsic_matrix = np.eye(4)
        extrinsic_matrix[0:3, 3] = translation
        # Calculate the inverse of the intrinsic matrix
        intrinsics_inv = np.linalg.inv(intrinsics_matrix)
        # Calculate the depth maps
        left_depth_map = compute_depth_map(left_disparity_map, max_disparity)
        right_depth_map = compute_depth_map(right_disparity_map, max_disparity)
        synthesized_image = np.zeros_like(left_image, dtype=np.uint8)
        u, v = np.meshgrid(np.arange(left_image.shape[1]), np.arange(left_image.shape[0]), indexing='xy')
        u = u.reshape(-1)
        v = v.reshape(-1)
        left_depth_map = left_depth_map.reshape(-1)
        right_depth_map = right_depth_map.reshape(-1)

        # Calculate the 3D points in the camera coordinate system
        points_3d_left = np.dot(intrinsics_inv, np.array([u, v, np.ones_like(u)])) * left_depth_map

        # Apply the extrinsic matrix transformation
        transformed_coords = np.dot(extrinsic_matrix, np.vstack((points_3d_left, np.ones_like(u))))

        # Map the 3D points back to the left camera plane
        transformed_coords /= transformed_coords[3]  # Divide by homogeneous coordinate
        transformed_coords = np.dot(intrinsics_matrix, transformed_coords[:3])

        # Convert the transformed coordinates to image coordinates
        u_new, v_new = transformed_coords[0] / transformed_coords[2], transformed_coords[1] / transformed_coords[2]
        u_new = np.round(u_new).astype(int)
        v_new = np.round(v_new).astype(int)
        # Mask out-of-bounds coordinates
        valid_indices = np.logical_and.reduce(
            (u_new >= 0, v_new >= 0, u_new < left_image.shape[1], v_new < left_image.shape[0]))
        # Copy pixel values from the left image to the synthesized image
        synthesized_image[v[valid_indices], u[valid_indices]] = left_image[v_new[valid_indices], u_new[valid_indices]]
        new_views.append(synthesized_image)

    return new_views


# Example usage
left_image = cv2.imread('example/im_left.jpg', cv2.IMREAD_COLOR)
right_image = cv2.imread('example/im_right.jpg', cv2.IMREAD_COLOR)
left_disparity_map = read_disparity_map('example/disp_left.jpg')
right_disparity_map = read_disparity_map('example/disp_right.jpg')
intrinsics_matrix = read_intrinsic_matrix('example/K.txt')
max_disparity = read_max_disparity('example/max_disp.txt')

# Generate new views
new_views = generate_new_views(left_image, right_image, left_disparity_map, right_disparity_map, intrinsics_matrix, max_disparity)

# Save synthesized images
for i, synthesized_image in enumerate(new_views):
    cv2.imwrite(f'synthesized_image_{i}.jpg', synthesized_image)
