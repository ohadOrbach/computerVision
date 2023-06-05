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


def generate_novel_views(reference_view, depth_map, intrinsics_matrix_inv, extrinsics_list):
    height, width = reference_view.shape[:2]
    novel_views = []

    # Construct the camera matrix P with identity rotation matrix and variable translation vectors
    camera_matrix = np.eye(3, 4)

    for extrinsics in extrinsics_list:
        # Set the translation vector T in the camera matrix
        camera_matrix[:, 3] = extrinsics[:3, 3]

        # Initialize novel_view for each extrinsics
        novel_view = np.zeros_like(reference_view)

        for y in range(height):
            for x in range(width):
                depth = depth_map[y, x]

                if not np.isinf(depth) and not np.isnan(depth) and depth > 0:
                    # Transform pixel coordinates to homogeneous coordinates
                    pixel_coords = np.array([x, y, 1])

                    # Reproject pixel coordinates to 3D using inverse intrinsics matrix
                    point_3d = depth * np.dot(intrinsics_matrix_inv, pixel_coords)

                    # Apply the camera matrix transformation
                    transformed_point = np.dot(camera_matrix, np.append(point_3d, 1))

                    # Normalize the transformed point's homogeneous coordinates
                    if transformed_point[2] != 0:
                        transformed_point /= transformed_point[2]

                        # Map the 3D point back to the reference view
                        transformed_coords = np.dot(intrinsics_matrix, transformed_point[:3])

                        # Retrieve the transformed pixel coordinates
                        x_proj, y_proj = transformed_coords[:2] / transformed_coords[2]

                        # Check if the projected coordinates are within the image boundaries
                        if 0 <= x_proj < width and 0 <= y_proj < height:
                            x_min, y_min = int(np.floor(x_proj)), int(np.floor(y_proj))
                            x_max, y_max = x_min + 1, y_min + 1

                            if x_max < width and y_max < height:
                                # Compute the depth value at the projected coordinates
                                interpolated_depth = (1 - (x_proj - x_min)) * (1 - (y_proj - y_min)) * depth_map[y_min, x_min] + \
                                                     (x_proj - x_min) * (1 - (y_proj - y_min)) * depth_map[y_min, x_max] + \
                                                     (1 - (x_proj - x_min)) * (y_proj - y_min) * depth_map[y_max, x_min] + \
                                                     (x_proj - x_min) * (y_proj - y_min) * depth_map[y_max, x_max]

                                if depth <= interpolated_depth:
                                    # Interpolate the pixel value using the neighboring pixels
                                    dx = x_proj - x_min
                                    dy = y_proj - y_min

                                    pixel_value = (1 - dx) * (1 - dy) * reference_view[y_min, x_min] + \
                                                  dx * (1 - dy) * reference_view[y_min, x_max] + \
                                                  (1 - dx) * dy * reference_view[y_max, x_min] + \
                                                  dx * dy * reference_view[y_max, x_max]

                                    novel_view[y, x] = pixel_value

        novel_views.append(novel_view)
        cv2.imshow(f'synthesized_image_{len(novel_views) - 1}.jpg', novel_view)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return novel_views


def create_extrinsics_list(baseline, num_views):
    extrinsics_list = []
    for i in range(num_views):
        # Compute the translation vector T for each camera position
        translation = np.array([i * baseline, 0, 0])

        # Construct the extrinsic matrix with identity rotation and the corresponding translation
        extrinsic_matrix = np.eye(4)
        extrinsic_matrix[:3, 3] = translation

        extrinsics_list.append(extrinsic_matrix)
    return extrinsics_list


left_image = cv2.imread('example/im_left.jpg', cv2.IMREAD_COLOR)
intrinsics_matrix = read_intrinsic_matrix('example/K.txt')
left_depth_map = np.loadtxt('example/depth_left.txt', delimiter=",")
baseline = 0.01  # Baseline in meters
num_views = 11  # Number of desired views

extrinsics_list = create_extrinsics_list(baseline, num_views)

# Generate new views
novel_views = generate_novel_views(left_image, left_depth_map, np.linalg.inv(intrinsics_matrix), extrinsics_list)
