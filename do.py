import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def read_images(main_dir):
    txt_files = []
    img_files = []
    for dirpath, dirnames, filenames in sorted(os.walk(main_dir)):
        for filename in filenames:
            if filename.endswith('.txt'):
                txt_files.append(os.path.join(dirpath, filename))
            elif 'pieces' in dirpath and filename.endswith('.jpg'):
                if not img_files or img_files[-1][0] != dirpath:
                    if len(img_files) > 0:
                        img_files[-1].pop(0)
                    img_files.append([dirpath])
                img_files[-1].append(cv2.imread(os.path.join(dirpath, filename)))
    return img_files, txt_files


def read_transform(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
        transform = np.array([[float(num) for num in line.split()] for line in lines])
    aff = True if filename.split("_")[1] == 'affine' else False
    height = int("".join([char for char in filename.split("_")[-5] if char.isdigit()]))
    width = int("".join([char for char in filename.split("_")[-2] if char.isdigit()]))
    return transform, aff, height, width

def get_kp_and_des(list_of_images):
    """Compute SIFT keypoints and descriptors for a list of images"""
    sift = cv2.SIFT_create()
    kp_des = []
    for idx, img in enumerate(list_of_images):
        kp, des = sift.detectAndCompute(img, None)
        kp_des.append([kp, des])
    return kp_des

def ratio_test(kp_des, ratio=0.7):
    bf = cv2.BFMatcher()
    kp1, des1 = kp_des[0]
    matches_list = []
    for idx, kp_des2 in enumerate(kp_des):
        if idx!=0:
            kp2, des2 = kp_des2
            # Match the descriptors using the Brute-Force Matcher
            matches = bf.knnMatch(des1, des2, k=2)
            # Apply the ratio test to filter out false matches
            good_matches = []
            for m, n in matches:
                if m.distance < ratio * n.distance:
                    good_matches.append(m)
            if len(good_matches) > 4:
                matches_list.append((idx, good_matches))
    return matches_list


def affine_homography_transform(src_pts, dst_pts, matches, aff=True):
    """
    Compute affine or perspective transformation matrix between two images
    using a set of point matches.
    """
    if (aff):
        # Compute the affine transformation matrix
        M = cv2.getAffineTransform(src_pts[:3], dst_pts[:3])
    else:
        # Compute the perspective transformation matrix
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    return M


def Warp_source(aff, img, best_transform, height, width):
    # Warp source image onto target image
    if aff:
        img = np.array(img, dtype=np.float32)
        warped = cv2.warpAffine(img, best_transform, (width, height), flags=cv2.INTER_LINEAR)
        warped = np.array(warped, dtype=np.uint8)
    else:
        img = np.array(img, dtype=np.float32)
        warped = cv2.warpPerspective(img, best_transform, (width, height), flags=cv2.INTER_LINEAR)
        warped = np.array(warped, dtype=np.uint8)
    return warped


def apply_transform(points, transform):
    """
    Applies a transformation to a set of points.

    Args:
        points (np.ndarray): A Nx2 array of (x,y) coordinates.
        transform (np.ndarray): A 3x3 transformation matrix.

    Returns:
        np.ndarray: A Nx2 array of transformed (x,y) coordinates.
    """
    # Add a third coordinate with value 1 to the points array
    points = np.hstack([points, np.ones((points.shape[0], 1))])

    # Apply the transformation matrix to the points
    transformed_points = np.dot(transform, points.T).T

    # Normalize the transformed points by dividing by their third coordinate
    points_transformed_homogeneous = np.hstack([transformed_points, np.ones((points.shape[0], 1))])
    points_normalized = points_transformed_homogeneous[:, :2] / points_transformed_homogeneous[:, 2:]

    return points_normalized


def calculate_residuals(src_pts, dst_pts, transform):
    transformed_pts = apply_transform(src_pts, transform)

    residuals = np.sqrt(np.sum((transformed_pts - dst_pts) ** 2, axis=1))

    return residuals


def ransac(src_pts, dst_pts, matches, n_iterations, threshold, aff=True):
    best_transform = None
    best_inliers = 0

    for i in range(n_iterations):
        # Randomly select a set of matches
        random_indices = np.random.choice(len(src_pts), 4, replace=False)

        # Compute the transformation
        transform = affine_homography_transform(src_pts[random_indices].copy(), dst_pts[random_indices].copy(), random_indices)

        # Calculate the residuals
        residuals = calculate_residuals(src_pts, dst_pts, transform)

        # Count the number of inliers
        inliers = (residuals < threshold).sum()

        # Update the best transformation if necessary
        if inliers > best_inliers:
            best_transform = transform
            best_inliers = inliers

    return best_transform, best_inliers

n_iterations, threshold = 1000, 10
def main_loop(img_list, aff):
    kp_des = get_kp_and_des(img_list)
    best_match_ratio = 0
    best_match_idx = -1
    best_match_transform = None
    kp1, des1 = kp_des[0]
    matches = ratio_test(kp_des)
    for idx, good_matches in matches:
        kp2, des2 = kp_des[idx]
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])
        best_transform, best_inliers = ransac(src_pts, dst_pts, good_matches, n_iterations, threshold, aff=True)
        match_ratio = best_inliers / len(good_matches)
        if match_ratio > best_match_ratio:
            best_match_idx = idx
            best_match_ratio = match_ratio
            best_match_transform = best_transform
    if best_match_idx == -1:
        return None, None
    return best_match_idx, best_match_transform

def paste(img1, img2):
    mask = img2 != 0

    # combine m1_t and m2_t using np.maximum
    result = np.zeros((height, width, 3), dtype=np.uint8)
    result = np.maximum(result, img1)
    result[mask] = img2[mask]
    return result

def update_group_list(img_list):
    grey_list = []
    for im in img_list:
        grey_list.append(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))

    return img_list, grey_list

img_files, txt_files = read_images('puzzles')
for i in range(len(img_files)):
    transform, aff, height, width = read_transform(txt_files[i])
    img_list = img_files[i]
    img_list[0] = cv2.warpPerspective(img_list[0], transform, (width, height), flags=cv2.INTER_LINEAR)
    img_list, grey_list = update_group_list(img_list)
    while len(img_list)>1:
        best_match_idx, best_match_transform = main_loop(grey_list, aff)
        if best_match_idx == None:
            plt.imshow(img_list[0]), plt.title(f"final #{i}"), plt.show()
            print(len(img_list))
        warped = Warp_source(aff, img_list[best_match_idx], best_match_transform, height, width)
        new_list = [paste(img_list[0], warped)]
        for i in range(len(img_list)):
            if i!= 0 and i!=best_match_idx:
                new_list.append(img_list[i])

        img_list = new_list
        img_list, grey_list = update_group_list(img_list)
        #plt.imshow(img_list[0]), plt.show()

    plt.imshow(img_list[0]), plt.title(f"final #{i}"), plt.show()