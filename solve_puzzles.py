import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageChops


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
    matches = []
    for i in range(len(kp_des)):
        for j in range(i + 1, len(kp_des)):
            # Match the descriptors using the Brute-Force Matcher
            matches_ij = bf.knnMatch(kp_des[i][-1], kp_des[j][-1], k=2)
            # Apply the ratio test to filter out false matches
            good_matches = []
            for m, n in matches_ij:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
            if len(good_matches) > 5:
                matches.append((i, j, good_matches))
    return matches


def affine_homography_transform(kp1, kp2, matches, aff=True):
    """
    Compute affine or perspective transformation matrix between two images
    using a set of point matches.
    """
    # Extract corresponding points
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    if (aff):
        # Compute the affine transformation matrix
        M = cv2.getAffineTransform(src_pts[:3], dst_pts[:3])
    else:
        # Compute the perspective transformation matrix
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    return M


def Warp_source(aff, img, best_transform):
    # Warp source image onto target image
    if aff:
        img = np.array(img, dtype=np.float32)
        warped = cv2.warpAffine(img, best_transform, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
        warped = np.array(warped, dtype=np.uint8)
    else:
        img = np.array(img, dtype=np.float32)
        warped = cv2.warpPerspective(img, best_transform, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
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


def calculate_residuals(kp1, kp2, matches, transform):
    """
    Calculates the residuals of a transformation on a set of point matches.

    Args:
        matches (list): A list of cv2.DMatch objects.
        transform (np.ndarray): A 3x3 transformation matrix.

    Returns:
        np.ndarray: A 1D array of residuals.
    """
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
    src_pts = np.float32([kp2[m.trainIdx].pt for m in matches])

    transformed_pts = apply_transform(src_pts, transform)

    residuals = np.sqrt(np.sum((transformed_pts - dst_pts) ** 2, axis=1))

    return residuals


def ransac(kp1, kp2, matches, n_iterations, threshold, aff=True):
    best_transform = None
    best_inliers = 0

    for i in range(n_iterations):
        # Randomly select a set of matches
        random_indices = np.random.choice(len(matches), 3, replace=False)
        random_matches = [matches[i] for i in random_indices]

        # Compute the transformation
        transform = affine_homography_transform(kp1, kp2, random_matches)

        # Calculate the residuals
        residuals = calculate_residuals(kp1, kp2, matches, transform)

        # Count the number of inliers
        inliers = (residuals < threshold).sum()

        # Update the best transformation if necessary
        if inliers > best_inliers:
            best_transform = transform
            best_inliers = inliers

    return best_transform, best_inliers


def added_image(added_images, height, width, group_index):
    zeros_img = np.zeros((height, width, 3), dtype=np.uint8)
    for i, img in enumerate(added_images):
        if np.count_nonzero(img) and i != 4:
            for i in range(len(img)):
                # iterate through columns
                for j in range(len(img[0])):
                    if not np.count_nonzero(zeros_img[i][j]):
                        zeros_img[i][j] = img[i][j]
    # zeros_img = cv2.warpPerspective(zeros_img, transform, (width, height))
    plt.imshow(zeros_img), plt.title(f"image group #{group_index}"), plt.show()
    return zeros_img


def update_list(img_list, transform, height, width):
    append_yet = 1
    k = 0
    zeros_img = np.zeros((height, width, 3), dtype=np.uint8)
    grey_list = []
    for i in range(len(img_list)):
        temp = zeros_img.copy()
        temp[:img_list[i].shape[0],:img_list[i].shape[1]] = img_list[i]
        if not np.count_nonzero(temp):
            continue
        if append_yet:
            temp = cv2.warpPerspective(temp, transform, (temp.shape[1], temp.shape[0]), flags=cv2.INTER_LINEAR)
            append_yet = 0
        img_list[k] = temp
        k += 1
        grey_list.append(cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY))
    return img_list, grey_list


def update_group_list(img_list):
    grey_list = []
    for im in img_list:
        grey_list.append(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))

    return img_list, grey_list


def read_images(main_dir):
    txt_files = []
    img_files = []
    for dirpath, dirnames, filenames in os.walk(main_dir):
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

def loop(n_iterations, threshold, img_list, transform, aff, height, width ,zeros_img):
    img_list, grey_list = update_list(img_list, transform, height, width)
    # run on the new matches base on image_0 relative place
    kp_des = get_kp_and_des(grey_list)
    matches = ratio_test(kp_des)
    added_image_list = []

    # find & merge groups of matches and
    added_images_num = 0
    img_num = len(grey_list)
    img_group = 0
    while added_images_num < img_num:
        added_images = np.empty(len(img_list), dtype=object)
        added_images[0] = img_list[0]
        img_list[0] = zeros_img
        added_images_num += 1
        for i, j, good_matches in matches:
            if i != 0 and j != 0:
                continue
            img_j = img_list[j]
            img_i = img_list[i]
            kp1, des1 = kp_des[i]
            kp2, des2 = kp_des[j]
            best_transform, best_inliers = ransac(kp1, kp2, good_matches, n_iterations, threshold, aff)
            if best_inliers > 3:
                if i != 0:
                    img_list[i] = Warp_source(aff, img_i, best_transform)
                    added_images[i] = img_list[i]
                    added_images[i] = zeros_img
                else:
                    img_list[j] = Warp_source(aff, img_j, best_transform)
                    added_images[j] = img_list[j]
                    img_list[j] = zeros_img
                added_images_num += 1

        img_group += 1
        new_image = added_image(added_images, height, width, img_group)
        added_image_list.append(new_image)
        img_list, grey_list = update_list(img_list, transform, height, width)
        kp_des = get_kp_and_des(grey_list)
        matches = ratio_test(kp_des)

    return added_image_list

def solve_puzzle(img_file, txt_file):
    n_iterations = 1000
    threshold = 10
    img_list = img_file
    transform, aff, height, width = read_transform(txt_file)
    height, width = height, width
    zeros_img = np.zeros((height, width, 3), dtype=np.uint8)

    final_list = loop(n_iterations, threshold, img_list, transform, aff, height, width, zeros_img)
    while len(final_list) > 1:
        final_list = loop(n_iterations, threshold, final_list, np.eye(3, 3), aff, height, width, zeros_img)

    return final_list[0]

    # merge the merged groups of matches that we found in the previous step
    # img_list, grey_list = update_group_list(added_image_list)
    # kp_des = get_kp_and_des(grey_list)
    # matches = ratio_test(kp_des)
    # print(matches)
    """
    TODO: 1. merge groups of matches (groups in added_image_list)
          2. mapped groups into blank pic by intersections
    """


# NOTE: This solution not ready yet.
def main():
    # Read images files
    img_files, txt_files = read_images('puzzles')

    # Solve for each image
    #for i, img_file in enumerate(img_files):
    #    solve_puzzle(img_file, txt_files[i])

    i = 4
    image = solve_puzzle(img_files[i], txt_files[i])
    plt.imshow(image), plt.title(f"final #{i}"), plt.show()

main()
