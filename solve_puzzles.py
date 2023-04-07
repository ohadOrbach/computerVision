import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

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
        for j in range(i+1, len(kp_des)):
            # Match the descriptors using the Brute-Force Matcher
            matches_ij = bf.knnMatch(kp_des[i][-1], kp_des[j][-1], k=2)
            # Apply the ratio test to filter out false matches
            good_matches = []
            for m,n in matches_ij:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
            if len(good_matches)>2:
                matches.append((i,j,good_matches))
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


def calculate_residuals(kp1, kp2,matches, transform):
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

    return best_transform



def main():
    main_dir = 'puzzles'
    txt_files = []
    img_files = []
    n_iterations = 1000
    threshold = 10

    for dirpath, dirnames, filenames in os.walk(main_dir):
        for filename in filenames:
            if filename.endswith('.txt'):
                txt_files.append(os.path.join(dirpath, filename))
            elif 'pieces' in dirpath and filename.endswith('.jpg'):
                if not img_files or img_files[-1][0] != dirpath:
                    if len(img_files)>0:
                        img_files[-1].pop(0)
                    img_files.append([dirpath])
                img_files[-1].append(cv2.imread(os.path.join(dirpath, filename), cv2.IMREAD_GRAYSCALE))

    img_list = img_files[4]
    filename = txt_files[4]

    with open(filename, "r") as f:
        lines = f.readlines()
        transform = np.array([[float(num) for num in line.split()] for line in lines])
    aff = True if filename.split("_")[1] == 'affine' else False
    height = int("".join([char for char in filename.split("_")[-5] if char.isdigit()]))
    width = int("".join([char for char in filename.split("_")[-2] if char.isdigit()]))
    kp_des = get_kp_and_des(img_list)

    matches = ratio_test(kp_des)
    plot_matches = True
    for i, j, good_matches in matches:
        img1 = img_list[i]
        img2 = img_list[j]
        kp1, des1 = kp_des[i]
        kp2, des2 = kp_des[j]

        if plot_matches:
            img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None,
                                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            plt.imshow(img_matches), plt.title(f"img{i}&img{j}"), plt.show()
        # Run RANSAC
        best_transform = ransac(kp1, kp2, good_matches, n_iterations, threshold, aff)

        # Print the best transformation
        print(best_transform)

        # Load target image
        target = img1

        # Warp source image onto target image
        if aff:
            warped = cv2.warpAffine(img2, best_transform, (target.shape[1], target.shape[0]), flags=cv2.INTER_LINEAR)
        else:
            warped = cv2.warpPerspective(img2, best_transform, (target.shape[1], target.shape[0]), flags=cv2.INTER_LINEAR)

        # Display warped image
        #plt.imshow(warped, cmap="gray"), plt.show()
        #plt.imshow(target, cmap="gray"), plt.show()
        # Blend the images together
        alpha = 1
        beta = 1
        blended = cv2.addWeighted(target, alpha, warped, beta, 0)
        # Paste the transformed image onto the black image
        #warped_img = cv2.warpPerspective(blended, transform, (height, width))
        # Display blended image
        plt.imshow(blended, cmap="gray"), plt.title(f"img{i}&img{j}"), plt.show()

main()