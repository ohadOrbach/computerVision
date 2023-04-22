import cv2
import matplotlib
import numpy as np
import os
import matplotlib.pyplot as plt
import itertools

number_of_good_matches_low = 4
number_of_good_matches_high = 15
n_iterations, threshold = 1000, 10


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


def get_kp_and_des(list_of_images):
    """Compute SIFT keypoints and descriptors for a list of images"""
    sift = cv2.SIFT_create()
    kp_des = []
    for idx, img in enumerate(list_of_images):
        kp, des = sift.detectAndCompute(img, None)
        kp_des.append([kp, des])
    return kp_des


def ratio_test(kp_des):
    ratio = 0.5
    bf = cv2.BFMatcher()
    kp1, des1 = kp_des[0]
    matches_list = []
    for idx, kp_des2 in enumerate(kp_des):
        if idx != 0:
            kp2, des2 = kp_des2
            # Match the descriptors using the Brute-Force Matcher
            matches = bf.knnMatch(des1, des2, k=2)
            # Apply the ratio test to filter out false matches
            cont = True
            change1 = False
            change2 = False
            while cont:
                good_matches = []
                for m, n in matches:
                    if m.distance < ratio * n.distance:
                        good_matches.append(m)
                if len(good_matches) > number_of_good_matches_low and len(good_matches) < number_of_good_matches_high:
                    matches_list.append((idx, good_matches))
                    ratio = 0.5
                    cont = False
                elif len(good_matches) > number_of_good_matches_low:
                    ratio = ratio - 0.01
                    change1 = True
                else:
                    ratio = ratio + 0.01
                    change2 = True
                if ratio == 0.1 or ratio == 0.9 or (change1 == True and change2 == True):
                    ratio = 0.5
                    cont = False
    return matches_list


def affine_homography_transform(src_pts, dst_pts, aff=True):
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


def ransac(src_pts, dst_pts, good_matches, n_iterations, threshold, aff=True):
    best_match_transform = None
    best_match_ratio = 0
    most_inliers = 0
    rnd = 0
    arr = list(range(len(src_pts)))
    combinations = itertools.combinations(arr, 3)

    # Convert the combinations to sets to remove duplicates
    unique_combinations = set([frozenset(c) for c in combinations])
    combinations_array = [np.array(list(c)) for c in unique_combinations]
    for i in range(len(combinations_array)):
        # Randomly select a set of matches
        random_indices = np.random.choice(len(src_pts), 4, replace=False)

        # Compute the transformation
        transform = affine_homography_transform(src_pts[combinations_array[i]].copy(),
                                                dst_pts[combinations_array[i]].copy(), aff)

        if transform is not None:
            # Calculate the residuals
            residuals = calculate_residuals(src_pts, dst_pts, transform)

            # Count the number of inliers
            inliers = (residuals < threshold).sum()

            match_ratio = inliers / len(good_matches)

            if inliers > most_inliers:
                most_inliers = inliers
                best_match_transform = transform

    return best_match_transform, most_inliers


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
        best_transform, match_ratio = ransac(src_pts, dst_pts, good_matches, n_iterations, threshold, aff)
        if match_ratio > best_match_ratio:
            best_match_idx = idx
            best_match_ratio = match_ratio
            best_match_transform = best_transform
    if best_match_idx == -1:
        return None, None
    return best_match_idx, best_match_transform


def paste(img1, img2, height, width):
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


def inc_coverage_count(coverage_count, img):
    if np.count_nonzero(img):
        non_black_pixels = np.any(img != [0, 0, 0], axis=-1)
        coverage_count[non_black_pixels] += 1

    return coverage_count


def show_coverage_count(coverage_count):
    fig = plt.figure(figsize=(10, 4))
    rows = 1
    columns = 2
    fig.add_subplot(rows, columns, 1)
    plt.imshow(coverage_count), plt.axis('off'), plt.title("coverage count")
    plt.colorbar()

    fig.add_subplot(rows, columns, 2)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=2)
    plt.imshow(coverage_count, norm=norm), plt.axis('off'), plt.title("coverage count (0 1 or 2)")
    plt.colorbar()

    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_relative_images(list):
    for i, img in enumerate(list):
        plt.imshow(img), plt.title(f"relative img in index #{i}"), plt.show()


def main():
    img_files, txt_files = read_images('puzzles')
    i = 4
    transform, aff, height, width = read_transform(txt_files[i])
    img_list = img_files[i]
    img_list[0] = cv2.warpPerspective(img_list[0], transform, (width, height), flags=cv2.INTER_LINEAR)
    img_list, grey_list = update_group_list(img_list)
    coverage_count = np.zeros((height, width), dtype=np.uint8)
    coverage_count = inc_coverage_count(coverage_count, img_list[0])
    relative_list = [img_list[0]]
    while len(img_list) > 1:
        best_match_idx, best_match_transform = main_loop(grey_list, aff)
        if best_match_idx is None:
            plt.imshow(img_list[0]), plt.title(f"final #{i}"), plt.show()
            print("error in", i, "puzzle, length of image list is ", len(img_list), "\n")
            img_list = []
        else:
            print("best match is: ", best_match_idx)
            warped = Warp_source(aff, img_list[best_match_idx], best_match_transform, height, width)
            coverage_count = inc_coverage_count(coverage_count, warped)
            relative_list.append(warped)
            new_list = [paste(img_list[0], warped, height, width)]
            for j in range(len(img_list)):
                if j != 0 and j != best_match_idx:
                    new_list.append(img_list[j])

            img_list = new_list
            img_list, grey_list = update_group_list(img_list)
            # plt.imshow(img_list[0]), plt.show()

    if len(img_list) > 0:
        plt.imshow(img_list[0]), plt.title(f"final #{i}"), plt.show()
        show_coverage_count(coverage_count)
        print("we matched", len(relative_list), "pieces successfully")
        show_relative_images(relative_list)


main()
