import cv2
import numpy as np
from scipy.ndimage import uniform_filter
from tqdm import tqdm
import os


def census_transform(img, window_size=(7, 9)):
    height, width = img.shape

    window_height, window_width = window_size
    window_size = window_height * window_width

    census = np.zeros((height, width, window_size - 1), dtype=bool)

    rows = np.arange(-(window_height // 2), window_height // 2 + 1)[:, np.newaxis]
    cols = np.arange(-(window_width // 2), window_width // 2 + 1)

    for i in tqdm(range(window_height // 2, height - window_height // 2)):
        for j in range(window_width // 2, width - window_width // 2):
            window = img[i + rows, j + cols]
            center = window[window_height // 2, window_width // 2]
            z = 0
            for k in range(window_size):
                if k != window_size // 2:
                    census[i, j, z] = (window[k // window_width, k % window_width] >= center)
                    z = z + 1

    return census


def cost_volume(left_census, right_census, max_disp):
    height, width, _ = left_census.shape

    cost_volume_L = np.zeros((height, width, max_disp), dtype=np.uint8)
    cost_volume_R = np.zeros((height, width, max_disp), dtype=np.uint8)
    for d in tqdm(range(max_disp)):
        right_census_shifted = np.roll(right_census, d, axis=1)
        cost_volume_L[:, :, d] = np.sum(left_census != right_census_shifted, axis=2)
        left_census_shifted = np.roll(left_census, -d, axis=1)
        cost_volume_R[:, :, d] = np.sum(right_census != left_census_shifted, axis=2)

    return cost_volume_L, cost_volume_R


def proc(cost_volume, filter_size=(5, 5)):
    cc = cost_volume
    filter_mode = 'reflect'
    cc2 = cc.copy()
    for i in range(cc.shape[2]):
        cc2[:, :, i] = uniform_filter(cc[:, :, i], size=filter_size, mode=filter_mode)
    min_indices = np.argmin(cc2, axis=-1)
    cc3 = min_indices.astype(np.uint8)
    return cc3

def left_right_consistency_test(disp_left, disp_right, threshold = 1):
    rows, cols = np.indices(disp_left.shape)

    x_coords = np.clip((cols - disp_left), 0, 1023)
    mask_left = (x_coords >= 0) & (x_coords < disp_right.shape[1])
    mask_left &= (np.abs(disp_left - disp_right[rows, x_coords]) <= threshold)
    disp_left[mask_left == False] = 0

    x_coords = np.clip((cols + disp_right), 0, 1023)
    mask_right = (x_coords >= 0) & (x_coords < disp_right.shape[1])
    mask_right &= (np.abs(disp_right - disp_left[rows, x_coords]) <= threshold)
    disp_right[mask_right == False] = 0

    return disp_left, disp_right

def depth(f, b, disp):
    result = np.zeros_like(disp)

    for i in tqdm(range(len(disp))):
        for j in range(len(disp[0])):
            if disp[i][j] != 0:
                result[i][j] = f*b / disp[i][j]
            else:
                result[i][j] = 0
    return result

directories = ['set_1', 'set_2', 'set_3', 'set_4', 'set_5']

for directory in directories:

    imgL = cv2.imread(os.path.join(directory, 'im_left.jpg'), cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread(os.path.join(directory, 'im_right.jpg'), cv2.IMREAD_GRAYSCALE)

    with open(os.path.join(directory, 'max_disp.txt')) as f:
        max_disp = int(f.read())

    K = np.loadtxt(os.path.join(directory, 'K.txt'))
    focal_length = K[0][0]

    base_line = 10

    window_shape = (11,13)
    filter = (7,7)
    censusL = census_transform(imgL, window_shape)
    censusR = census_transform(imgR, window_shape)
    costVL, costVR = cost_volume(censusL, censusR, max_disp)

    disp_left = proc(costVL, filter)
    disp_right = proc(costVR, filter)

    dl, dr = left_right_consistency_test(disp_left, disp_right)

    depth_left = depth(focal_length, base_line, dl)
    depth_right = depth(focal_length, base_line, dr)

    #output_dir = os.path.join('results', directory)
    #os.makedirs(output_dir, exist_ok=True)

    #cv2.imwrite(os.path.join(output_dir,'disp_l.jpg'), dl/ np.max(dl)*255)
    #cv2.imwrite(os.path.join(output_dir,'disp_r.jpg'), dr/ np.max(dr)*255)

    #cv2.imwrite(os.path.join(output_dir,'depth_l.jpg'),  depth_left/ np.max(depth_left)*255)
    #cv2.imwrite(os.path.join(output_dir,'depth_r.jpg'), depth_right/ np.max(depth_right)*255)