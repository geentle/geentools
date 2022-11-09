import json
import os.path

import numpy as np
from PIL import Image
from skimage import measure
import pycocotools.mask as mask_util
from arr2rle import *

def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour

def binary_mask_to_polygon(binary_mask, tolerance=0):
    """Converts a binary mask to COCO polygon representation
    Args:
    binary_mask: a 2D binary numpy array where '1's represent the object
    tolerance: Maximum distance from original points of polygon to approximated
    polygonal chain. If tolerance is 0, the original coordinate array is returned.
    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    contours = np.subtract(contours, 1)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)
    return polygons

def get_paired_coord(coord):
    points = None
    for i in range(0, len(coord), 2):
        point = np.array(coord[i: i+2], dtype=np.int32).reshape(1, 2)
        if (points is None): points = point
        else: points = np.concatenate([points, point], axis=0)
    return points


def rle_json2labelme_json(json_path, save_path, threshold):
    with open(json_path, 'rb') as fp:
        rle_json = json.load(fp)
    fp.close()

    labelme_json = {
        "version": "5.0.1",
        "flags": {},
        "shapes": [],
        "imagePath": os.path.basename(rle_json['imagePath']),
        "imageData": None,
        "imageHeight": rle_json['rle'][0]['rle']['size'][0],
        "imageWidth": rle_json['rle'][0]['rle']['size'][1]
    }

    for item in rle_json['rle']:
        confidence = item['bbox'][-1]
        if confidence < threshold:
            continue
        one_rle = item['rle']
        # rle 转 mask 矩阵
        mask = np.array(rle2mask(one_rle)) > 0
        # Image.fromarray(mask).show()

        # mask to polygon
        poly = binary_mask_to_polygon(mask, tolerance=2)
        # print(poly[0])
        poly_paired = get_paired_coord(poly[0])

        labelme_json['shapes'].append({
            "label": "pig",
            "points": poly_paired.tolist(),
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        })

    with open(save_path, 'w') as fp:
        json.dump(labelme_json, fp)
    fp.close()


if __name__ == '__main__':
    rle_json2labelme_json(
        'train/12-Ceiling_Cam_00100200_day/12-Ceiling_Cam_00100200_day_frame@107.json',
        '12-Ceiling_Cam_00100200_day_frame@107.json',
        threshold = 0.4
    )










# arr = np.load('12-Ceiling_Cam_00100200_day_frame@107.npy')
#
#
# img = cv2.imread("12-Ceiling_Cam_00100200_day_frame@107.jpg")
# poly_0 = binary_mask_to_polygon(arr[12], tolerance=2)
# poly0_0 = get_paired_coord(poly_0[0])
# print(poly0_0)
# #
# p0_img = img
# p0_points = np.array(poly0_0, dtype=np.int32)
# cv2.polylines(p0_img, [p0_points], True, (255, 0, 0), 3)
# cv2.imwrite("poly_dog_0.jpeg", p0_img)

