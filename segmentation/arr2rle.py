import numpy as np
import json
import pycocotools.mask as mask_util


def singleMask2rle(mask):
    """
    mask 转 rle
    :param mask: 二值mask
    :return: Rle json {'size':[height, width], 'counts': RLE}
    """
    rle = mask_util.encode(np.array(mask[:, :, None], order='F', dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def masks2json(masks, json_path, bboxes, img_path):
    """
    :param masks: 预测结果的多个mask矩阵
    :return: 生成为json文件
    """
    print('mask to json, start...')
    data = {
        'rle': [],
        'imagePath': img_path
    }
    for index in range(len(masks)):
        rle = singleMask2rle(masks[index])
        data['rle'].append({
            'rle': rle,
            'index': index,
            'bbox': bbox
        })
    print('generate json to ' + json_path)
    with open(json_path, 'w') as fp:
        json.dump(data, fp)
    fp.close()
    print('mask to json, end')


def rle2mask(rle):
    """
    :param rle: Rle json {'size':[height, width], 'counts': RLE}
    :return: mask (np array)
    """
    mask = np.array(mask_util.decode(rle))
    return mask

if __name__ == '__main__':
    masks = np.load('12-Ceiling_Cam_00100200_day_frame@107.npy')
    print(len(masks))
    masks2json(masks, '1.json', {'bbox': [1,2,3,4], 'cnf': 0.8}, './1.jpg')
