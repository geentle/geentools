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


def masks2json(masks, bboxs, json_path, img_path):
    """

    Args:
        masks:  masks矩阵 (n, h, w) n个mask
        bboxs: bboxs矩阵 (n, 5)
        json_path: 保存json的路径
        img_path:  mask对应的原图片的路径

    Returns: None

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
            'bbox': bboxs[index].tolist()
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
    masks2json(masks, [1,2,3,4,0.8], '1.json', './1.jpg')
