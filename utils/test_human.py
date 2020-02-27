import os
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix

# name_list = ['19: Hat', '18: Hair', '17: Glove', '16: Sunglasses', '15: UpperClothes', '14: Dress', '13: Coat', '12: Socks', '11: Pants',
#              '10: Torso-skin', '9: Scarf', '8: Skirt', '7: Face', '6: Left-arm', '5: Right-arm', '4: Left-leg', '3: Right-leg', '2: Left-shoe',
#              '1: Right-shoe', '0: BG'
#              ]


name_list = ['0: BG', '1: Hat', '2: Hair', '3: Glove', '4: Sunglasses', '5: UpperClothes', '6: Dress', '7: Coat', '8: Socks', '9: Pants',
             '10: Torso-skin', '11: Scarf', '12: Skirt', '13: Face', '14: Left-arm', '15: Right-arm', '16: Left-leg', '17: Right-leg', '18: Left-shoe',
             '19: Right-shoe'
             ]

def main():
    image_paths, label_paths = init_path()
    hist = compute_hist(image_paths, label_paths)
    show_result(hist)


def init_path():
    list_file = './human/list/val_id.txt'
    file_names = []
    with open(list_file, 'rb') as f:
        for fn in f:
            file_names.append(fn.strip())

    image_dir = './human/features/attention/val/results/'
    label_dir = './human/data/labels/'

    image_paths = []
    label_paths = []
    for file_name in file_names:
        image_paths.append(os.path.join(image_dir, file_name + '.png'))
        label_paths.append(os.path.join(label_dir, file_name + '.png'))
    return image_paths, label_paths


def fast_hist(lbl, pred, n_cls):
    '''
    compute the miou
    :param lbl: label
    :param pred: output
    :param n_cls: num of class
    :return:
    '''
    # print(n_cls)
    k = (lbl >= 0) & (lbl < n_cls)
    # print(lbl.shape)
    # print(k)
    # print(lbl[k].shape)
    # print(np.bincount(n_cls * lbl[k].astype(int) + pred[k], minlength=n_cls ** 2).shape)
    return np.bincount(n_cls * lbl[k].astype(int) + pred[k], minlength=n_cls ** 2).reshape(n_cls, n_cls)


def compute_hist(images, labels,n_cls=20):
    hist = np.zeros((n_cls, n_cls))
    for img_path, label_path in zip(images, labels):

        print(img_path)
        label = Image.open(label_path)
        label_array = np.array(label, dtype=np.int32)
        image = Image.open(img_path)
        image_array = np.array(image, dtype=np.int32)

        if '105047_427469' in img_path:
            continue

        gtsz = label_array.shape
        imgsz = image_array.shape
        if not gtsz == imgsz:
            image = image.resize((gtsz[1], gtsz[0]), Image.ANTIALIAS)
            image_array = np.array(image, dtype=np.int32)

        print(label_array.shape, image_array.shape)
        hist += fast_hist(label_array, image_array, n_cls)

    return hist


def show_result(hist):

    f = open('cihp_iou.txt', 'w+')

    classes = ['background', 'hat', 'hair', 'glove', 'sunglasses', 'upperclothes',
               'dress', 'coat', 'socks', 'pants', 'jumpsuits', 'scarf', 'skirt',
               'face', 'leftArm', 'rightArm', 'leftLeg', 'rightLeg', 'leftShoe',
               'rightShoe']
    # num of correct pixels
    num_cor_pix = np.diag(hist)
    # num of gt pixels
    num_gt_pix = hist.sum(1)
    print('=' * 50)
    print('=' * 50, file=f)

    # @evaluation 1: overall accuracy
    acc = num_cor_pix.sum() / hist.sum()
    print('>>>', 'overall accuracy', acc)
    print('>>>', 'overall accuracy', acc, file=f)
    print('-' * 50)
    print('=' * 50, file=f)

    # @evaluation 2: mean accuracy & per-class accuracy
    print('Accuracy for each class (pixel accuracy):')
    for i in range(20):
        print('%-15s: %f' % (classes[i], num_cor_pix[i] / num_gt_pix[i]))
    acc = num_cor_pix / num_gt_pix
    temp = np.nanmean(acc)
    print('>>>', 'mean accuracy', temp)

    print('>>>', 'mean accuracy', temp, file=f)
    print('-' * 50)
    print('=' * 50, file=f)

    # @evaluation 3: mean IU & per-class IU
    print('Per class miou:')
    union = num_gt_pix + hist.sum(0) - num_cor_pix
    for i in range(20):
        print('%-15s: %f' % (classes[i], num_cor_pix[i] / union[i]))
        print('%-15s: %f' % (classes[i], num_cor_pix[i] / union[i]), file=f)

    iu = num_cor_pix / (num_gt_pix + hist.sum(0) - num_cor_pix)

    temp = np.nanmean(iu)
    print('>>>', 'mean IU', temp)
    print('>>>', 'mean IU', temp, file=f)
    print('-' * 50)

    # @evaluation 4: frequency weighted IU
    freq = num_gt_pix / hist.sum()
    print('>>>', 'fwavacc', (freq[freq > 0] * iu[freq > 0]).sum())
    print('=' * 50)

    f.close()


def show_result_pascal(hist):

    f = open('pascal_iou.txt', 'w+')

    classes = ['background', 'head', 'torso', 'upper-arm', 'lower-arm', 'upper-leg',
               'lower-leg']
    # num of correct pixels
    num_cor_pix = np.diag(hist)
    # num of gt pixels
    num_gt_pix = hist.sum(1)
    print('=' * 50)
    print('=' * 50, file=f)

    # @evaluation 1: overall accuracy
    acc = num_cor_pix.sum() / hist.sum()
    print('>>>', 'overall accuracy', acc)
    print('>>>', 'overall accuracy', acc, file=f)
    print('-' * 50)
    print('=' * 50, file=f)

    # @evaluation 2: mean accuracy & per-class accuracy
    print('Accuracy for each class (pixel accuracy):')
    for i in range(7):
        print('%-15s: %f' % (classes[i], num_cor_pix[i] / num_gt_pix[i]))
    acc = num_cor_pix / num_gt_pix
    temp = np.nanmean(acc)
    print('>>>', 'mean accuracy', temp)

    print('>>>', 'mean accuracy', temp, file=f)
    print('-' * 50)
    print('=' * 50, file=f)

    # @evaluation 3: mean IU & per-class IU
    print('Per class miou:')
    union = num_gt_pix + hist.sum(0) - num_cor_pix
    for i in range(7):
        print('%-15s: %f' % (classes[i], num_cor_pix[i] / union[i]))
        print('%-15s: %f' % (classes[i], num_cor_pix[i] / union[i]), file=f)

    iu = num_cor_pix / (num_gt_pix + hist.sum(0) - num_cor_pix)

    temp = np.nanmean(iu)
    print('>>>', 'mean IU', temp)
    print('>>>', 'mean IU', temp, file=f)
    print('-' * 50)

    # @evaluation 4: frequency weighted IU
    freq = num_gt_pix / hist.sum()
    print('>>>', 'fwavacc', (freq[freq > 0] * iu[freq > 0]).sum())
    print('=' * 50)

    f.close()


def show_result_atr(hist):

    f = open('atr_iou.txt', 'w+')

    classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17"]
    # num of correct pixels
    num_cor_pix = np.diag(hist)
    # num of gt pixels
    num_gt_pix = hist.sum(1)
    print('=' * 50)
    print('=' * 50, file=f)

    # @evaluation 1: overall accuracy
    acc = num_cor_pix.sum() / hist.sum()
    print('>>>', 'overall accuracy', acc)
    print('>>>', 'overall accuracy', acc, file=f)
    print('-' * 50)
    print('=' * 50, file=f)

    # @evaluation 2: mean accuracy & per-class accuracy
    print('Accuracy for each class (pixel accuracy):')
    for i in range(18):
        print('%-15s: %f' % (classes[i], num_cor_pix[i] / num_gt_pix[i]))
    acc = num_cor_pix / num_gt_pix
    temp = np.nanmean(acc)
    print('>>>', 'mean accuracy', temp)

    print('>>>', 'mean accuracy', temp, file=f)
    print('-' * 50)
    print('=' * 50, file=f)

    # @evaluation 3: mean IU & per-class IU
    print('Per class miou:')
    union = num_gt_pix + hist.sum(0) - num_cor_pix
    for i in range(18):
        print('%-15s: %f' % (classes[i], num_cor_pix[i] / union[i]))
        print('%-15s: %f' % (classes[i], num_cor_pix[i] / union[i]), file=f)

    iu = num_cor_pix / (num_gt_pix + hist.sum(0) - num_cor_pix)

    temp = np.nanmean(iu)
    print('>>>', 'mean IU', temp)
    print('>>>', 'mean IU', temp, file=f)
    print('-' * 50)

    # @evaluation 4: frequency weighted IU
    freq = num_gt_pix / hist.sum()
    print('>>>', 'fwavacc', (freq[freq > 0] * iu[freq > 0]).sum())
    print('=' * 50)

    f.close()


def get_iou(pred,lbl,n_cls):
    '''
    need tensor cpu
    :param pred:
    :param lbl:
    :param n_cls:
    :return:
    '''
    hist = np.zeros((n_cls,n_cls))
    for i,j in zip(range(pred.size(0)),range(lbl.size(0))):
        pred_item = pred[i].data.numpy()
        lbl_item = lbl[j].data.numpy()
        hist += fast_hist(lbl_item, pred_item, n_cls)
    # num of correct pixels
    num_cor_pix = np.diag(hist)
    # num of gt pixels
    num_gt_pix = hist.sum(1)
    union = num_gt_pix + hist.sum(0) - num_cor_pix
    # for i in range(20):
    #     print('%-15s: %f' % (classes[i], num_cor_pix[i] / union[i]))
    iu = num_cor_pix / (num_gt_pix + hist.sum(0) - num_cor_pix)
    print('>>>', 'mean IU', np.nanmean(iu))
    miou = np.nanmean(iu)
    print('-' * 50)
    return miou


def get_iou_from_list(pred,lbl,n_cls):
    '''
    need tensor cpu
    :param pred: list
    :param lbl: list
    :param n_cls:
    :return:
    '''
    hist = np.zeros((n_cls,n_cls))
    for i,j in zip(range(len(pred)),range(len(lbl))):
        pred_item = pred[i].data.numpy()
        lbl_item = lbl[j].data.numpy()
        # print(pred_item.shape,lbl_item.shape)
        hist += fast_hist(lbl_item, pred_item, n_cls)

    # num of correct pixels
    num_cor_pix = np.diag(hist)
    # num of gt pixels
    num_gt_pix = hist.sum(1)
    union = num_gt_pix + hist.sum(0) - num_cor_pix
    # for i in range(20):
    acc = num_cor_pix.sum() / hist.sum()
    print('>>>', 'overall accuracy', acc)
    print('-' * 50)
    #     print('%-15s: %f' % (classes[i], num_cor_pix[i] / union[i]))
    iu = num_cor_pix / (num_gt_pix + hist.sum(0) - num_cor_pix)
    print('>>>', 'mean IU', np.nanmean(iu))
    miou = np.nanmean(iu)
    print('-' * 50)

    acc = num_cor_pix / num_gt_pix
    print('>>>', 'mean accuracy', np.nanmean(acc))
    print('-' * 50)

    return miou


def get_acc(pred,lbl,n_cls):
    '''
    need tensor cpu
    :param pred: list
    :param lbl: list
    :param n_cls:
    :return:
    '''
    hist = np.zeros((n_cls,n_cls))
    for i,j in zip(range(len(pred)),range(len(lbl))):
        pred_item = pred[i].data.numpy()
        lbl_item = lbl[j].data.numpy()
        # print(pred_item.shape,lbl_item.shape)
        hist += fast_hist(lbl_item, pred_item, n_cls)

    # num of correct pixels
    num_cor_pix = np.diag(hist)
    # num of gt pixels
    num_gt_pix = hist.sum(1)
    union = num_gt_pix + hist.sum(0) - num_cor_pix
    # for i in range(20):
    acc = num_cor_pix.sum() / hist.sum()
    print('>>>', 'classification overall accuracy', acc)

    return acc


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def get_iou_from_list_cat_iou(pred,lbl,n_cls, dataset='cihp'):
    '''
    need tensor cpu
    :param pred: list
    :param lbl: list
    :param n_cls:
    :return:
    '''
    hist = compute_hist(pred, lbl, n_cls=n_cls)

    if dataset == 'cihp':
        show_result(hist)
    elif dataset == 'atr':
        show_result_atr(hist)
    else:
        show_result_pascal(hist)

    print("\n\n\n\n\n")
    print(hist)
    print("\n\n\n\n\n")

    return


def cal_tb_(lbl, pred, n_cls):
    tp = np.zeros((n_cls,1))
    precision_m = np.zeros((n_cls,1))
    recall_m = np.zeros((n_cls,1))
    for i in range(n_cls):
        pred_cls = (pred == i)
        lbl_cls = lbl == i
        pred_sum = pred_cls.sum()
        lbl_sum = lbl_cls.sum()
        true_pred = ((2*pred_cls - lbl_cls) == 1).sum()
        tp[i] = true_pred
        precision_m[i] = pred_sum
        recall_m[i] = lbl_sum
    return tp,precision_m,recall_m


# def confusion_matrix(pred, lbl, n_cls):
#     '''
#     need tensor cpu
#     :param pred: list
#     :param lbl: list
#     :param n_cls:
#     :return:
#     '''
#
#     return


if __name__ == '__main__':
    import torch
    pred = torch.autograd.Variable(torch.ones((2,1,32,32)).int())*20
    pred2 = torch.autograd.Variable(torch.zeros((2,1, 32, 32)).int())
    # lbl = [torch.zeros((32,32)).int() for _ in range(len(pred))]
    get_iou(pred,pred2,7)
