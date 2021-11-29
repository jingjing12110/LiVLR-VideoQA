# @File : object_relation_embedding.py.py
# @Time : 2020/8/9 
# @Email : jingjingjiang2017@gmail.com
import os
import random

import math
import h5py
import numpy as np
from easydict import EasyDict
from torch.utils.data import Dataset

from utils.tools import load_ujson, save_json


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def build_graph(bbox, spatial):
    """ Build spatial graph
    Args:
        :param bbox: [num_boxes, 4]
        :param spatial:
    Returns:
        adj_matrix: [num_boxes, num_boxes]
    """
    num_box = bbox.shape[0]
    adj_matrix = np.zeros((num_box, num_box))
    xmin, ymin, xmax, ymax = np.split(bbox, 4, axis=1)
    # [num_boxes, 1]
    bbox_width = xmax - xmin + 1.
    bbox_height = ymax - ymin + 1.
    image_h = bbox_height[0] / spatial[0, -1]
    image_w = bbox_width[0] / spatial[0, -2]
    center_x = 0.5 * (xmin + xmax)
    center_y = 0.5 * (ymin + ymax)
    image_diag = math.sqrt(image_h ** 2 + image_w ** 2)
    for i in range(num_box):
        bbA = bbox[i]
        if sum(bbA) == 0:
            continue
        adj_matrix[i, i] = 12
        for j in range(i + 1, num_box):
            bbB = bbox[j]
            if sum(bbB) == 0:
                continue
            # class 1: inside (j inside i)
            if xmin[i] < xmin[j] and xmax[i] > xmax[j] and \
                    ymin[i] < ymin[j] and ymax[i] > ymax[j]:
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 2
            # class 2: cover (j covers i)
            elif (xmin[j] < xmin[i] and xmax[j] > xmax[i] and
                  ymin[j] < ymin[i] and ymax[j] > ymax[i]):
                adj_matrix[i, j] = 2
                adj_matrix[j, i] = 1
            else:
                ioU = bb_intersection_over_union(bbA, bbB)
                # class 3: i and j overlap
                if ioU >= 0.5:
                    adj_matrix[i, j] = 3
                    adj_matrix[j, i] = 3
                else:
                    y_diff = center_y[i] - center_y[j]
                    x_diff = center_x[i] - center_x[j]
                    diag = math.sqrt(y_diff ** 2 + x_diff ** 2)
                    if diag < 0.5 * image_diag:
                        sin_ij = y_diff / diag
                        cos_ij = x_diff / diag
                        if sin_ij >= 0 and cos_ij >= 0:
                            label_i = np.arcsin(sin_ij)
                            label_j = 2 * math.pi - label_i
                        elif sin_ij < 0 and cos_ij >= 0:
                            label_i = np.arcsin(sin_ij) + 2 * math.pi
                            label_j = label_i - math.pi
                        elif sin_ij >= 0 and cos_ij < 0:
                            label_i = np.arccos(cos_ij)
                            label_j = 2 * math.pi - label_i
                        else:
                            label_i = -np.arccos(sin_ij) + 2 * math.pi
                            label_j = label_i - math.pi
                        adj_matrix[i, j] = int(
                            np.ceil(label_i / (math.pi / 4))) + 3
                        adj_matrix[j, i] = int(
                            np.ceil(label_j / (math.pi / 4))) + 3
    return adj_matrix


class ObjectFeatureProcess:
    def __init__(self, opt, mode='train'):
        super(ObjectFeatureProcess, self).__init__()
        self.opt = opt
        self.mode = mode

        self.feat_dir = os.path.join(self.opt.data_dir, 'feats/')

        print(f'loading {self.mode} annotation file.')
        if self.opt.dataset_name == 'TVQA':
            self.ann_dir = os.path.join(self.opt.data_dir, 'tvqa/')
            self.anns = load_ujson(os.path.join(self.ann_dir,
                                                f'tvqa_{mode}_processed.json'))

        elif self.opt.dataset_name == 'MSRVTT_QA':
            self.ann_dir = os.path.join(self.opt.data_dir, 'msrvtt_qa/')
            self.anns = load_ujson(os.path.join(self.ann_dir, f'{mode}_qa.json'))

        self.video_names = set([f"video{q['video_id']}" for q in self.anns])
        self.video_names = sorted(list(self.video_names))

    def process_msrvtt_test_qa(self):
        for cate in ['where', 'what', 'who', 'when', 'how']:
            qa_list = [a for a in self.anns if
                       a['question'].strip().split(' ')[0] == cate]
            save_json(qa_list, os.path.join(self.ann_dir, f'{cate}_test_qa.json'))
            print(f'{cate}: {len(qa_list)}')

    def preprocess_feat_obj(self):
        self.feat_obj = h5py.File(
            os.path.join(self.feat_dir, f'{self.mode}_obj_feat_2048.h5'), 'r')

        h5_data_file = os.path.join(
            self.feat_dir, f'{self.mode}_obj_feat_2048_new.h5')
        h5_data = h5py.File(h5_data_file, "w")
        g_img_bb = h5_data.create_group('img_bb')
        g_img_features = h5_data.create_group('img_features')
        g_spatial_features = h5_data.create_group('spatial_features')
        # g_pos_boxes = h5_data.create_group('pos_boxes')

        for name in tqdm(self.video_names):
            # [self.opt.sample*n_obj, 2048]
            feat_obj = self.feat_obj['img_features'].get(name)[:]
            # [self.opt.sample*n_obj, 6]
            feat_spatial = self.feat_obj['spatial_features'].get(name)[:]
            # [self.opt.sample*n_obj, 4]
            img_bb = self.feat_obj['img_bb'].get(name)[:]
            # [self.opt.sample, 2]
            pos_box = self.feat_obj['pos_boxes'].get(name)[:]
            feat_obj_list = []
            feat_spatial_list = []
            img_bb_list = []
            step = [n_b[1] - n_b[0] for n_b in pos_box]
            num_img = len([s for s in step if s > 0])
            if num_img < len(step):
                img_id = np.linspace(0, num_img, 64, endpoint=False,
                                     dtype=np.int32)
                for i in range(64):
                    feat_obj_list.append(
                        feat_obj[pos_box[img_id[i]][0]:pos_box[img_id[i]][1]][:10])
                    feat_spatial_list.append(
                        feat_spatial[pos_box[img_id[i]][0]:pos_box[img_id[i]][1]][:10])
                    img_bb_list.append(
                        img_bb[pos_box[img_id[i]][0]:pos_box[img_id[i]][1]][:10])
            else:
                for n_b in pos_box:
                    feat_obj_list.append(feat_obj[n_b[0]:n_b[1], :][:10, :])
                    feat_spatial_list.append(
                        feat_spatial[n_b[0]:n_b[1], :][:10, :])
                    img_bb_list.append(img_bb[n_b[0]:n_b[1], :][:10, :])

            feat_obj_list = np.stack(feat_obj_list, axis=0)
            feat_spatial_list = np.stack(feat_spatial_list, axis=0)
            img_bb_list = np.stack(img_bb_list, axis=0)
            if feat_obj_list.shape[0] != 64:
                print(f'shape: {feat_obj_list.shape}')

            g_img_features.create_dataset(name, data=feat_obj_list,
                                          dtype=np.float32)
            g_spatial_features.create_dataset(name,
                                              data=feat_spatial_list,
                                              dtype=np.float32)
            g_img_bb.create_dataset(name, data=img_bb_list,
                                    dtype=np.float32)
        h5_data.close()


if __name__ == '__main__':
    from tqdm import tqdm
    from option import args

    # preprocess feat_obj
    for mode in ['train', 'val', 'test']:
        dataset = ObjectFeatureProcess(args, mode=mode)
        dataset.preprocess_feat_obj()

    # split qa_test.json
    dataset = ObjectFeatureProcess(args, mode='test')
    dataset.process_msrvtt_test_qa()

