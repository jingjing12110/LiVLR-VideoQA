# @File : obj_feat_extracting.py 
# @Time : 2020/8/21
import ast
import os
import sys
import base64
import csv
import h5py
import numpy as np
import skvideo.io
from tqdm import tqdm
import pandas as pd

csv.field_size_limit(sys.maxsize)

# FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes',
#               'features']
feature_length = 2048
num_fixed_boxes = 36

FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes',
              'features', 'concepts']


data_dir = '/media/kaka/HD2T/dataset/VideoQA/KnowIT/'
video_dir = '/media/kaka/HD2T/dataset/VideoQA/KnowIT/Frames/'
save_dir = '/media/kaka/HD2T/dataset/VideoQA/KnowIT/feats/'
infile = os.path.join(data_dir, 'codeData/Concepts/',
                      'knowit_resnet101_faster_rcnn_genome_vcps_all.tsv')


def read_tsv(frm_list):
    counter = 0
    num_boxes = 0
    img_bb = []
    pos_boxes = np.zeros((len(frm_list), 2), dtype=np.int32)
    img_features = []
    spatial_img_features = []

    infile = os.path.join(data_dir, 'codeData/Concepts/',
                          'knowit_resnet101_faster_rcnn_genome_vcps_all.tsv')
    with open(infile, "r") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t',
                                fieldnames=FIELDNAMES)
        for item in reader:
            image_id = int(item['image_id'])
            if image_id in frm_list:
                item['num_boxes'] = int(item['num_boxes'])
                image_w = float(item['image_w'])
                image_h = float(item['image_h'])
                bboxes = np.frombuffer(
                    base64.decodestring(item['boxes']),
                    dtype=np.float32).reshape((item['num_boxes'], -1))
                img_bb.append(bboxes)

                box_width = bboxes[:, 2] - bboxes[:, 0]
                box_height = bboxes[:, 3] - bboxes[:, 1]
                scaled_width = box_width / image_w
                scaled_height = box_height / image_h
                scaled_x = bboxes[:, 0] / image_w
                scaled_y = bboxes[:, 1] / image_h

                scaled_width = scaled_width[..., np.newaxis]
                scaled_height = scaled_height[..., np.newaxis]
                scaled_x = scaled_x[..., np.newaxis]
                scaled_y = scaled_y[..., np.newaxis]

                spatial_features = np.concatenate(
                    (scaled_x,
                     scaled_y,
                     scaled_x + scaled_width,
                     scaled_y + scaled_height,
                     scaled_width,
                     scaled_height),
                    axis=1)
                spatial_img_features.append(spatial_features)

                pos_boxes[counter, :] = np.array([num_boxes, num_boxes + item['num_boxes']])

                img_features.append(np.frombuffer(
                    base64.decodestring(item['features']),
                    dtype=np.float32).reshape((item['num_boxes'], -1)))

                frm_list.remove(image_id)
                counter += 1
                num_boxes += item['num_boxes']
    img_bb = np.concatenate(img_bb, axis=0)
    img_features = np.concatenate(img_features, axis=0)
    spatial_img_features = np.concatenate(spatial_img_features, axis=0)

    return img_bb, pos_boxes, img_features, spatial_img_features


def generate_feature(datas, frm_list):
    counter = 0
    num_boxes = 0
    indices = {}
    img_bb = []
    pos_boxes = np.zeros((len(frm_list), 2), dtype=np.int32)
    img_features = []
    spatial_img_features = []
    for image_id in frm_list:
        data = datas[image_id]
        image_w = data['image_w']
        image_h = data['image_h']
        bboxes = data['bboxes']
        img_bb.append(bboxes)
        img_features.append(data['features'])

        indices[image_id] = counter
        pos_boxes[counter, :] = np.array([num_boxes, num_boxes + data['num_boxes']])

        box_width = bboxes[:, 2] - bboxes[:, 0]
        box_height = bboxes[:, 3] - bboxes[:, 1]
        scaled_width = box_width / image_w
        scaled_height = box_height / image_h
        scaled_x = bboxes[:, 0] / image_w
        scaled_y = bboxes[:, 1] / image_h

        scaled_width = scaled_width[..., np.newaxis]
        scaled_height = scaled_height[..., np.newaxis]
        scaled_x = scaled_x[..., np.newaxis]
        scaled_y = scaled_y[..., np.newaxis]

        spatial_features = np.concatenate(
            (scaled_x,
             scaled_y,
             scaled_x + scaled_width,
             scaled_y + scaled_height,
             scaled_width,
             scaled_height),
            axis=1)
        spatial_img_features.append(spatial_features)

        counter += 1
        num_boxes += data['num_boxes']
    img_bb_ = np.concatenate(img_bb, axis=0)
    img_features_ = np.concatenate(img_features, axis=0)
    spatial_img_features_ = np.concatenate(spatial_img_features, axis=0)
    return img_bb_, pos_boxes, img_features_, spatial_img_features_


def msrvtt_extract():
    h5_data_file = os.path.join(save_dir, 'test_obj_feat_2048.h5')
    h5_data = h5py.File(h5_data_file, "w")
    g_img_bb = h5_data.create_group('img_bb')
    g_img_features = h5_data.create_group('img_features')
    g_spatial_features = h5_data.create_group('spatial_features')
    g_pos_boxes = h5_data.create_group('pos_boxes')

    for i in tqdm(range(7010, 10000)):
        video_pth = os.path.join(video_dir, 'video{}.mp4'.format(i))
        video = skvideo.io.vread(video_pth)
        frm_list = np.linspace(0, video.shape[0], 64,
                               endpoint=False, dtype=np.int32)
        infile = os.path.join(data_dir, 'video{}'.format(i),
                              'resnet101_faster_rcnn_video{}.tsv'.format(i))
        img_bb, pos_boxes, img_features, spatial_img_features = \
            read_tsv(infile, frm_list.tolist())

        g_img_bb.create_dataset('video{}'.format(i), data=img_bb,
                                dtype=np.float32)
        g_pos_boxes.create_dataset('video{}'.format(i), data=pos_boxes,
                                   dtype=np.int32)
        g_img_features.create_dataset('video{}'.format(i), data=img_features,
                                      dtype=np.float32)
        g_spatial_features.create_dataset('video{}'.format(i),
                                          data=spatial_img_features,
                                          dtype=np.float32)

    h5_data.close()
    print("finished.")


if __name__ == '__main__':
    mode = 'val'
    h5_data = h5py.File(os.path.join(
        save_dir, '{}_obj_feat_2048.h5'.format(mode)), "w")
    anns = pd.read_csv(os.path.join(
        data_dir, 'knowit_qa/knowit_data_{}.csv'.format(mode)), delimiter='\t')
    video_names = anns['scene'].unique().tolist()

    with open(os.path.join(data_dir, 'knowit_qa/list_frames.csv'), mode='r') as f:
        ids_frame = csv.reader(f, delimiter='\t')
        next(ids_frame)
        paths2ids = {rows[0]: int(rows[1]) for rows in ids_frame}

    g_img_bb = h5_data.create_group('img_bb')
    g_img_features = h5_data.create_group('img_features')
    g_spatial_features = h5_data.create_group('spatial_features')
    g_pos_boxes = h5_data.create_group('pos_boxes')
    tbar = tqdm(total=len(video_names))
    for video_name in video_names:
        vid, scene, start_v, end_v = video_name.split('_')
        total_ids = np.arange(int(start_v), int(end_v) + 1)
        total_frames = ['{}/frame_{}.jpeg'.format(vid, str(j).zfill(4))
                        for j in total_ids]
        frm_list = [paths2ids[i] for i in total_frames]

        # datas = {}
        # with open(infile, "r") as tsv_in_file:
        #     reader = csv.DictReader(tsv_in_file, delimiter='\t',
        #                             fieldnames=FIELDNAMES)
        #     for item in reader:
        #         if int(item['image_id']) in frm_list:
        #             featFrame = {}
        #             featFrame['image_id'] = int(item['image_id'])
        #             featFrame['image_w'] = float(item['image_w'])
        #             featFrame['image_h'] = float(item['image_h'])
        #             # featFrame['concepts'] = ast.literal_eval(item['concepts'])
        #             featFrame['num_boxes'] = int(item['num_boxes'])
        #             featFrame['bboxes'] = np.frombuffer(
        #                     base64.decodestring(item['boxes']),
        #                     dtype=np.float32).reshape((featFrame['num_boxes'], -1))
        #             featFrame['features'] = np.frombuffer(
        #                     base64.decodestring(item['features']),
        #                     dtype=np.float32).reshape((featFrame['num_boxes'], -1))
        #
        #             datas[featFrame['image_id']] = featFrame
        # bb, pos_box, img_feat, sp_img_feat = generate_feature(datas, frm_list)

        bb, pos_box, img_feat, sp_img_feat = read_tsv(frm_list)

        g_img_bb.create_dataset('{}'.format(video_name), data=bb,
                                dtype=np.float32)
        g_pos_boxes.create_dataset('{}'.format(video_name), data=pos_box,
                                   dtype=np.int32)
        g_img_features.create_dataset('{}'.format(video_name), data=img_feat,
                                      dtype=np.float32)
        g_spatial_features.create_dataset('{}'.format(video_name),
                                          data=sp_img_feat, dtype=np.float32)
        tbar.update(1)
    h5_data.close()
    tbar.close()

    print("finished.")



