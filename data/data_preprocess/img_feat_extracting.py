# @File : img_feat_extracting.py

import argparse
import json
import os
import random
import pandas as pd

import h5py
import numpy as np
import skvideo.io
import torch
import torchvision
from PIL import Image
from torch import nn
from tqdm import tqdm
import cv2


def build_resnet():
    if not hasattr(torchvision.models, args.model):
        raise ValueError('Invalid model "%s"' % args.model)
    if not 'resnet' in args.model:
        raise ValueError('Feature extraction only supports ResNets')
    cnn = getattr(torchvision.models, args.model)(pretrained=True)
    model = torch.nn.Sequential(*list(cnn.children())[:-1])
    model.cuda()
    model.eval()
    return model


def run_batch(cur_batch, model):
    """
    Args:
        cur_batch: treat a video as a batch of images
        model: ResNet model for feature extraction
    Returns:
        ResNet extracted feature.
    """
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.224]).reshape(1, 3, 1, 1)

    image_batch = np.concatenate(cur_batch, 0).astype(np.float32)
    image_batch = (image_batch / 255.0 - mean) / std
    image_batch = torch.from_numpy(image_batch.astype(np.float32)).cuda()
    with torch.no_grad():
        feats = model(image_batch)

    return feats.data.cpu().clone().numpy()


def load_video_paths(args):
    """ Load a list of (path, image_id tuples) """
    video_paths = []
    modes = ['train', 'val', 'test']
    for mode in modes:
        with open(args.annotation_file.format(mode), 'r') as anno_file:
            instances = json.load(anno_file)
        video_ids = [instance['video_id'] for instance in instances]
        video_ids = set(video_ids)
        if mode in ['train', 'val']:
            for video_id in video_ids:
                video_paths.append((
                    args.video_dir + 'TrainValVideo/video{}.mp4'.format(
                        video_id), video_id))
        else:
            for video_id in video_ids:
                video_paths.append((
                    args.video_dir + 'TestVideo/video{}.mp4'.format(
                        video_id), video_id))

    return video_paths


def load_knowit_video_paths(args):
    """ Load a list of (path, image_id tuples) """
    video_paths = []
    modes = ['train', 'val', 'test']
    for mode in modes:
        instances = pd.read_csv(args.annotation_file.format(mode), delimiter='\t')
        video_ids = instances['scene'].unique().tolist()

        for video_id in video_ids:
            video_paths.append((
                args.video_dir + f"{video_id.split('_')[0]}", video_id))
    return video_paths


def extract_frame(path):
    video_data = skvideo.io.vread(path)
    total_frames = video_data.shape[0]
    img_size = (args.image_height, args.image_width)
    new_clip = []
    for j in range(total_frames):
        frame_data = video_data[j]
        img = Image.fromarray(frame_data)
        # img = imresize(img, img_size, interp='bicubic')
        img = np.array(img.resize(img_size))
        img = img.transpose(2, 0, 1)[None]
        frame_data = np.array(img)
        new_clip.append(frame_data)
    new_clip = np.asarray(new_clip)

    return new_clip


def extract_knowit_frame(path, video_id):
    vid, scene, start_v, end_v = video_id.split('_')
    total_frames = np.arange(int(start_v), int(end_v) + 1)
    img_size = (args.image_height, args.image_width)
    new_clip = []
    for j in total_frames:
        img_path = os.path.join(path, f'frame_{str(j).zfill(4)}.jpeg')
        frame_data = cv2.imread(img_path)
        img = Image.fromarray(frame_data)
        img = np.array(img.resize(img_size))
        img = img.transpose(2, 0, 1)[None]
        frame_data = np.array(img)
        new_clip.append(frame_data)
    new_clip = np.asarray(new_clip)

    return new_clip


def generate_h5(model, video_ids, outfile):
    """
    Args:
        model: loaded pretrained model for feature extraction
        video_ids: list of video ids
        outfile: path of output file to be written
    Returns:
        h5 file containing visual features of splitted clips.
    """
    clip_lens = []
    with h5py.File(outfile, 'w') as fd:
        tbar = tqdm(total=len(video_ids))
        for i, (video_path, video_id) in enumerate(video_ids):
            # clips = extract_frame(video_path)
            clips = extract_knowit_frame(video_path, video_id)

            clip_feat = []
            step = list(range(0, clips.shape[0], 64))
            step.append(clips.shape[0])
            for f_i in range(len(step) - 1):
                clip = clips[step[f_i]:step[f_i + 1], :]
                feats = run_batch(clip, model)  # (16, 2048)

                clip_feat.append(feats)

            clip_feat = np.concatenate(clip_feat, axis=0).squeeze()
            clip_lens.append(clip_feat.shape[0])
            # (num_frame, 2048)
            fd.create_dataset(f'{video_id}', data=clip_feat, dtype=np.float32)

            tbar.update(1)
        tbar.close()
    print(f'max: {max(clip_lens)}\n min: {min(clip_lens)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='specify which gpu will be used')
    # dataset info
    parser.add_argument('--dataset', default='knowit',
                        choices=['TVQA', 'msrvtt-qa', 'knowit'], type=str)
    parser.add_argument('--save_dir', type=str,
                        default='/media/kaka/HD2T/dataset/VideoQA/KnowIT/feats')
    # output
    parser.add_argument('--out', dest='outfile',
                        help='output filepath',
                        default="{}_{}_feat.h5",
                        type=str)
    # image sizes
    parser.add_argument('--image_height', default=224, type=int)
    parser.add_argument('--image_width', default=224, type=int)

    # network params
    parser.add_argument('--model', default='resnet101',
                        choices=['resnet101', 'resnext101'], type=str)
    parser.add_argument('--seed', default='666', type=int, help='random seed')
    args = parser.parse_args()
    if args.model == 'resnet101':
        args.feature_type = 'appearance'
        # args.feature_type = 'resnet101'
    elif args.model == 'resnext101':
        args.feature_type = 'motion'
    else:
        raise Exception('Feature type not supported!')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # annotation files
    if args.dataset == 'msrvtt-qa':
        args.annotation_file = \
            '/media/kaka/HD2T/dataset/VideoQA/MSRVTT/2016/msrvtt_qa/{}_qa.json'
        args.video_dir = '/media/kaka/HD2T/dataset/VideoQA/MSRVTT/2016/raw_data/'
        video_paths = load_video_paths(args)
        random.shuffle(video_paths)

        # load model
        # if args.model == 'resnet101':
        model = build_resnet()
        generate_h5(model, video_paths,
                    args.outfile.format(args.dataset, args.model,
                                        args.feature_type))
    elif args.dataset == 'knowit':
        args.annotation_file = \
            '/media/kaka/HD2T/dataset/VideoQA/KnowIT/knowit_qa/knowit_data_{}.csv'
        args.video_dir = '/media/kaka/HD2T/dataset/VideoQA/KnowIT/Frames/'
        video_paths = load_knowit_video_paths(args)
        random.shuffle(video_paths)

        # load model
        # if args.model == 'resnet101':
        model = build_resnet()
        outfile = os.path.join(args.save_dir,
                               f'{args.dataset}_{args.model}_2048.h5')
        generate_h5(model, video_paths, outfile)

