# @File : msrvtt_dataset.py
# @Time : 2020/8/15 
# @Email : jingjingjiang2017@gmail.com
import os

import random
import h5py
import numpy as np
import torch
from easydict import EasyDict
from torch.utils.data import Dataset, DataLoader
from prefetch_generator import BackgroundGenerator

from data.feature_dataset import pad_seq_2d, pad_seq_1d
from utils.tools import load_ujson, save_json


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class MSRVTTDataset(Dataset):
    def __init__(self, opt, mode='train'):
        super(MSRVTTDataset, self).__init__()
        self.opt = opt
        self.mode = mode
        self.feat_dir = os.path.join(self.opt.data_dir, 'feats/')

        self.ann_dir = 'data/msrvtt-qa/'
        self.anns = load_ujson(os.path.join(
            self.ann_dir, f'{mode}_qa_1000.json'))
        random.shuffle(self.anns)
        self.all_answer = load_ujson(os.path.join(
            self.ann_dir, 'Top1000_vocab.json'))['answer_token_to_idx']

        self.open_h5()

    def process_ann(self):
        ann_list = []
        for index in range(len(self.anns)):
            ans = self.anns[index]['answer']
            if ans in self.all_answer:
                ann_list.append(self.anns[index])
        save_json(ann_list, os.path.join(
            self.ann_dir, f'{self.mode}_qa_1000.json'))

    def __len__(self):
        return len(self.anns)

    def open_h5(self):
        # question embed
        self.q_bert = h5py.File(
            os.path.join(self.feat_dir, f'{self.mode}_que_bert_768.h5'), 'r')

        # semantic role
        self.srg_bert = h5py.File(
            os.path.join(self.feat_dir, f'{self.mode}_srg_bert.h5'), 'r')

        # object feature
        obj_feat = h5py.File(os.path.join(
            self.feat_dir, f'{self.mode}_obj_feat_2048_new.h5'), 'r')
        self.obj_feat_o = obj_feat['img_features']
        self.obj_feat_s = obj_feat['spatial_features']

        self.adj_matrix = h5py.File(os.path.join(
            self.feat_dir, f'{self.mode}_obj_adj_matrix_11.h5'), 'r')

        # image feature
        self.img_feat = h5py.File(os.path.join(
            self.feat_dir, f'{self.mode}_img_feat_2048.h5'), 'r')

    def process_img_feat(self):
        self.img_feat = h5py.File(
            os.path.join(self.feat_dir, 'msrvtt_resnet101_2048.h5'), 'r')
        vid_names = set([f"video{a['video_id']}" for a in self.anns])
        with h5py.File(os.path.join(
                self.feat_dir, f'{self.mode}_img_feat_2048.h5'), 'w') as fd:
            for vid_name in tqdm(vid_names):
                feat_img = self.img_feat.get(vid_name)[:]
                frm_list = np.linspace(0, feat_img.shape[0],
                                       64, endpoint=False, dtype=np.int32)
                feat_img = feat_img[frm_list, :]
                fd.create_dataset(vid_name, data=feat_img, dtype=np.float32)

    def __getitem__(self, index):  # one Question
        items = EasyDict()
        # if not hasattr(self, 'img_feat') or hasattr(self, 'q_bert'):
        #     self.open_h5()

        # self.anns[index]['id'] = self.anns[index]['id']
        # items['ques'] = self.anns[index]['question']
        # items['ans'] = self.anns[index]['answer']
        ans = self.anns[index]['answer']
        # items['vid_name'] = f"video{self.anns[index]['video_id']}"
        items['target'] = self.all_answer[ans] - 2
        # if ans in self.all_answer:
        #     items['target'] = self.all_answer[ans]
        # elif self.mode == 'train':
        #     items['target'] = 0
        # elif self.mode in ['val', 'test']:
        #     items['target'] = 1

        # question-answer embedding
        qid = self.anns[index]['id']
        items['q_bert'] = self.q_bert.get(f"q{qid}")[:]  # [n_token, 768]

        vid_name = f"video{self.anns[index]['video_id']}"
        # semantic role graph
        for k in self.srg_bert.keys():
            if k == 'sent_bert':
                items[k] = [self.srg_bert[k].get(vid_name).get(f'{i}')[:]
                            for i in range(12)]
            else:
                items[k] = self.srg_bert[k].get(vid_name)[:]

        # [#img, 2048]
        items['img_feat'] = self.img_feat.get(vid_name)[:]  # [n_img, 2048]

        # Bottom-Up Attention (object feature)
        # items['attr_feat'] = self.obj_attr_feat.get(vid_name)[:]
        # items['obj_feat'] = self.obj_feat['img_features'].get(vid_name)[
        #                     :]  # [64, 10, 2048]
        # items['sp_feat'] = self.obj_feat['spatial_features'].get(
        #     vid_name)[:]  # [64, 10, 6]
        items['obj_feat'] = self.obj_feat_o.get(vid_name)[:]  # [64, 10, 2048]
        items['sp_feat'] = self.obj_feat_s.get(vid_name)[:]  # [64, 10, 6]
        items['obj_adj_matrix'] = self.adj_matrix.get(vid_name)[:]

        return items

    @staticmethod
    def pad_collate(data):
        batch = EasyDict()
        item = list(data[0].keys())
        for k in item:
            if k == 'q_bert':
                batch['q_bert'], batch['q_mask'] = pad_seq_1d(
                    [d[k] for d in data])
            elif k == 'sent_bert':
                batch['sent_bert'], batch['sent_mask'] = pad_seq_2d(
                    [d[k] for d in data])
            else:
                batch[k] = torch.from_numpy(np.array([d[k] for d in data]))
                # if k in ['ques', 'ans', 'vid_name']:
                #     batch[k] = [d[k] for d in data]
                # else:
                #     batch[k] = torch.from_numpy(np.array([d[k] for d in data]))
        return batch


class TestMRSVTTDataset(MSRVTTDataset):
    def __init__(self, opt, cate=None, mode='test'):
        super(TestMRSVTTDataset, self).__init__(opt, mode)
        self.cate = cate
        if cate is not None:
            self.anns = load_ujson(os.path.join(self.ann_dir,
                                                f'{cate}_test_qa_1000.json'))
        print(f'{cate}: {len(self.anns)}')
        # self.obj_seq_id = (np.arange(64) / 64)[:, np.newaxis, np.newaxis]
        # self.obj_seq_id = np.tile(
        #     self.obj_seq_id, (1, 10, 1)).astype(np.float32)

    def __len__(self):
        return len(self.anns)

    @staticmethod
    def pad_collate(data):
        batch = EasyDict()
        item = list(data[0].keys())
        for k in item:
            if k == 'q_bert':
                batch['q_bert'], batch['q_mask'] = pad_seq_1d(
                    [d[k] for d in data])
            elif k == 'sent_bert':
                batch['sent_bert'], batch['sent_mask'] = pad_seq_2d(
                    [d[k] for d in data])
            else:
                if k in ['ques', 'ans', 'vid_name', 'id']:
                    batch[k] = [d[k] for d in data]
                else:
                    batch[k] = torch.from_numpy(np.array([d[k] for d in data]))
        return batch

    def __getitem__(self, index):  # one QA pair
        items = EasyDict()
        if not hasattr(self, 'img_feat') or hasattr(self, 'q_bert'):
            self.open_h5()

        # items['id'] = self.anns[index]['id']
        # items['ques'] = self.anns[index]['question']
        # items['ans'] = self.anns[index]['answer']
        ans = self.anns[index]['answer']
        # items['vid_name'] = f"video{self.anns[index]['video_id']}"
        items['target'] = self.all_answer[ans] - 2
        # if ans in self.all_answer:
        #     items['target'] = self.all_answer[ans]
        # elif self.mode == 'train':
        #     items['target'] = 0
        # elif self.mode in ['val', 'test']:
        #     items['target'] = 1

        # question-answer embedding
        qid = self.anns[index]['id']
        items['q_bert'] = self.q_bert.get(f"q{qid}")[:]  # [n_token, 768]

        vid_name = f"video{self.anns[index]['video_id']}"
        # semantic role graph
        for k in self.srg_bert.keys():
            if k == 'sent_bert':
                items[k] = [self.srg_bert[k].get(vid_name).get(f'{i}')[:]
                            for i in range(12)]
            else:
                items[k] = self.srg_bert[k].get(vid_name)[:]

        # [#img, 2048]
        items['img_feat'] = self.img_feat.get(vid_name)[:]  # [n_img, 2048]

        # Bottom-Up Attention (object feature)
        # items['attr_feat'] = self.obj_attr_feat.get(vid_name)[:]
        # items['obj_feat'] = self.obj_feat['img_features'].get(vid_name)[
        #                     :]  # [64, 10, 2048]
        # items['sp_feat'] = self.obj_feat['spatial_features'].get(
        #     vid_name)[:]  # [64, 10, 6]
        items['obj_feat'] = self.obj_feat_o.get(vid_name)[:]  # [64, 10, 2048]
        items['sp_feat'] = self.obj_feat_s.get(vid_name)[:]  # [64, 10, 6]
        # items['sp_feat'] = np.concatenate(
        #     (items['sp_feat'], self.obj_seq_id), -1)
        items['obj_adj_matrix'] = self.adj_matrix.get(vid_name)[:]

        return items


if __name__ == '__main__':
    from tqdm import tqdm
    from option import args
    from torch.utils.data import DataLoader

    TVQADataset = MSRVTTDataset(args, mode='test')
    # TVQADataset.process_img_feat()
    TVQADataset.process_ann()

    # data_loader = DataLoader(TVQADataset,
    #                          batch_size=16,
    #                          shuffle=True,
    #                          collate_fn=TVQADataset.pad_collate,
    #                          num_workers=0,
    #                          pin_memory=True)
    # print('=' * 60)
    # tbar = tqdm(total=len(data_loader.dataset))
    # for items in enumerate(data_loader):
    #     tbar.update(1)
