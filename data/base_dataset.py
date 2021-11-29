# @File : base_dataset.py
# @Time : 2020/7/7 
# @Email : jingjingjiang2017@gmail.com
import os

import h5py
import numpy as np
import torch
from easydict import EasyDict
from torch.utils.data import Dataset, DataLoader
from prefetch_generator import BackgroundGenerator

from utils.tools import load_ujson, load_pickle


class DataPrefetcher(object):
    def __init__(self, loader):
        self.loader = iter(loader)
        # self.stream = torch.cuda.Stream()

        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return
        # with torch.cuda.stream(self.stream):
        #     for k in self.batch:
        #         self.batch[k] = self.batch[k].to(device='cuda:0',
        #                                          non_blocking=True)

    def next(self):
        # torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
    

class BaseDataset(Dataset):
    def __init__(self, opt, mode='train'):
        super(BaseDataset, self).__init__()
        self.opt = opt
        self.mode = mode
        self.feat_dir = os.path.join(self.opt.data_dir, 'feats/')

        print(f'loading {self.mode} annotation file.')
        if self.opt.dataset_name == 'TVQA':
            self.ann_dir = os.path.join(self.opt.data_dir, 'tvqa/')
            self.anns = load_ujson(os.path.join(self.ann_dir,
                                                f'tvqa_{mode}_processed.json'))
            self.concept = load_pickle(
                os.path.join(self.feat_dir, 'det_visual_concepts_hq.pickle'))

        elif self.opt.dataset_name == 'MSRVTT_QA':
            self.ann_dir = os.path.join(self.opt.data_dir, 'msrvtt_qa/')
            self.anns = load_ujson(os.path.join(self.ann_dir,
                                                f'{mode}_qa.json'))
            # self.ans_idx = load_ujson(os.path.join(self.ann_dir,
            #                                        'ans_id.json'))
            self.all_answer = load_ujson(os.path.join(
                self.ann_dir, 'Top4000_vocab.json'))['answer_token_to_idx']

        elif self.opt.dataset_name == 'KnowIT':
            self.ann_dir = os.path.join(self.opt.data_dir, 'knowit_qa/')
            self.anns = load_ujson(os.path.join(self.ann_dir, f''))

    def open_h5(self):
        # print(f'loading features from h5 file:')
        if self.opt.dataset_name == 'TVQA':
            self.img_feat = h5py.File(
                os.path.join(self.feat_dir, f'{self.mode}_img_feat_2048.h5'),
                'r')
        elif self.opt.dataset_name == 'MSRVTT_QA':
            self.feat_img = h5py.File(
                os.path.join(self.feat_dir, 'msrvtt_resnet101_2048.h5'), 'r')
        # semantic role
        self.cap_srg = h5py.File(
            os.path.join(self.feat_dir, f'{self.mode}_cap_srg.h5'), 'r')
        # object feature
        self.feat_obj = h5py.File(
            os.path.join(self.feat_dir, f'{self.mode}_obj_feat_2048_new.h5'), 'r')
        self.pembed_obj = h5py.File(
            os.path.join(self.feat_dir,
                         f'{self.mode}_obj_position_embed_64.h5'), 'r')
        # self.adj_matrix = h5py.File(
        #     os.path.join(self.feat_dir,
        #                  f'{self.mode}_bb_adj_matrix_11.h5'), 'r')
        # question embed
        # self.bert_qa = h5py.File(
        #     os.path.join(self.feat_dir, f'{self.mode}_qa_bert_768.h5'), 'r')
        self.bert_q = h5py.File(
            os.path.join(self.feat_dir, f'{self.mode}_que_bert_768.h5'), 'r')


class FeatureDataset(BaseDataset):
    def __init__(self, opt, mode='train'):
        super(FeatureDataset, self).__init__(opt, mode)

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, index):  # one QA pair
        items = EasyDict()
        if not hasattr(self, 'feat_img') or hasattr(self, 'cap_srg'):
            self.open_h5()

        if self.opt.dataset_name == 'MSRVTT_QA':
            items['qid'] = self.anns[index]['id']
            items['ques'] = self.anns[index]['question']
            items['ans'] = self.anns[index]['answer']
            items['vid_name'] = f"video{self.anns[index]['video_id']}"
            if items['ans'] in self.all_answer:
                items['target'] = self.all_answer[items['ans']]
            elif self.mode == 'train':
                items['target'] = 0
            elif self.mode in ['val', 'test']:
                items['target'] = 1

        else:
            items['qid'] = self.anns[index]['qid']
            items['vid_name'] = self.anns[index]['vid_name']
            # correct answer
            items['target'] = int(self.anns[index]['answer_idx'])
            # items['feat_vc'] = self.feat_vc[items['vid_name']]

        # semantic role graph
        for k in self.cap_srg.keys():
            # if k != 'num_sent':
            items[k] = self.cap_srg[k].get(items['vid_name'])[:]

        # [#img, 2048]
        feat_img = self.feat_img.get(items['vid_name'])[:]
        # TODO : change the method of sampling from the video clip
        frm_list = np.linspace(0, feat_img.shape[0], self.opt.sample,
                               endpoint=False, dtype=np.int32)
        items['img_feat'] = feat_img[frm_list, :]  # [n_img, 2048]

        # Bottom-Up Attention (object feature)
        # items['img_bb'] = self.feat_obj['img_bb'].get(items['vid_name'])[
        #                   :]  # [64, 10, 4]
        items['obj_feat'] = self.feat_obj['img_features'].get(items['vid_name'])[
                            :]  # [64, 10, 2048]
        items['sp_feat'] = self.feat_obj['spatial_features'].get(
            items['vid_name'])[:]  # [64, 10, 6]
        # items['pos_embed'] = self.pembed_obj.get(items['vid_name'])[:]
        # items['adj_matrix'] = self.adj_matrix.get(items['vid_name'])[:]

        # question-answer embedding
        items['q_bert'] = self.bert_q.get(f"q{items['qid']}")[:]  # [n_token, 768]

        return items

    @staticmethod
    def pad_collate(data):
        batch = EasyDict()
        item = list(data[0].keys())
        for k in item:
            if k == 'q_bert':
                # bert_q = [d['bert_q'] for d in data]
                batch['q_bert'], batch['q_bert_mask'] = pad_seq_1d(
                    [d['q_bert'] for d in data])
            else:
                if k in ['ques', 'ans', 'vid_name']:
                    batch[k] = [d[k] for d in data]
                else:
                    batch[k] = torch.from_numpy(np.array([d[k] for d in data]))
        return batch


def pad_seq_1d(seqs):
    extra_dims = seqs[0].shape[1:]  # tuple
    lengths = [len(seq) for seq in seqs]
    padded_seqs = torch.zeros((len(seqs), max(lengths)) + extra_dims,
                              dtype=torch.float32)
    mask = torch.zeros((len(seqs), max(lengths)), dtype=torch.int32)
    for idx, seq in enumerate(seqs):
        end = lengths[idx]
        padded_seqs[idx, :end] = torch.from_numpy(seq)
        mask[idx, :end] = 1
    return padded_seqs, mask


def pad_seq_2d(seqs):
    bs = len(seqs)
    max_para_len = max([len(seq) for seq in seqs])

    sent_lens = [[len(word_seq) for word_seq in seq] for seq in seqs]
    max_sent_len = max([item for sublist in sent_lens for item in sublist])

    seqs = [[torch.from_numpy(word_seq) for word_seq in seq] for seq in seqs]
    extra_dims = seqs[0][0].shape[1:]
    padded_seqs = torch.zeros((bs, max_para_len, max_sent_len) + extra_dims,
                              dtype=torch.float32)
    mask = torch.zeros((bs, max_para_len, max_sent_len), dtype=torch.int32)
    for b in range(bs):
        for idx, sent_l in enumerate(sent_lens[b]):
            padded_seqs[b, idx, :sent_l] = seqs[b][idx]
            mask[b, idx, :sent_l] = 1
    return padded_seqs, mask


class TestMRSVTTDataset(BaseDataset):
    def __init__(self, opt, cate=None, mode='test'):
        super(TestMRSVTTDataset, self).__init__(opt, mode)
        self.cate = cate
        if cate is not None:
            self.anns = load_ujson(os.path.join(self.ann_dir,
                                                f'{cate}_test_qa.json'))
        print(f'{cate}: {len(self.anns)}')

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, index):  # one QA pair
        items = EasyDict()
        if not hasattr(self, 'feat_img') or hasattr(self, 'cap_srg'):
            self.open_h5()

        if self.opt.dataset_name == 'MSRVTT_QA':
            items['qid'] = self.anns[index]['id']
            items['ques'] = self.anns[index]['question']
            items['ans'] = self.anns[index]['answer']
            items['vid_name'] = f"video{self.anns[index]['video_id']}"
            if items['ans'] in self.all_answer:
                items['target'] = self.all_answer[items['ans']]
            elif self.mode == 'train':
                items['target'] = 0
            elif self.mode in ['val', 'test']:
                items['target'] = 1

        else:
            items['qid'] = self.anns[index]['qid']
            items['vid_name'] = self.anns[index]['vid_name']
            # correct answer
            items['target'] = int(self.anns[index]['answer_idx'])
            # items['feat_vc'] = self.feat_vc[items['vid_name']]

        # semantic role graph
        for k in self.cap_srg.keys():
            # if k != 'num_sent':
            items[k] = self.cap_srg[k].get(items['vid_name'])[:]

        # [#img, 2048]
        feat_img = self.feat_img.get(items['vid_name'])[:]
        # TODO : change the method of sampling from the video clip
        frm_list = np.linspace(0, feat_img.shape[0], self.opt.sample,
                               endpoint=False, dtype=np.int32)
        items['img_feat'] = feat_img[frm_list, :]  # [n_img, 2048]

        # Bottom-Up Attention (object feature)
        # items['img_bb'] = self.feat_obj['img_bb'].get(items['vid_name'])[
        #                   :]  # [64, 10, 4]
        items['obj_feat'] = self.feat_obj['img_features'].get(items['vid_name'])[
                            :]  # [64, 10, 2048]
        items['sp_feat'] = self.feat_obj['spatial_features'].get(
            items['vid_name'])[:]  # [64, 10, 6]
        items['pos_embed'] = self.pembed_obj.get(items['vid_name'])[:]
        # items['adj_matrix'] = self.adj_matrix.get(items['vid_name'])[:]

        # question-answer embedding
        items['q_bert'] = self.bert_q.get(f"q{items['qid']}")[
                          :]  # [n_token, 768]

        return items

    @staticmethod
    def pad_collate(data):
        batch = EasyDict()
        item = list(data[0].keys())
        for k in item:
            if k == 'q_bert':
                batch['q_bert'], batch['q_bert_mask'] = pad_seq_1d(
                    [d['q_bert'] for d in data])
            else:
                if k in ['ques', 'ans', 'vid_name']:
                    batch[k] = [d[k] for d in data]
                else:
                    batch[k] = torch.from_numpy(np.array([d[k] for d in data]))
        return batch


if __name__ == '__main__':
    from tqdm import tqdm
    from option import args
    from torch.utils.data import DataLoader

    TVQADataset = FeatureDataset(args)
    # TVQADataset = TestMRSVTTDataset(args, cate='where')
    data_loader = DataLoader(TVQADataset,
                             batch_size=16,
                             shuffle=True,
                             collate_fn=TVQADataset.pad_collate,
                             num_workers=0,
                             pin_memory=True)
    i = 0
    print('=' * 60)
    qids = []
    tbar = tqdm(total=len(data_loader.dataset))
    for items in enumerate(data_loader):
        # print(len(items))
        # qids.append(items['qid'].item())
        tbar.update(1)
    print(f'len qids: {len(qids)}')
    # print(f'max: {max(qids)}')
    # print(f'unique len: {len(set(qids))}')
