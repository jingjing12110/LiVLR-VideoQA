# @File : bert_embedding.py
# @Time : 2020/7/8 
# @Email : jingjingjiang2017@gmail.com 

import argparse
import os

import h5py
import numpy as np
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel
from tqdm import tqdm

from utils.tools import load_ujson


def extract_qa_bert_embedding(ann, save_dir):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                              do_lower_case=True)
    model = BertModel.from_pretrained('bert-base-uncased')
    model.cuda()
    model.eval()

    question = [(ann[i]['id'], ann[i]['question']) for i in range(len(ann))]
    answer = [(ann[i]['id'], ann[i]['answer']) for i in range(len(ann))]
    outfile = os.path.join(save_dir, 'test_qa_bert_768.h5')
    pbar = tqdm(total=len(question))
    with h5py.File(outfile, 'w') as fd:
        for i in range(len(question)):
            qid, que = question[i]
            _, ans = answer[i]
            marked_q = '[CLS] ' + que + ' [SEP]'
            marked_qa = '[CLS] ' + que + ' [SEP] ' + ans + ' [SEP]'
            tokenized_q = tokenizer.tokenize(marked_q)
            tokenized_qa = tokenizer.tokenize(marked_qa)
            q_token = tokenizer.convert_tokens_to_ids(tokenized_q)
            indexed_token = tokenizer.convert_tokens_to_ids(tokenized_qa)
            segment_id = [0] * len(q_token) + [1] * (
                    len(indexed_token) - len(q_token))
            token_tensor = torch.tensor([indexed_token])
            segment_tensor = torch.tensor([segment_id])

            with torch.no_grad():
                encoded_layers, _ = model(token_tensor, segment_tensor)

            embeds = encoded_layers[11].squeeze(dim=0)
            embeds = torch.cat([embeds[1:len(q_token) - 1],
                                embeds[len(q_token):-1]], dim=0).numpy()

            fd.create_dataset(f'q{qid}', data=embeds, dtype=np.float32)
            pbar.update(1)


class PretrainBertEmbedding:
    def __init__(self, cfg):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                                       do_lower_case=True)
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.model.cuda()
        self.model.eval()

        self.cfg = cfg

    def h5_question_embeds(self):
        question = [(self.cfg.ann[i]['id'], self.cfg.ann[i]['question']) for i in
                    range(len(self.cfg.ann))]
        out_file = os.path.join(self.cfg.save_dir,
                                f'{self.cfg.mode}_que_bert_768_debug.h5')
        pbar = tqdm(total=len(question))
        with h5py.File(out_file, 'w') as fd:
            for i in range(len(question)):
                qid, que = question[i]
                embed = self.sentence_embedding(que)
                fd.create_dataset(f'q{qid}', data=embed, dtype=np.float32)
                pbar.update(1)
        pbar.close()

    def sentence_embedding(self, sent, pooling='sentence'):
        marked_q = '[CLS] ' + sent + ' [SEP]'
        tokenized_q = self.tokenizer.tokenize(marked_q)

        q_token = self.tokenizer.convert_tokens_to_ids(tokenized_q)
        segment_id = [0] * len(q_token)

        token_tensor = torch.tensor([q_token]).cuda()
        segment_tensor = torch.tensor([segment_id]).cuda()

        with torch.no_grad():
            encoded_layers, _ = self.model(token_tensor, segment_tensor)
        if pooling == 'token':
            # last layer
            # embed = encoded_layers[11].squeeze(dim=0)
            # embed = embed[1:len(q_token) - 1].cpu().numpy()
            # last four layer
            embed = torch.cat(encoded_layers[-4:], dim=0).sum(dim=0)
            embed = embed[1:len(q_token) - 1].cpu().numpy()  # [#token, 768]
            return embed, tokenized_q
        elif pooling == 'sentence':
            embed = encoded_layers[-2][0].mean(dim=0)
            embed = embed.cpu().numpy()  # [768]
            return embed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='textual BERT embedding')
    parser.add_argument('--data_name', type=str, default='MSRVTT_QA',
                        help='TVQA, MSRVTT_QA or KnowIT')
    parser.add_argument('--data_dir', type=str,
                        default='/media/kaka/HD2T/dataset/VideoQA/MSRVTT/2016/')
    parser.add_argument('--mode', type=str, default='train',
                        help='test, val or train')

    args = parser.parse_args()

    if args.data_name == 'MSRVTT_QA':
        ann_dir = os.path.join(args.data_dir, 'msrvtt_qa')
        save_dir = os.path.join(args.data_dir, 'feats')

    ann = load_ujson(os.path.join(ann_dir, f'{args.mode}_qa.json'))
    args.ann = ann
    args.save_dir = save_dir

    ############################################################################
    # generate bert embedding for question sentence
    ############################################################################
    bert_generate = PretrainBertEmbedding(args)
    bert_generate.h5_question_embeds()

    ############################################################################
    # generate bert vector of caption sentence
    ############################################################################
    # cap_sentences = [j for i in range(10000) for j in ann_cap[f'video{i}']]
    # cap_sentences = list(set(cap_sentences))
    #
    # outfile = os.path.join(save_dir, 'bert_caption_768.h5')
    # with h5py.File(outfile, 'w') as fd:
    #     for i in tqdm(range(len(cap_sentences))):
    #         cap = cap_sentences[i]
    #         bert_sent, captions = extract_bert_embedding(cap)
    #         fd.create_dataset(cap, data=bert_sent, dtype=np.float32)
