# @File : semantic_role_embedding.py
# @Time : 2020/8/14 
# @Email : jingjingjiang2017@gmail.com

import os
import random

import h5py
import numpy as np
import pandas as pd
from easydict import EasyDict
from tqdm import tqdm

from data.data_preprocess.bert_embedding import PretrainBertEmbedding
from utils.tools import load_ujson

BOS, EOS, UNK = 0, 1, 2
ROLES = ['V', 'ARG1', 'ARG0', 'ARG2', 'ARG3', 'ARG4',
         'ARGM-LOC', 'ARGM-MNR', 'ARGM-TMP', 'ARGM-DIR', 'ARGM-ADV',
         'ARGM-PRP', 'ARGM-PRD', 'ARGM-COM', 'ARGM-MOD', 'NOUN']


def sort_text(texts: list, srl_len):
    text_len = [len(t.split()) for t in texts]
    text_dict = {'text': texts, 'len': text_len}
    df = pd.DataFrame.from_dict(text_dict)
    df = df.sort_values(by='len', ascending=False)[:srl_len]
    return df['text'].to_list()


class SRGRepresentation:
    def __init__(self, opt, mode='train'):
        super(SRGRepresentation, self).__init__()
        self.opt = opt
        self.mode = mode

        self.num_verbs = 4  # num_verbs
        self.num_nouns = 6  # num_nouns
        self.srl_len = 12  # num of caption for each video
        self.max_words_in_sent = 30  #

        self.feat_dir = os.path.join(self.opt.data_dir, 'feats/')

        print(f'loading {self.mode} annotation file.')
        if self.opt.dataset_name == 'TVQA':
            self.ann_dir = os.path.join(self.opt.data_dir, 'tvqa/')

        elif self.opt.dataset_name == 'MSRVTT_QA':
            self.ann_dir = os.path.join(self.opt.data_dir, 'msrvtt_qa/')
            self.anns = load_ujson(os.path.join(self.ann_dir, f'{mode}_qa.json'))
            self.video_captions = load_ujson(
                os.path.join(self.ann_dir, 'video_caption_pair.json'))

        self.video_names = set([f"video{q['video_id']}" for q in self.anns])
        self.video_names = sorted(list(self.video_names))

        self.role2int = {}
        for i, role in enumerate(ROLES):
            self.role2int[role] = i
            self.role2int['C-%s' % role] = i
            self.role2int['R-%s' % role] = i

        self.sent_graphs = load_ujson(
            os.path.join(self.ann_dir, 'cap_sent2rolegraph.augment.json'))
        self.word2int = load_ujson(os.path.join(self.ann_dir, 'word2int.json'))

    def get_captions_srl_graph(self, caps):
        caps_graph = []
        for cap in caps:
            graph = self.sent_graphs[cap]
            caps_graph.append(self.get_srl_graph(cap, graph))
        return caps_graph

    def get_srl_graph(self, sent, graph):
        graph_nodes, graph_edges = graph
        cap_graph = EasyDict()

        verb_node2idxs, noun_node2idxs = {}, {}
        edges = []
        cap_graph['node_roles'] = np.zeros((self.num_verbs + self.num_nouns,),
                                           np.int32)
        # root node
        sent_ids, sent_len = self.process_sent(sent, self.max_words_in_sent)
        cap_graph['sent_ids'] = sent_ids  # word2int: token id
        cap_graph['sent_lens'] = sent_len
        # graph: add verb nodes
        node_idx = 1
        cap_graph['verb_masks'] = np.zeros((self.num_verbs,
                                            self.max_words_in_sent), np.bool)
        for knode, vnode in graph_nodes.items():
            k = node_idx - 1
            if k >= self.num_verbs:
                break
            if vnode['role'] == 'V' and np.min(
                    vnode['spans']) < self.max_words_in_sent:
                verb_node2idxs[knode] = node_idx
                for widx in vnode['spans']:
                    if widx < self.max_words_in_sent:
                        cap_graph['verb_masks'][k][widx] = True
                cap_graph['node_roles'][node_idx - 1] = self.role2int['V']
                # add root to verb edge
                edges.append((0, node_idx))
                node_idx += 1
        # graph: add noun nodes
        node_idx = 1 + self.num_verbs
        cap_graph['noun_masks'] = np.zeros((self.num_nouns,
                                            self.max_words_in_sent), np.bool)
        for knode, vnode in graph_nodes.items():
            k = node_idx - self.num_verbs - 1
            if k >= self.num_nouns:
                break
            if vnode['role'] not in ['ROOT', 'V'] and np.min(
                    vnode['spans']) < self.max_words_in_sent:
                noun_node2idxs[knode] = node_idx
                for widx in vnode['spans']:
                    if widx < self.max_words_in_sent:
                        cap_graph['noun_masks'][k][widx] = True
                cap_graph['node_roles'][node_idx - 1] = \
                    self.role2int.get(vnode['role'], self.role2int['NOUN'])
                node_idx += 1
        # graph: add verb_node to noun_node edges
        for e in graph_edges:
            if e[0] in verb_node2idxs and e[1] in noun_node2idxs:
                edges.append((verb_node2idxs[e[0]], noun_node2idxs[e[1]]))
                edges.append((noun_node2idxs[e[1]], verb_node2idxs[e[0]]))

        num_nodes = 1 + self.num_verbs + self.num_nouns
        rel_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        for src_nodeidx, tgt_nodeidx in edges:
            rel_matrix[tgt_nodeidx, src_nodeidx] = 1
        # row norm
        for i in range(num_nodes):
            s = np.sum(rel_matrix[i])
            if s > 0:
                rel_matrix[i] /= s

        cap_graph['rel_edges'] = rel_matrix
        return cap_graph

    def process_sent(self, sent, max_words):
        tokens = [self.word2int.get(w, UNK) for w in sent.split()]
        # # add BOS, EOS?
        # tokens = [BOS] + tokens + [EOS]
        tokens = tokens[:max_words]
        tokens_len = len(tokens)
        tokens = np.array(tokens + [EOS] * (max_words - tokens_len))
        return tokens, tokens_len

    def generate_video_srg_pair(self):
        outfile = os.path.join(self.feat_dir, f'{self.mode}_cap_srg_debug.h5')
        with h5py.File(outfile, 'w') as fd:
            g_node_role = fd.create_group('node_roles')
            g_rel_edge = fd.create_group('rel_edges')
            g_sent_id = fd.create_group('sent_ids')
            g_sent_len = fd.create_group('sent_lens')
            g_verb_mask = fd.create_group('verb_masks')
            g_noun_mask = fd.create_group('noun_masks')

            for name in tqdm(self.video_names):
                g_node_role.create_dataset(name, (self.srl_len, 10),
                                           dtype=np.int32)
                g_rel_edge.create_dataset(name, (self.srl_len, 11, 11),
                                          dtype=np.float32)
                g_sent_id.create_dataset(name, (self.srl_len, 30),
                                         dtype=np.int32)
                g_sent_len.create_dataset(name, (self.srl_len,),
                                          dtype=np.int32)
                g_verb_mask.create_dataset(name, (self.srl_len, 4, 30),
                                           dtype=np.bool)
                g_noun_mask.create_dataset(name, (self.srl_len, 6, 30),
                                           dtype=np.bool)

                ref_cap_temp = self.video_captions[name]
                if len(ref_cap_temp) >= self.srl_len:
                    ref_cap = random.sample(ref_cap_temp, self.srl_len)
                else:
                    ref_cap = ref_cap_temp
                    ref_cap.append(ref_cap_temp[-1])

                txt_graph_dict = self.get_captions_srl_graph(ref_cap)
                for i, g_dict in enumerate(txt_graph_dict):
                    g_node_role.get(name)[i:i + 1] = g_dict['node_roles']
                    g_rel_edge.get(name)[i:i + 1] = g_dict['rel_edges']
                    g_sent_id.get(name)[i:i + 1] = g_dict['sent_ids']
                    g_sent_len.get(name)[i:i + 1] = g_dict['sent_lens']
                    g_verb_mask.get(name)[i:i + 1] = g_dict['verb_masks']
                    g_noun_mask.get(name)[i:i + 1] = g_dict['noun_masks']


class SRGBertRepresentation:
    def __init__(self, opt, mode='train'):
        super(SRGBertRepresentation, self).__init__()
        self.opt = opt
        self.mode = mode

        self.num_verbs = 4  # num_verbs
        self.num_nouns = 6  # num_nouns
        self.srl_len = 12  # num of caption for each video
        self.max_token_in_sent = 30  #

        self.feat_dir = os.path.join(self.opt.data_dir, 'feats/')

        if self.opt.dataset_name == 'TVQA':
            self.ann_dir = os.path.join(self.opt.data_dir, 'tvqa/')

        elif self.opt.dataset_name == 'MSRVTT_QA':
            self.ann_dir = os.path.join(self.opt.data_dir, 'msrvtt_qa/')
            self.anns = load_ujson(os.path.join(self.ann_dir, f'{mode}_qa.json'))
            self.video_captions = load_ujson(
                os.path.join(self.ann_dir, 'video_caption_pair.json'))

        self.video_names = set([f"video{q['video_id']}" for q in self.anns])
        self.video_names = sorted(list(self.video_names))
        random.shuffle(self.video_names)

        self.role2int = {}
        for i, role in enumerate(ROLES):
            self.role2int[role] = i
            self.role2int['C-%s' % role] = i
            self.role2int['R-%s' % role] = i

        self.sent_graphs = load_ujson(
            os.path.join(self.ann_dir, 'cap_sent2rolegraph.augment.json'))
        self.word2int = load_ujson(os.path.join(self.ann_dir, 'word2int.json'))

        self.sent_bert = PretrainBertEmbedding(opt)

    def h5_video_sre_embeds(self):
        out_file = os.path.join(self.feat_dir, f'{self.mode}_srg_bert.h5')
        h5_fd = h5py.File(out_file, 'w')
        g_sent_bert = h5_fd.create_group('sent_bert')
        g_node_role = h5_fd.create_group('node_role')
        g_adjacency_matrix = h5_fd.create_group('adj_matrix')
        g_verb_token_id = h5_fd.create_group('verb_token_mask')
        g_noun_token_id = h5_fd.create_group('noun_token_mask')

        for name in tqdm(self.video_names):
            vid_sents = self.video_captions[name]
            if len(vid_sents) >= self.srl_len:
                vid_sents = sort_text(vid_sents, self.srl_len)
            else:
                vid_sents.append(vid_sents[-1])
            # generate sentences srg
            vid_sent_graph = [self.sentence_sr_graph(sent, self.sent_graphs[sent])
                              for sent in vid_sents]
            # save to h5
            g_node_role.create_dataset(name, (self.srl_len, 10), dtype=np.int32)
            g_adjacency_matrix.create_dataset(name, (self.srl_len, 11, 11),
                                              dtype=np.float32)
            g_verb_token_id.create_dataset(name, (self.srl_len, 4, 30),
                                           dtype=np.bool)
            g_noun_token_id.create_dataset(name, (self.srl_len, 6, 30),
                                           dtype=np.bool)
            sent_bert_i = g_sent_bert.create_group(name=name)
            for i, g in enumerate(vid_sent_graph):
                g_node_role.get(name)[i:i + 1] = g['node_role']
                g_adjacency_matrix.get(name)[i:i + 1] = g['adj_matrix']
                g_verb_token_id.get(name)[i:i + 1] = g['verb_token_id']
                g_noun_token_id.get(name)[i:i + 1] = g['noun_token_id']
                sent_bert_i.create_dataset(name=f'{i}', data=g['sent_bert'],
                                           dtype=np.float32)
        h5_fd.close()

    def sentence_sr_graph(self, sent, graph):
        graph_nodes, graph_edges = graph
        sent_graph = EasyDict()

        verb_node2idxs, noun_node2idxs = {}, {}
        edges = []
        sent_graph['node_role'] = np.zeros((self.num_verbs + self.num_nouns,),
                                           np.int32)
        # root node
        sent_embed, sent_token = self.sent_bert.sentence_embedding(sent,
                                                                   pooling='token')
        sent_graph['sent_bert'] = sent_embed
        token_index = {}
        for i in range(len(sent_token[1:-1])):
            token_index[sent_token[1 + i]] = i

        # graph: add verb nodes
        node_idx = 1
        sent_graph['verb_token_id'] = np.zeros((self.num_verbs,
                                                self.max_token_in_sent), np.bool)
        for knode, vnode in graph_nodes.items():
            k = node_idx - 1
            if k >= self.num_verbs:
                break
            if vnode['role'] == 'V' and np.min(
                    vnode['spans']) < self.max_token_in_sent:
                verb_node2idxs[knode] = node_idx
                words = ''
                for word in vnode['words']:
                    words += f'{word} '
                tokens = self.sent_bert.tokenizer.tokenize(
                    '[CLS] ' + words + '[SEP]')
                for token in tokens[1:-1]:
                    if token_index[token] < self.max_token_in_sent:
                        sent_graph['verb_token_id'][k][token_index[token]] = True

                sent_graph['node_role'][node_idx - 1] = self.role2int['V']
                # add root to verb edge
                edges.append((0, node_idx))
                node_idx += 1

        # graph: add noun nodes
        node_idx = 1 + self.num_verbs
        sent_graph['noun_token_id'] = np.zeros((self.num_nouns,
                                                self.max_token_in_sent), np.bool)
        for knode, vnode in graph_nodes.items():
            k = node_idx - self.num_verbs - 1
            if k >= self.num_nouns:
                break
            if vnode['role'] not in ['ROOT', 'V'] and np.min(
                    vnode['spans']) < self.max_token_in_sent:
                noun_node2idxs[knode] = node_idx
                words = ''
                for word in vnode['words']:
                    words += f'{word} '
                tokens = self.sent_bert.tokenizer.tokenize(
                    '[CLS] ' + words + '[SEP]')
                for token in tokens[1:-1]:
                    if token_index[token] < self.max_token_in_sent:
                        sent_graph['noun_token_id'][k][token_index[token]] = True

                sent_graph['node_role'][node_idx - 1] = \
                    self.role2int.get(vnode['role'], self.role2int['NOUN'])
                node_idx += 1

        # graph: add verb_node to noun_node edges
        for e in graph_edges:
            if e[0] in verb_node2idxs and e[1] in noun_node2idxs:
                edges.append((verb_node2idxs[e[0]], noun_node2idxs[e[1]]))
                edges.append((noun_node2idxs[e[1]], verb_node2idxs[e[0]]))

        num_nodes = 1 + self.num_verbs + self.num_nouns
        rel_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        for src_nodeidx, tgt_nodeidx in edges:
            rel_matrix[tgt_nodeidx, src_nodeidx] = 1

        # row norm
        for i in range(num_nodes):
            s = np.sum(rel_matrix[i])
            if s > 0:
                rel_matrix[i] /= s

        sent_graph['adj_matrix'] = rel_matrix
        return sent_graph


if __name__ == '__main__':
    from option import args

    # generate semantic role graph of video -> word embedding
    # dataset = SRGRepresentation(args)
    # dataset.generate_video_srg_pair()

    # generate semantic role graph of video -> bert embedding
    sre = SRGBertRepresentation(args)
    sre.h5_video_sre_embeds()
    
    sre = SRGBertRepresentation(args, mode='val')
    sre.h5_video_sre_embeds()

    sre = SRGBertRepresentation(args, mode='test')
    sre.h5_video_sre_embeds()
