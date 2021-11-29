# @File : sentence_semantic_role_labeling.py.py
# @Github : https://github.com/cshizhe/hgr_v2t

import json
import os

import numpy as np
import spacy
from allennlp.predictors.predictor import Predictor
from easydict import EasyDict
from tqdm import tqdm

from utils.tools import load_ujson, save_json

SRL_BERT = (
    "https://s3-us-west-2.amazonaws.com/allennlp/models/bert-base-srl-2019.06.17.tar.gz")
predictor2 = Predictor.from_path(archive_path=SRL_BERT,
                                 predictor_name='semantic-role-labeling',
                                 cuda_device=0)


def semantic_role_labeling(ann_path):
    # predictor = Predictor.from_path(
    #     "https://s3-us-west-2.amazonaws.com/allennlp/models/bert-base-srl-2019.06.17.tar.gz",
    #     cuda_device=0)

    ref_caps = json.load(open(os.path.join(ann_path, 'video_caption_pair.json')))
    uniq_sents = set()
    for key, sents in ref_caps.items():
        for sent in sents:
            uniq_sents.add(sent)
    uniq_sents = list(uniq_sents)
    print('unique sents', len(uniq_sents))
    # Predicts the semantic roles of the supplied sentence tokens and returns
    # a dictionary with the misc.
    outs = {}
    for i, sent in enumerate(uniq_sents):
        if sent in outs:
            continue
        try:
            # out = predictor.predict_tokenized(sent.split())
            out = predictor2.predict_tokenized(sent.split())
        except KeyboardInterrupt:
            break
        except:
            continue
        outs[sent] = out
        if i % 1000 == 0:
            print('finish %d / %d = %.2f%%' % (
                i, len(uniq_sents), i / len(uniq_sents) * 100))
    out_file = os.path.join(anno_dir, 'sent2srl.json')
    with open(out_file, 'w') as f:
        json.dump(outs, f)

    return outs


def create_role_graph_data(srl_data):
    words = srl_data['words']
    verb_items = srl_data['verbs']

    graph_nodes = {}
    graph_edges = []

    root_name = 'ROOT'
    graph_nodes[root_name] = {'words': words, 'spans': list(range(0, len(words))),
                              'role': 'ROOT'}

    # parse all verb_items
    phrase_items = []
    for i, verb_item in enumerate(verb_items):
        tags = verb_item['tags']
        tag2idxs = {}
        tagname_counter = {}  # multiple args of the same role
        for t, tag in enumerate(tags):
            if tag == 'O':
                continue
            if t > 0 and tag[0] != 'B':
                # deal with some parsing mistakes, e.g. (B-ARG0, O-ARG1)
                # change it into (B-ARG0, B-ARG1)
                if tag[2:] != tags[t - 1][2:]:
                    tag = 'B' + tag[1:]
            tagname = tag[2:]
            if tag[0] == 'B':
                if tagname not in tagname_counter:
                    tagname_counter[tagname] = 1
                else:
                    tagname_counter[tagname] += 1
            new_tagname = '%s:%d' % (tagname, tagname_counter[tagname])
            tag2idxs.setdefault(new_tagname, [])
            tag2idxs[new_tagname].append(t)
        if len(tagname_counter) > 1 and 'V' in tagname_counter and \
                tagname_counter['V'] == 1:
            phrase_items.append(tag2idxs)

    node_idx = 1
    spanrole2nodename = {}
    for i, phrase_item in enumerate(phrase_items):
        # add verb node to graph
        tagname = 'V:1'
        role = 'V'
        spans = phrase_item[tagname]
        spanrole = '-'.join([str(x) for x in spans] + [role])
        if spanrole in spanrole2nodename:
            continue
        node_name = str(node_idx)
        tag_words = [words[idx] for idx in spans]
        graph_nodes[node_name] = {
            'role': role, 'spans': spans, 'words': tag_words,
        }
        spanrole2nodename[spanrole] = node_name
        verb_node_name = node_name
        node_idx += 1

        # add arg nodes and edges of the verb node
        for tagname, spans in phrase_item.items():
            role = tagname.split(':')[0]
            if role != 'V':
                spanrole = '-'.join([str(x) for x in spans] + [role])
                if spanrole in spanrole2nodename:
                    node_name = str(spanrole2nodename[spanrole])
                else:
                    # add new node or duplicate a node with a different role
                    node_name = str(node_idx)
                    tag_words = [words[idx] for idx in spans]
                    graph_nodes[node_name] = {
                        'role': role, 'spans': spans, 'words': tag_words,
                    }
                    spanrole2nodename[spanrole] = node_name
                    node_idx += 1
                # add edge
                graph_edges.append((verb_node_name, node_name, role))

    return graph_nodes, graph_edges


def augment_srg(sent2graph):
    # Augment Graph if no SRL is detected (no verb)
    nlp = spacy.load("en_core_web_sm")
    for sent, graph in sent2graph.items():
        nodes, edges = graph
        node_idx = len(nodes)

        # add noun and verb word node if no noun and no noun phrases
        if len(nodes) == 1:
            doc = nlp(sent)
            assert len(doc) == len(nodes['ROOT']['words']), sent

            # add noun nodes
            for w in doc.noun_chunks:
                node_name = str(node_idx)
                nodes[node_name] = {
                    'role': 'NOUN', 'spans': np.arange(w.start, w.end).tolist()
                }
                nodes[node_name]['words'] = [doc[j].text for j in
                                             nodes[node_name]['spans']]
                node_idx += 1
            if len(nodes) == 1:
                for w in doc:
                    node_name = str(node_idx)
                    if w.tag_.startswith('NN'):
                        nodes[node_name] = {
                            'role': 'NOUN', 'spans': [w.i], 'words': [w.text],
                        }
                        node_idx += 1

            # add verb nodes
            for w in doc:
                node_name = str(node_idx)
                if w.tag_.startswith('VB'):
                    nodes[node_name] = {
                        'role': 'V', 'spans': [w.i], 'words': [w.text],
                    }
                    node_idx += 1

        sent2graph[sent] = (nodes, edges)

    return sent2graph


def generate_cap_srg(anno_dir):
    # Semantic Role Labeling
    if not os.path.exists(os.path.join(anno_dir, 'cap_sent2srl.json')):
        sent2srl = semantic_role_labeling(anno_dir)
    else:
        sent2srl_file = os.path.join(anno_dir, 'cap_sent2srl.json')
        sent2srl = json.load(open(sent2srl_file))

    # Convert Sentence's semantic role label to Role Graph
    sent2rg_file = os.path.join(anno_dir, 'cap_sent2rolegraph.json')
    if not os.path.exists(sent2rg_file):
        sent2graph = {}
        for sent, srl in sent2srl.items():
            try:
                graph_nodes, graph_edges = create_role_graph_data(srl)
                sent2graph[sent] = (graph_nodes, graph_edges)
            except:
                print(sent)
        json.dump(sent2graph, open(sent2rg_file, 'w'))
    else:
        sent2graph = json.load(open(sent2rg_file, 'r'))

    n = 0
    for sent, graph in sent2graph.items():
        if len(graph[0]) == 1:  # node; graph[1]->edge
            n += 1
    print('#sents without non-root nodes:', n)

    sent2rga_file = os.path.join(anno_dir, 'cap_sent2rolegraph.augment.json')
    if not os.path.exists(sent2rga_file):
        sent2graph = augment_srg(sent2graph)
    else:
        sent2graph = json.load(open(sent2rga_file))

    print(len(sent2graph))
    json.dump(sent2graph, open(sent2rga_file, 'w'))


def generate_que_srg(anno_dir):
    que_path = os.path.join(anno_dir, 'train_ques.json')
    ques_dict = EasyDict()
    if not os.path.exists(que_path):
        ann = json.load(open(os.path.join(anno_dir, 'train_qa.json')))
        for i in range(len(ann)):
            temp = ann[i]['question'][:-1]  # remove '?'
            temp = temp.split('-')
            if len(temp) == 1:
                ques_dict[f"{ann[i]['question']}"] = temp[0]
            elif len(temp) == 2:
                ques_dict[f"{ann[i]['question']}"] = f'{temp[0]} - {temp[1]}'
            elif len(temp) == 3:
                ques_dict[
                    f"{ann[i]['question']}"] = f'{temp[0]} - {temp[1]} - {temp[2]}'
            elif len(temp) == 4:
                ques_dict[
                    f"{ann[i]['question']}"] = \
                    f'{temp[0]} - {temp[1]} - {temp[2]} - {temp[3]}'
                print(ques_dict[f"{ann[i]['question']}"])
        json.dump(ques_dict, open(que_path, 'w'))
    else:
        ques_dict = json.load(open(que_path, 'r'))

    outs = {}
    for i, sent_dict in enumerate(ques_dict.items()):
        sent_o, sent = sent_dict
        if sent in outs:
            continue
        try:
            out = predictor2.predict_tokenized(sent.split())
        except KeyboardInterrupt:
            break
        except:
            continue
        outs[sent] = out
        if i % 1000 == 0:
            print('finish %d / %d = %.2f%%' % (
                i, len(ques_dict), i / len(ques_dict) * 100))
    out_file = os.path.join(anno_dir, 'ques2srl.json')
    with open(out_file, 'w') as f:
        json.dump(outs, f)

    sent2graph = {}
    for sent, srl in outs.items():
        try:
            graph_nodes, graph_edges = create_role_graph_data(srl)
            sent2graph[sent] = (graph_nodes, graph_edges)
        except:
            print(sent)
    json.dump(sent2graph, open(os.path.join(anno_dir,
                                            'ques2rolegraph.json'), 'w'))
    n = 0
    for sent, graph in sent2graph.items():
        if len(graph[0]) == 1:
            n += 1
    print('#sents without non-root nodes:', n)

    # augment
    sent2graph = augment_srg(sent2graph)

    print(len(sent2graph))
    json.dump(sent2graph, open(os.path.join(anno_dir,
                                            'ques2rolegraph.augment.json'), 'w'))


def combine_caption(path):
    cap_train = load_ujson(os.path.join(path, 'train_val_videodatainfo.json'))
    cap_test = load_ujson(os.path.join(path, 'test_videodatainfo.json'))
    cap_train = cap_train['sentences']
    cap_test = cap_test['sentences']
    caps = cap_train + cap_test

    cap_dict = EasyDict()
    for i in tqdm(range(10000)):
        video_id = f'video{i}'
        cap = [caps[j]['caption'] for j in range(len(caps)) if
               caps[j]['video_id'] == video_id]
        cap_dict[video_id] = cap
    save_json(cap_dict, os.path.join(path, 'video_caption_pair.json'))

    return cap_dict


if __name__ == '__main__':
    root_dir = '/media/kaka/HD2T/dataset/VideoQA/MSRVTT/2016/msrvtt_qa'

    anno_dir = os.path.join(root_dir, 'RET')
    os.makedirs(anno_dir, exist_ok=True)  # 递归目录创建

    if not os.path.exists(os.path.join(anno_dir, 'video_caption_pair.json')):
        ann_cap = combine_caption(os.path.join(anno_dir, 'caption_org'))
    else:
        ann_cap = load_ujson(os.path.join(anno_dir, 'video_caption_pair.json'))

    # generate semantic role graph for MSRVTT captions
    generate_cap_srg(anno_dir)

    # -------------------- question srl
    generate_que_srg(root_dir)
