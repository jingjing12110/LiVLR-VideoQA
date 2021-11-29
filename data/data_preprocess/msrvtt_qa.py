# @File : msrvtt.py
# @Github : https://github.com/thaolmk54/hcrn-videoqa

import argparse
import json
import os
import pickle
from collections import Counter

import nltk
# nltk.download('punkt')
# nltk.download()
import numpy as np

import utils.data_util as utils


def process_questions(args):
    ''' Encode question tokens'''
    print('Loading data')
    with open(args.annotation_file, 'r') as dataset_file:
        instances = json.load(dataset_file)

    # Either create the vocab or load it from disk
    if args.mode in ['train']:
        print('Building vocab')
        answer_cnt = {}
        for instance in instances:
            answer = instance['answer']
            answer_cnt[answer] = answer_cnt.get(answer, 0) + 1

        answer_token_to_idx = {'<UNK0>': 0, '<UNK1>': 1}
        answer_counter = Counter(answer_cnt)
        frequent_answers = answer_counter.most_common(args.answer_top)
        total_ans = sum(item[1] for item in answer_counter.items())
        total_freq_ans = sum(item[1] for item in frequent_answers)
        print("Number of unique answers:", len(answer_counter))
        print("Total number of answers:", total_ans)
        print("Top %i answers account for %f%%" % (
            len(frequent_answers), total_freq_ans * 100.0 / total_ans))

        for token, cnt in Counter(answer_cnt).most_common(args.answer_top):
            answer_token_to_idx[token] = len(answer_token_to_idx)
        print('Get answer_token_to_idx, num: %d' % len(answer_token_to_idx))

        question_token_to_idx = {'<NULL>': 0, '<UNK>': 1}
        for i, instance in enumerate(instances):
            question = instance['question'].lower()[:-1]
            for token in nltk.word_tokenize(question):
                if token not in question_token_to_idx:
                    question_token_to_idx[token] = len(question_token_to_idx)
        print('Get question_token_to_idx')
        print(len(question_token_to_idx))

        vocab = {
            'question_token_to_idx': question_token_to_idx,
            'answer_token_to_idx': answer_token_to_idx,
            'question_answer_token_to_idx': {'<NULL>': 0, '<UNK>': 1}
        }

        print('Write into %s' % args.vocab_json.format(args.dataset,
                                                       args.dataset))
        with open(args.vocab_json.format(args.dataset, 
                                         f'Top{args.answer_top}'), 'w') as f:
            json.dump(vocab, f, indent=4)
    else:
        print('Loading vocab')
        with open(args.vocab_json.format(args.dataset,
                                         f'Top{args.answer_top}'), 'r') as f:
            vocab = json.load(f)

        # 统计答案在vocab中的占比
        num_in = 0
        num_out = 0
        for idx, instance in enumerate(instances):
            if instance['answer'] in vocab['answer_token_to_idx']:
                num_in += 1
            else:
                num_out += 1
        print(f'in answers set: {num_in/len(instances) * 100: .4f} %')

    # Encode all questions
    print('Encoding data')
    questions_encoded = []
    questions_len = []
    question_ids = []
    video_ids_tbw = []
    video_names_tbw = []
    all_answers = []
    for idx, instance in enumerate(instances):
        question = instance['question'].lower()[:-1]
        question_tokens = nltk.word_tokenize(question)
        question_encoded = utils.encode(question_tokens,
                                        vocab['question_token_to_idx'],
                                        allow_unk=True)
        questions_encoded.append(question_encoded)
        questions_len.append(len(question_encoded))
        question_ids.append(idx)
        im_name = instance['video_id']
        video_ids_tbw.append(im_name)
        video_names_tbw.append(im_name)

        if instance['answer'] in vocab['answer_token_to_idx']:
            answer = vocab['answer_token_to_idx'][instance['answer']]
        elif args.mode in ['train']:
            answer = 0
        elif args.mode in ['val', 'test']:
            answer = 1

        all_answers.append(answer)
    max_question_length = max(len(x) for x in questions_encoded)
    for qe in questions_encoded:
        while len(qe) < max_question_length:
            qe.append(vocab['question_token_to_idx']['<NULL>'])

    questions_encoded = np.asarray(questions_encoded, dtype=np.int32)
    questions_len = np.asarray(questions_len, dtype=np.int32)
    print(questions_encoded.shape)

    glove_matrix = None
    if args.mode == 'train':
        token_itow = {i: w for w, i in vocab['question_token_to_idx'].items()}
        print("Load glove from %s" % args.glove_pt)
        glove = pickle.load(open(args.glove_pt, 'rb'))
        dim_word = glove['the'].shape[0]
        glove_matrix = []
        for i in range(len(token_itow)):
            vector = glove.get(token_itow[i], np.zeros((dim_word,)))
            glove_matrix.append(vector)
        glove_matrix = np.asarray(glove_matrix, dtype=np.float32)
        print(glove_matrix.shape)

    print('Writing',
          args.output_pt.format(args.dataset, args.dataset, args.mode))
    obj = {
        'questions': questions_encoded,
        'questions_len': questions_len,
        'question_id': question_ids,
        'video_ids': np.asarray(video_ids_tbw),
        'video_names': np.array(video_names_tbw),
        'answers': all_answers,
        'glove': glove_matrix,
    }
    with open(args.output_pt.format(args.dataset, args.dataset, args.mode),
              'wb') as f:
        pickle.dump(obj, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='msrvtt-qa',
                        choices=['msrvtt-qa', 'msvd-qa'], type=str)
    parser.add_argument('--data_dir',
                        default='/media/kaka/HD2T/dataset/VideoQA/MSRVTT/2016/',
                        type=str)
    parser.add_argument('--glove_pt',
                        default='/media/kaka/HD2T/dataset/VideoQA/glove/glove.840.300d.pkl',
                        help='glove pickle file, should be a map whose key are '
                             'words and value are word vectors represented by '
                             'numpy arrays. Only needed in train mode')
    parser.add_argument('--output_pt', type=str,
                        default='data/{}/{}_{}_questions.pt')
    parser.add_argument('--vocab_json', type=str,
                        default='data/{}/{}_vocab.json')

    parser.add_argument('--answer_top', default=6000, type=int)
    parser.add_argument('--mode', choices=['train', 'val', 'test'],
                        default='test')
    parser.add_argument('--question_type', default='none',
                        choices=['frameqa', 'action', 'transition', 'count',
                                 'none'])
    parser.add_argument('--seed', type=int, default=666)

    args = parser.parse_args()
    np.random.seed(args.seed)

    args.annotation_file = os.path.join(args.data_dir,
                                        f'msrvtt_qa/{args.mode}_qa.json')
    # check if data folder exists
    if not os.path.exists('./data/{}'.format(args.dataset)):
        os.makedirs('./data/{}'.format(args.dataset))

    process_questions(args)
