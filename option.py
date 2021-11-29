# @File : option.py
# @Time : 2019/12/28 
# @Email : jingjingjiang2017@gmail.com 
import json
import os
import argparse
from easydict import EasyDict

# MSRVTT-QA
MSRVTT_QA_dir = '/media/kaka/HD2T/Dataset/VideoQA/MSRVTT/2016/'
# KnowIT
KnowIt_dir = '/media/kaka/HD2T/dataset/VideoQA/KnowIT/'


parser = argparse.ArgumentParser(description='parameters')
# ================================== dataset args ==============================
parser.add_argument('--dataset_name', type=str, default='MSRVTT_QA',
                    help='MSRVTT_QA or KnowIT')
parser.add_argument('--data_dir', type=str,
                    default='/media/kaka/HD2T/Dataset/VideoQA/MSRVTT/2016/')
parser.add_argument('--exp_path', type=str,
                    default='Exp_DiVS/',
                    help='dir to save experiment misc')
parser.add_argument('--exp_name', type=str,
                    default='debug')

# =================================== model args ===============================
parser.add_argument('--config', default='./config/msrvtt_config.json',
                    help='parameters of VideoQA model.')

# ================================= training args ==============================
parser.add_argument('--bs', default=256, type=int)
parser.add_argument('--n_works', default=8, type=int)
parser.add_argument('--epoch', default=80, type=int,
                    help='epoch')
parser.add_argument('--start_epoch', default=0, type=int,
                    help='start epoch')
parser.add_argument('--val_every_epoch', default=1, type=int,
                    help='epoch of validation.')
parser.add_argument('--val_every_iter', default=80, type=int,
                    help='interval of validation.')
parser.add_argument('--freq_print', default=1, type=int,
                    help='print result very {} epoch')

parser.add_argument('--grad_norm', default=1., type=float,
                    help='max norm of the gradients.')  # 梯度的最大范数
parser.add_argument('--lr', default=0.0001, type=float,
                    help='learning rate')
parser.add_argument('--lr_scheduler', default=True, type=bool,
                    help='max norm of the gradients.')
parser.add_argument('--num_head', default=8, type=int,
                    help='max norm of the gradients.')

# parser.add_argument("--local_rank", type=int, default=-1,
#                     help='node rank for distributed training')
# ==============================================================================

args = parser.parse_args()

if args.dataset_name == 'MSRVTT_QA':
    args.data_dir = MSRVTT_QA_dir
else:
    args.data_dir = KnowIt_dir

if not os.path.exists(os.path.join(args.exp_path, args.exp_name)):
    os.makedirs(os.path.join(args.exp_path, args.exp_name), exist_ok=True)
    
model_config = EasyDict(json.load(open(args.config, 'r')))

# print(args)
