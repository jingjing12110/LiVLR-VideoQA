# @File: test.py
# @Time: 2020/8/1
# @Email: jingjingjiang2017@gmail.com
import time
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from thop import profile

from data.msrvtt_dataset import TestMRSVTTDataset as TestDataset
from model.h2gr import H2GR
from option import args, model_config


def count_parameters(m, x, y):
    total_params = 0
    for p in m.parameters():
        total_params += torch.DoubleTensor([p.numel()])
    m.total_params[0] = total_params
    

def profile(model: nn.Module, inputs, custom_ops=None, verbose=True):
    from thop.profile import register_hooks, prRed
    handler_collection = {}
    types_collection = set()
    if custom_ops is None:
        custom_ops = {}

    def add_hooks(m: nn.Module):
        m.register_buffer('total_ops', torch.zeros(1, dtype=torch.float64))
        m.register_buffer('total_params', torch.zeros(1, dtype=torch.float64))

        m_type = type(m)

        fn = None
        # if defined both op maps, use custom_ops to overwrite.
        if m_type in custom_ops:
            fn = custom_ops[m_type]
            if m_type not in types_collection and verbose:
                print("[INFO] Customize rule %s() %s." % (fn.__qualname__, m_type))
        elif m_type in register_hooks:
            fn = register_hooks[m_type]
            if m_type not in types_collection and verbose:
                print("[INFO] Register %s() for %s." % (fn.__qualname__, m_type))
        else:
            if m_type not in types_collection and verbose:
                prRed("[WARN] Cannot find rule for %s. Treat it as zero Macs and zero Params." % m_type)

        if fn is not None:
            handler_collection[m] = (m.register_forward_hook(fn), m.register_forward_hook(count_parameters))
        types_collection.add(m_type)

    prev_training_status = model.training

    model.eval()
    model.apply(add_hooks)

    with torch.no_grad():
        model(inputs)

    def dfs_count(module: nn.Module, prefix="\t") -> (int, int):
        total_ops, total_params = 0, 0
        for m in module.children():
            # if not hasattr(m, "total_ops") and not hasattr(m, "total_params"):  # and len(list(m.children())) > 0:
            #     m_ops, m_params = dfs_count(m, prefix=prefix + "\t")
            # else:
            #     m_ops, m_params = m.total_ops, m.total_params
            if m in handler_collection and not isinstance(m, (nn.Sequential, nn.ModuleList)):
                m_ops, m_params = m.total_ops.item(), m.total_params.item()
            else:
                m_ops, m_params = dfs_count(m, prefix=prefix + "\t")
            total_ops += m_ops
            total_params += m_params
        #  print(prefix, module._get_name(), (total_ops.item(), total_params.item()))
        return total_ops, total_params

    total_ops, total_params = dfs_count(model)

    # reset model to original status
    model.train(prev_training_status)
    for m, (op_handler, params_handler) in handler_collection.items():
        op_handler.remove()
        params_handler.remove()
        m._buffers.pop("total_ops")
        m._buffers.pop("total_params")

    return total_ops, total_params



def test_5c_acc(model, ckpt, cate):
    test_data = TestDataset(args, cate=cate)
    test_loader = DataLoader(test_data,
                             batch_size=1,
                             shuffle=False,
                             collate_fn=test_data.pad_collate,
                             num_workers=0,
                             pin_memory=True)
    # model.load_state_dict(torch.load(os.path.join(exp_dir, ckpt)))
    test_prob = []
    test_target = []
    num_test_iter = 0
    tbar = tqdm(total=len(test_loader), ascii=True)
    tbar.set_description(f'{ckpt}')
    total_time = 0
    ms_qs = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            target = batch['target']
            batch.pop('target')
            
            # start_time = time.time()
            out_dict = model(batch)
            # torch.cuda.synchronize()
            # end_time = time.time()
            # q_ms = (end_time - start_time) / out_dict['logit'].shape[0] * 1000
            # ms_qs.append(q_ms)
            
            test_prob.append(
                out_dict['logit'].detach().argmax(1).cpu().numpy())
            test_target.append(target.numpy())
            num_test_iter += 1
            tbar.update(1)
    print(f'mean ms_q: {np.array(ms_qs).mean():.2f}')
    
    # test_prob = np.concatenate(test_prob, axis=0)
    # test_target = np.concatenate(test_target, axis=0)
    # average_acc = float(np.mean(test_target == test_prob))
    #
    # print(f'total_time: {total_time / test_prob.shape[0]:.2f}')

    # without answer set (0/1)
    # test_prob = test_prob[test_target != 1]
    # test_target = test_target[test_target != 1]
    # average_acc_topK = float(np.mean(test_target == test_prob))
    #
    # tbar.set_postfix(average_acc=f'{average_acc:.4f}',
    #                  # average_acc_topK=f'{average_acc_topK:.4f}'
    #                  )
    tbar.close()

    return average_acc


if __name__ == '__main__':
    args.exp_name = 'vse'
    exp_time = '2020-10-06T-19-55-33'  # '2020-10-30T-15-39-50'
    exp_dir = os.path.join(args.exp_path, args.exp_name, f'{exp_time}/ckpts')
    m_txt = open(os.path.join(args.exp_path, args.exp_name,
                              f'{exp_time}/test.txt'), 'a+')

    # ================================================================== model
    model_config.num_head = args.num_head
    args.model_config = model_config

    model = H2GR(model_config)
    model = model.cuda(device=model_config.gpu_ids[0])
    print(f'num of params: '
          f'{sum(x.numel() for x in model.parameters() if x.requires_grad)/1e6}')
    
    # model = nn.DataParallel(model, device_ids=model_config.gpu_ids)
    model.eval()

    # all_ckpt = [ckp for ckp in os.listdir(exp_dir) if ckp.endswith('pth')]
    all_ckpt = [ckp for ckp in os.listdir(exp_dir) if ckp.startswith('best')]
    # all_ckpt = ['model_epoch_97.pth']
    for ckpt in sorted(all_ckpt):
        m_txt.write(f'ckpt_name:{ckpt}\n')
        # average_acc, average_acc_topK = test_one_time(model, ckpt)
        for cate in [None, 'what', 'who', 'how', 'when', 'where']:
            average_acc = test_5c_acc(model, ckpt, cate)
            m_txt.write(f'{cate}:  '
                        f'average_acc:{average_acc:.4f}\n'
                        # f'average_acc_topK:{average_acc_topK:.4f}\n'
                        )
    m_txt.close()
