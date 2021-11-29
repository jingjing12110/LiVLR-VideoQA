# @File : train_msrvtt.py
# @Time : 2020/5/13
# @Email : jingjingjiang2017@gmail.com
import json
import os
from datetime import datetime
import shutil

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm

from data.msrvtt_dataset import MSRVTTDataset as FeatureDataset
from data.msrvtt_dataset import DataLoaderX
from model.h2gr import H2GR
from option import args, model_config
from utils.tools import count_parameters

# import torch.distributed as dist
# from torch.utils.data.distributed import DistributedSampler

seed = 9
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
cudnn.benchmark = False
cudnn.deterministic = True
TIMESTAMP = "{0:%Y-%m-%dT-%H-%M-%S/}".format(datetime.now())


class Train:
    def __init__(self, args, model, data: list):
        super(Train, self).__init__()
        self.args = args
        self.exp_name = args.exp_name
        self.model = model
        self.train_loader, self.val_loader = data
        self.best_val_acc = 0.

        self.writer = SummaryWriter(os.path.join(self.args.exp_path,
                                                 self.exp_name,
                                                 f'{TIMESTAMP}/logs'))
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.args.lr,
            betas=(0.9, 0.999),
            weight_decay=0.5
        )

        if self.args.lr_scheduler:
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=[60, 70],
                gamma=0.1,
            )
        else:
            self.scheduler = None

        self.criterion = nn.CrossEntropyLoss()

        self.train()

    def train(self):
        if self.args.start_epoch:
            self.model.load_state_dict(torch.load(
                os.path.join(self.args.exp_path, self.exp_name,
                             f'2020-08-31T-14-31-58/ckpts/model_epoch_'
                             f'{self.args.start_epoch - 1}.pth')))
            self.args.num_train_iter = (self.args.start_epoch - 1) * len(
                self.train_loader)
            self.args.num_val_iter = (self.args.start_epoch - 1) * len(
                self.val_loader)
        else:
            self.args.num_train_iter = 0
            self.args.num_val_iter = 0

        if not os.path.exists(os.path.join(self.args.exp_path, self.exp_name,
                                           f'{TIMESTAMP}/ckpts')):
            os.makedirs(
                os.path.join(self.args.exp_path, self.exp_name,
                             f'{TIMESTAMP}/ckpts'))

        # save args
        with open(os.path.join(self.args.exp_path, self.args.exp_name,
                               f'{TIMESTAMP}/args.json'), 'a') as outfile:
            json.dump({'train_args': vars(self.args)},
                      outfile, ensure_ascii=False, indent=4)
            outfile.write('\n')

        # save model architecture
        if len(self.args.model_config.gpu_ids) > 1:
            with open(os.path.join(self.args.exp_path, self.args.exp_name,
                                   f'{TIMESTAMP}/model.txt'), 'w') as f:
                f.write(str(dict(self.model.module.__dict__['_modules'])))
        else:
            with open(os.path.join(self.args.exp_path, self.args.exp_name,
                                   f'{TIMESTAMP}/model.txt'), 'w') as f:
                f.write(str(dict(self.model.__dict__['_modules'])))

        model_file = [m for m in os.listdir('./model') if m.startswith('h2gr')]
        for m in model_file:
            shutil.copyfile(
                os.path.join(f'./model/{m}'),
                os.path.join(self.args.exp_path,
                             f'{self.args.exp_name}/{TIMESTAMP}/{m}'))

        # _, val_acc = self.val(epoch)
        # if val_acc > self.best_val_acc:
        #     self.best_val_acc = val_acc

        print('=' * 100)
        self.model.train()
        for epoch in range(self.args.start_epoch, self.args.epoch):
            self.writer.add_scalar(
                'learning rate',
                float(self.optimizer.param_groups[0]['lr']), epoch)
            # if len(self.args.model_config.gpu_ids) > 1:
            #     self.writer.add_scalar(
            #         'alpha',
            #         self.model.module.or_encoder.alpha, epoch)
            #     self.writer.add_scalar(
            #         'gamma',
            #         self.model.module.or_encoder.gamma, epoch)

            self.train_one_epoch(epoch)

            if epoch % self.args.val_every_epoch == 0:
                val_loss, val_acc = self.val(epoch)
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    torch.save(self.model.state_dict(), os.path.join(
                        self.args.exp_path, self.exp_name,
                        f'{TIMESTAMP}/ckpts/best_val_model.pth'))
                # if self.scheduler:
                #     self.scheduler.step()
                #     self.warmup_scheduler.dampen()
                #     # self.scheduler.step(val_loss)
                #     # self.scheduler.step(val_acc)
                #     # self.scheduler_warmup.step(epoch, val_loss)

            # if epoch == 15 and \
            #         self.optimizer.param_groups[0]['lr'] - self.args.lr == 0:
            #     self.optimizer.param_groups[0]['lr'] = 0.5 * self.args.lr

        self.writer.close()

    def train_one_epoch(self, epoch):
        total_loss = 0.
        total_acc = 0.
        tbar = tqdm(total=len(self.train_loader.dataset) // self.args.bs,
                    ascii=False)
        tbar.set_description(f'Epoch{epoch:2d}')
        for bs_idx, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            target = batch['target'].cuda()
            out_dict = self.model(batch)
            logit = out_dict.get('logit')

            loss = self.criterion(logit, target)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(),
                                     max_norm=self.args.grad_norm,
                                     norm_type=2)
            self.optimizer.step()
            total_loss += loss.detach()

            agreeing = (logit.detach().argmax(1) == target.detach()).float()
            total_acc += agreeing.sum().item()

            self.writer.add_scalar('Train/average_loss',
                                   total_loss / (bs_idx + 1),
                                   self.args.num_train_iter)
            self.writer.add_scalar('Train/average_acc',
                                   total_acc / (self.args.bs * (bs_idx + 1)),
                                   self.args.num_train_iter)

            self.writer.add_scalar('Train/batch_acc',
                                   agreeing.mean(),
                                   self.args.num_train_iter)
            self.writer.add_scalar('Train/batch_loss',
                                   loss.detach(),
                                   self.args.num_train_iter)

            self.args.num_train_iter += 1

            if bs_idx % self.args.freq_print == 0 and bs_idx > 0:
                tbar.set_postfix(
                    avg_loss=f'{total_loss / (bs_idx + 1):.4f}',
                    avg_acc=f'{total_acc / (self.args.bs * (bs_idx + 1)):.4f}')
                tbar.update(self.args.freq_print)
            # bs_idx += 1
            # if bs_idx >= len(self.train_loader):
            #     break
            # batch = pre_fetcher.next()
        tbar.close()

        m_txt = open(os.path.join(self.args.exp_path, self.exp_name,
                                  f'{TIMESTAMP}/train.txt'), 'a+')
        m_txt.write(f'epoch:{epoch}  '
                    f'avg_loss:{total_loss / (bs_idx + 1):.4f}  '
                    f'avg_acc:{total_acc / (self.args.bs * (bs_idx + 1)):.4f}\n')
        m_txt.close()
        # for each epoch
        if self.args.lr_scheduler:
            self.scheduler.step()
            # self.warmup_scheduler.dampen()

        torch.save(self.model.state_dict(),
                   os.path.join(self.args.exp_path, self.exp_name,
                                f'{TIMESTAMP}/ckpts/model_epoch_{epoch}.pth'))

    def val(self, epoch):
        self.model.eval()

        val_prob = []
        val_target = []  #
        # val_prob = np.zeros((len(self.val_loader), self.args.bs))
        # val_target = np.zeros((len(self.val_loader), self.args.bs))
        total_loss = 0.0

        pbar = tqdm(total=len(self.val_loader), ascii=True, ncols=60)
        pbar.set_description(f'Val')
        with torch.no_grad():  #
            for bs_idx, batch in enumerate(self.val_loader):
                out_dict = self.model(batch)

                logit, target = out_dict.get('logit'), out_dict.get('target')
                loss = self.criterion(logit, target)

                total_loss += loss.item() * target.shape[0]

                val_prob.append(logit.detach().argmax(1).cpu().numpy())
                val_target.append(target.detach().cpu().numpy())
                # val_prob[bs_idx, :] = logit.detach().argmax(1).cpu().numpy()
                # val_target[bs_idx, :] = target.detach().cpu().numpy()

                self.writer.add_scalar('Val/batch_loss',
                                       loss, self.args.num_val_iter)
                self.args.num_val_iter += 1
                pbar.update(1)

        val_target = np.concatenate(val_target, axis=0)
        val_prob = np.concatenate(val_prob, axis=0)
        # val_target = val_target.flatten()
        # val_prob = val_prob.flatten()
        average_acc = float(np.mean(val_target == val_prob))
        average_loss = total_loss / val_target.shape[0]

        self.writer.add_scalar('Val/average_acc', average_acc,
                               int(epoch / self.args.val_every_epoch))
        self.writer.add_scalar('Val/average_loss', average_loss,
                               int(epoch / self.args.val_every_epoch))

        # 计算去掉不在answer set中的答案（答案类别为0/1）的精度
        # val_prob = val_prob[val_target != 1]
        # val_target = val_target[val_target != 1]
        # average_acc_topK = float(np.mean(val_target == val_prob))

        pbar.close()
        print(f'  \t  avg_loss={average_loss:.4f}, '
              f'avg_acc={average_acc:.4f}'
              # f'avg_acc_topK={average_acc_topK:.4f}'
              )

        m_txt = open(os.path.join(self.args.exp_path, self.exp_name,
                                  f'{TIMESTAMP}/val.txt'), 'a+')
        m_txt.write(f'epoch:{epoch}  '
                    f'avg_loss:{average_loss:.4f}  '
                    f'avg_acc:{average_acc:.4f}\n'
                    )
        m_txt.close()

        self.model.train()
        return average_loss, average_acc

    def l1_loss(self):
        regularization_loss = torch.tensor(0., requires_grad=True).cuda(
            non_blocking=True)
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                # regularization_loss += torch.sum(abs(param))
                regularization_loss += torch.norm(param, 1)
        return regularization_loss


def main():
    # dist.init_process_group(backend='nccl')
    # torch.cuda.set_device(args.local_rank)
    # ============================================================== load data
    train_data = FeatureDataset(args, mode='train')
    val_data = FeatureDataset(args, mode='val')
    # train_sampler = DistributedSampler(train_data)
    # val_sampler = DistributedSampler(val_data)
    train_loader = DataLoaderX(train_data,
                               batch_size=args.bs,
                               # sampler=train_sampler,
                               shuffle=True,
                               collate_fn=train_data.pad_collate,
                               num_workers=args.n_works,
                               pin_memory=True,
                               drop_last=True)
    val_loader = DataLoader(val_data,
                            batch_size=args.bs,
                            # sampler=val_sampler,
                            shuffle=False,
                            collate_fn=val_data.pad_collate,
                            num_workers=args.n_works,
                            pin_memory=True,
                            drop_last=False)

    # ================================================================== model
    model_config.num_head = args.num_head
    args.model_config = model_config

    model = H2GR(model_config)
    num_model_parameter = count_parameters(model)
    args.num_model_parameter = {'all': num_model_parameter[0],
                                'trainable': num_model_parameter[1]}
    print(100 * '-')

    model = model.cuda(device=model_config.gpu_ids[0])
    if len(model_config.gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=model_config.gpu_ids)
    # model = nn.parallel.DistributedDataParallel(model,
    #                                             device_ids=[args.local_rank])

    print(args)
    Train(args, model, [train_loader, val_loader])


if __name__ == '__main__':
    main()
