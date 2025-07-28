import os
from abc import abstractmethod

import time
import torch
import torch.distributed as dist
import pandas as pd
import numpy as np
from numpy import inf
from .metrics_clinical import CheXbertMetrics
import copy
from .optims import LinearWarmupCosineLRScheduler
from .losses import FocalLoss
import torch.nn.functional as F
import json
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

pad_observation_labels = [
    'enlarged cardiomediastinum',
    'cardiomegaly',
    'lung opacity',
    'lung lesion',
    'edema',
    'consolidation',
    'pneumonia',
    'atelectasis',
    'pneumothorax',
    'pleural effusion',
    'pleural other',
    'fracture',
    'support devices',
    'no finding',
]
class BaseTrainer(object):
    def __init__(self, model, criterion_cls, base_probs, metric_ftns, args, device, is_main_process):
        self.args = args
        self.model = model
        self.device = device
        self.is_main_process = is_main_process

        self.chexbert_metrics = CheXbertMetrics('./checkpoints/stanford/chexbert/chexbert.pth', args.batch_size, device)

        self.criterion_cls = criterion_cls

        self.base_probs = base_probs
        self.metric_ftns = metric_ftns
        #################
        self.optimizer = None
        num_parameters = 0
        p_wd, p_non_wd = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue  # frozen weights
            if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
                p_non_wd.append(p)
            else:
                p_wd.append(p)
            num_parameters += p.data.nelement()
        print("number of trainable parameters: {}".format(num_parameters))
        optim_params = [
            {
                "params": p_wd,
                "weight_decay": float(self.args.weight_decay),
            },
            {"params": p_non_wd, "weight_decay": 0},
        ]
        beta2 = 0.999
        self.optimizer = torch.optim.AdamW(
            optim_params,
            lr=float(self.args.init_lr),
            weight_decay=float(self.args.weight_decay),
            betas=(0.9, beta2),
        )

        # ve_params = list(map(id, model.module.visual_encoder.parameters()))
        # ed_params = filter(lambda x: id(x) not in ve_params, model.module.parameters())
        # self.optimizer = torch.optim.AdamW(
        #     [{'params': model.module.visual_encoder.parameters(), 'lr': float(self.args.lr_ve)},
        #     {'params': ed_params, 'lr': float(self.args.lr_ed)}],
        #     # lr=float(self.args.init_lr),
        #     weight_decay=float(self.args.weight_decay),
        #     betas=(0.9, beta2),
        # )

        #################

        self.epochs = self.args.epochs

        self.mnt_metric = 'val_' + args.monitor_metric

        self.mnt_best = 0 
        self.log_best = {}

        self.start_epoch = self.args.start_epoch
        self.checkpoint_dir = args.save_dir

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.mode = args.mode

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            if self.args.distributed:
                # for different shuffling
                self.train_dataloader.sampler.set_epoch(epoch)
            if self.mode == 'pretrain':
                result = self.train_dynamic_logit_adj(epoch)
                # result = self.train_recap_cls(epoch)
                # result = self.train_focal_cls(epoch)
                # result = self.train_side_cls(epoch)
                # save logged information 
                log = {'epoch': epoch}
                log.update(result)
                # print logged information 
                for key, value in log.items():
                    print('\t{:15s}: {}'.format(str(key), value))
                
                # if epoch < 9:  
                best_path = os.path.join(self.checkpoint_dir, 'model_3ex_lb01_4.pth')
                torch.save(self.model.module.state_dict(), best_path)
                print("Saving current best to {}".format(best_path))
                continue
                
            if self.mode == 'MIX':
                result = self.train_mix_up(epoch)
                # save logged information 
                log = {'epoch': epoch}
                log.update(result)

                # print logged information 
                for key, value in log.items():
                    print('\t{:15s}: {}'.format(str(key), value))

                # best_path = os.path.join(self.checkpoint_dir, 'model_MIX_best.pth')
                # torch.save(self.model.module.state_dict(), best_path)
                # print("Saving current best to {}".format(best_path))
                continue

            if self.mode == 'graphtrain':
                result = self.train_prompt_graph(epoch)
                # save logged information 
                log = {'epoch': epoch}
                log.update(result)

                # print logged information 
                for key, value in log.items():
                    print('\t{:15s}: {}'.format(str(key), value))
                best_path = os.path.join(self.checkpoint_dir, 'model_graph_best.pth')
                torch.save(self.model.module.state_dict(), best_path)
                print("Saving current best to {}".format(best_path))
                continue

            while count < 1:
                result = {}
                result = self.eval_blip(result, test=False, check=True)
                result = self.eval_blip(result, test=True, check=True)
                # print('successfully evaluated')
                count += 1
            
            result = self._train_epoch_blip(epoch)
            dist.barrier()
            result = self.eval_blip(result, test=False)

            # result = self.test_time_training(epoch)
            # dist.barrier()

            # save logged information 
            log = {'epoch': epoch}
            log.update(result)

            # record best
            if self.is_main_process:
                if log[self.mnt_metric] >= self.mnt_best:
                    best_path = os.path.join(self.checkpoint_dir, f'model_best.pth')
                    torch.save(self.model.module.state_dict(), best_path)
                    print("Saving current best to {}".format(best_path))
                    result = self.eval_blip(result, test=True)
                    log.update(result)
                    self.mnt_best = log[self.mnt_metric]
                    self.log_best = copy.deepcopy(log)

            # print logged information 
            for key, value in log.items():
                print('\t{:15s}: {}'.format(str(key), value))

        if self.is_main_process:
            print('Best results w.r.t {}:'.format(self.mnt_metric))
            for key, value in self.log_best.items():
                print('\t{:15s}: {}'.format(str(key), value))

class Trainer(BaseTrainer):
    def __init__(self, model, criterion_cls, base_probs, metric_ftns, args, train_dataloader, val_dataloader, test_dataloader, device, is_main_process):
        super(Trainer, self).__init__(model, criterion_cls, base_probs, metric_ftns, args, device, is_main_process)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.lr_scheduler = LinearWarmupCosineLRScheduler(
            self.optimizer, 
            self.args.epochs, 
            self.args.min_lr, 
            self.args.init_lr, 
            decay_rate=None, 
            warmup_start_lr=self.args.warmup_lr,
            warmup_steps=self.args.warmup_steps,
        )

        # different lr for visual_encoder and text_decoder
        

    def _train_epoch_blip(self, epoch):

        # Function to compute and print gradient norms
        def compute_gradient_norms(model):
            grad_norms = {}
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    grad_norms[name] = grad_norm
                    print(f'Layer: {name}, Gradient norm: {grad_norm}')
            return grad_norms

        train_loss = 0
        self.model.train()
        logits = {'bla': [], 'pos': [], 'neg': [], 'unc': []}
        counts = {'bla': [], 'pos': [], 'neg': [], 'unc': []}
        for batch_idx, (ids, images, captions, cls_labels) in enumerate(self.train_dataloader):
            images = images.to(self.device)
            cls_labels = cls_labels.to(self.device)
            # pred_logits = pred_logits.to(self.device)
            clip_memory = None
            self.lr_scheduler.step(cur_epoch=epoch, cur_step=batch_idx)
            loss_lm, loss_cls = self.model(images, captions, cls_labels, clip_memory, self.criterion_cls, mode=self.mode, base_probs=self.base_probs)
            # print(weights)
            loss = loss_lm + self.args.cls_weight*loss_cls
            if batch_idx%10 == 0:
                print("{}/{} loss: {}, loss_lm: {}, loss_cls: {}".format(batch_idx, len(self.train_dataloader), loss.item(), loss_lm.item(), self.args.cls_weight*loss_cls.item()))
                # print("{}/{} loss: {}".format(batch_idx, len(self.train_dataloader), loss.item()))
            train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()
            self.optimizer.zero_grad()

        #     with torch.no_grad():               
        #         cls_preds_logits_1, cls_preds_logits_2, cls_preds_logits_3 = cls_preds_logits

        #         cls_labels_0 = (cls_labels==0).float()
        #         logit_0 = cls_preds_logits_2[:, 0, :]*cls_labels_0
        #         logits['bla'].append(logit_0.cpu().numpy())
        #         counts['bla'].append(cls_labels_0.cpu().numpy())

        #         cls_labels_1 = (cls_labels==1).float()
        #         logit_1 = cls_preds_logits_2[:, 1, :]*cls_labels_1
        #         logits['pos'].append(logit_1.cpu().numpy())
        #         counts['pos'].append(cls_labels_1.cpu().numpy())

        #         cls_labels_2 = (cls_labels==2).float()
        #         logit_2 = cls_preds_logits_3[:, 2, :]*cls_labels_2
        #         logits['neg'].append(logit_2.cpu().numpy())
        #         counts['neg'].append(cls_labels_2.cpu().numpy())

        #         cls_labels_3 = (cls_labels==3).float()
        #         logit_3 = cls_preds_logits_1[:, 3, :]*cls_labels_3
        #         logits['unc'].append(logit_3.cpu().numpy())
        #         counts['unc'].append(cls_labels_3.cpu().numpy())

        # logit_b = np.concatenate(logits['bla'], axis=0)
        # count_b = np.concatenate(counts['bla'], axis=0)
        # logit_b = np.sum(logit_b, axis=0)
        # count_b = np.sum(count_b, axis=0)
        # # nan to 1
        # count_b[count_b==0] = 1.0
        # logit_b = logit_b/count_b

        # logit_p = np.concatenate(logits['pos'], axis=0)
        # count_p = np.concatenate(counts['pos'], axis=0)
        # logit_p = np.sum(logit_p, axis=0)
        # count_p = np.sum(count_p, axis=0)
        # # nan to 1
        # count_p[count_p==0] = 1.0
        # logit_p = logit_p/count_p

        # logit_n = np.concatenate(logits['neg'], axis=0)
        # count_n = np.concatenate(counts['neg'], axis=0)
        # logit_n = np.sum(logit_n, axis=0)
        # count_n = np.sum(count_n, axis=0)
        # # nan to 1
        # count_n[count_n==0] = 1.0
        # logit_n = logit_n/count_n

        # logit_p2 = np.concatenate(logits['unc'], axis=0)
        # count_p2 = np.concatenate(counts['unc'], axis=0)
        # logit_p2 = np.sum(logit_p2, axis=0)
        # count_p2 = np.sum(count_p2, axis=0)
        # # nan to 1
        # count_p2[count_p2==0] = 1.0
        # logit_p2 = logit_p2/count_p2

        # ratio = 0
        # self.base_probs[0] = logit_b
        # self.base_probs[1] = logit_p
        # self.base_probs[2] = logit_n
        # self.base_probs[3] = logit_p2

        # self.base_probs[2, -1] = 1.0
        # self.base_probs[3, -1] = 1.0
        # print(self.base_probs)
        # grad_norms = compute_gradient_norms(self.model)
        log = {'train_loss': train_loss / len(self.train_dataloader)}

        return log

    def eval_blip(self, log, test=True, check=False):
        self.model.module.eval()

        if test is not True:
            with torch.no_grad():
                val_gts, val_res = [], []
                val_prompt_gts, val_prompt_res = [], []
                for batch_idx, (ids, images, captions, cls_labels) in enumerate(self.val_dataloader):
                    images = images.to(self.device) 
                    cls_labels = cls_labels.to(self.device)
                    # pred_logits = pred_logits.to(self.device)
                    # cls_labels[cls_labels==3] = 1
                    # view_ids = view_ids.to(self.device)
                    clip_memory = None
                    # clip_memory = clip_memory.to(self.device)
                    ground_truths = captions
                    reports, cls_preds = self.model.module.generate(images, cls_labels, clip_memory, sample=False, num_beams=self.args.beam_size, 
                                                                    max_length=self.args.gen_max_len, min_length=self.args.gen_min_len, mode=self.mode)
                    
                    if batch_idx%10 == 0:
                        print(f'val: {batch_idx}/{len(self.val_dataloader)}')
                    val_res.extend(reports)
                    val_gts.extend(ground_truths)

                    val_prompt_gts.extend(cls_labels.cpu().numpy().tolist())
                    val_prompt_res.extend(cls_preds)

                    if check is True:
                        print('check val successfully')
                        break

                val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                        {i: [re] for i, re in enumerate(val_res)})
                val_ce = self.chexbert_metrics.compute(val_gts, val_res)
                val_prompt_ce = self.chexbert_metrics.compute_label(val_prompt_gts, val_prompt_res)
                log.update(**{'val_' + k: v for k, v in val_met.items()})
                log.update(**{'val_' + k: v for k, v in val_ce.items()})
                log.update(**{'val_prompt_' + k: v for k, v in val_prompt_ce.items()})
            return log
        
        elif test is True:
            with torch.no_grad():
                test_gts, test_res = [], []
                test_prompt_gts, test_prompt_res = [], []
                logits_level_all = []
                cls_labels_all, cls_preds_all = [], []
                for batch_idx, (ids, images, captions, cls_labels) in enumerate(self.test_dataloader):
                    images = images.to(self.device) 
                    clip_memory = None
                    cls_labels = cls_labels.to(self.device)
                    # cls_preds_fd = cls_preds_fd.to(self.device)
                    # pred_logits = pred_logits.to(self.device)
                    # cls_labels[cls_labels==3] = 1
                    ground_truths = captions
                    reports, cls_preds = self.model.module.generate(images, cls_labels, clip_memory, cls_preds_fd=None, sample=False, num_beams=self.args.beam_size, 
                                                                    max_length=self.args.gen_max_len, min_length=self.args.gen_min_len, mode=self.mode)

                    if batch_idx%10 == 0:
                        print(f'val: {batch_idx}/{len(self.test_dataloader)}')

                    test_res.extend(reports)
                    test_gts.extend(ground_truths)
                    # logits_level_all.extend(logits_level.cpu().numpy().tolist())
                    # cls_labels_all.extend(cls_labels.cpu().numpy().tolist())
                    # cls_preds_all.extend(cls_preds)

                    test_prompt_gts.extend(cls_labels.cpu().numpy().tolist())
                    test_prompt_res.extend(cls_preds)

                    if check is True:
                        print('check test successfully')
                        break

                test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                            {i: [re] for i, re in enumerate(test_res)})
                test_ce = self.chexbert_metrics.compute(test_gts, test_res)

                test_prompt_ce = self.chexbert_metrics.compute_label(test_prompt_gts, test_prompt_res)
                try:
                    test_res, test_gts = pd.DataFrame(test_res), pd.DataFrame(test_gts)
                    test_gts.to_csv(os.path.join(self.args.save_dir, "gts.csv"), index=False, header=False)
                    test_res.to_csv(os.path.join(self.args.save_dir, "res.csv"), index=False, header=False)
                except:
                    print('Error in saving the reports')
                log.update(**{'test_' + k: v for k, v in test_met.items()})
                log.update(**{'test_' + k: v for k, v in test_ce.items()})
                log.update(**{'test_prompt_' + k: v for k, v in test_prompt_ce.items()})
            return log

    def train_prompt(self, epoch):

        self.model.train()
        train_loss = 0
        for batch_idx, (images, captions, cls_labels, clip_memory) in enumerate(self.train_dataloader):
            images = images.to(self.device)
            cls_labels = cls_labels.to(self.device)
            # cls_labels = cls_labels[:, :-1]
            clip_memory = clip_memory.to(self.device)
            self.lr_scheduler.step(cur_epoch=epoch, cur_step=batch_idx)
            loss_cls, _, _ = self.model(images, captions, cls_labels, clip_memory, self.criterion_cls, mode='pretrain')
            loss =  self.args.cls_weight*loss_cls
            if batch_idx%1000 == 0:
                print("{}/{} loss: {}".format(batch_idx, len(self.train_dataloader), loss.item()))
            train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()
            self.optimizer.zero_grad()
        log = {'train_loss': train_loss / len(self.train_dataloader)}

        self.model.eval()
        val_prompt_gts, val_prompt_res = {0:[], 1:[], 2:[]}, {0:[], 1:[], 2:[]}
        val_f1_score = {0:[], 1:[], 2:[]}
        with torch.no_grad():
            for batch_idx, (ids, images, captions, cls_labels, clip_memory) in enumerate(self.val_dataloader):
                images = images.to(self.device) 
                cls_labels = cls_labels.to(self.device)
                # cls_labels = cls_labels[:, :-1]
                clip_memory = clip_memory.to(self.device)
                _, cls_preds, _ = self.model(images, captions, cls_labels, clip_memory, self.criterion_cls, mode='pretrain')


                for i, cls_preds_i in enumerate(cls_preds):
                
                    val_prompt_gts[i].extend(cls_labels.cpu().numpy().tolist())
                    val_prompt_res[i].extend(cls_preds_i.cpu().numpy().tolist())
            for i in range(3):
                val_prompt_ce = self.chexbert_metrics.compute_label(val_prompt_gts[i], val_prompt_res[i])
                log.update(**{f'val_{i}_' + k: v for k, v in val_prompt_ce.items()})

                # val_f1_score[i] = val_prompt_ce['f1_score']
                #
            # val_prompt_ce = self.chexbert_metrics.compute_label(val_prompt_gts, val_prompt_res)
            # log.update(**{f'val_{i}_' + k: v for k, v in val_prompt_ce.items()})

        self.model.eval()
        test_prompt_gts, test_prompt_res = {0:[], 1:[], 2:[]}, {0:[], 1:[], 2:[]}
        with torch.no_grad():
            for batch_idx, (ids, images, captions, cls_labels, clip_memory) in enumerate(self.test_dataloader):
                images = images.to(self.device) 
                cls_labels = cls_labels.to(self.device)
                # cls_labels = cls_labels[:, :-1]
                clip_memory = clip_memory.to(self.device)
                _, cls_preds, _ = self.model(images, captions, cls_labels, clip_memory, self.criterion_cls, mode='pretrain')

                for i, cls_preds_i in enumerate(cls_preds):
                    test_prompt_gts[i].extend(cls_labels.cpu().numpy().tolist())
                    test_prompt_res[i].extend(cls_preds_i.cpu().numpy().tolist())
                # test_prompt_gts.extend(cls_labels.cpu().numpy().tolist())
                # test_prompt_res.extend(cls_preds.cpu().numpy().tolist())
            for i in range(3):
                test_prompt_ce = self.chexbert_metrics.compute_label(test_prompt_gts[i], test_prompt_res[i])
                log.update(**{f'test_{i}_' + k: v for k, v in test_prompt_ce.items()})
        # test_prompt_ce = self.chexbert_metrics.compute_label(test_prompt_gts, test_prompt_res)
        # log.update(**{'test_' + k: v for k, v in test_prompt_ce.items()})

        return log

    def train_prompt_graph(self, epoch):
        
        self.model.train()
        train_loss = 0
        for batch_idx, (images, captions, cls_labels) in enumerate(self.train_dataloader):
            images = images.to(self.device)
            cls_labels = cls_labels.to(self.device)
            clip_memory =None
            self.lr_scheduler.step(cur_epoch=epoch, cur_step=batch_idx)
            loss_cls, _ = self.model(images, captions, cls_labels, clip_memory, self.criterion_cls, mode='graphtrain')
            loss =  self.args.cls_weight*loss_cls
            if batch_idx%10 == 0:
                print("{}/{} loss: {}".format(batch_idx, len(self.train_dataloader), loss.item()))
            train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()
            self.optimizer.zero_grad()
        log = {'train_loss': train_loss / len(self.train_dataloader)}

        self.model.eval()
        val_prompt_gts, val_prompt_res = [], []
        val_acc = 0
        with torch.no_grad():
            for batch_idx, (images, captions, cls_labels) in enumerate(self.val_dataloader):
                images = images.to(self.device) 
                cls_labels = cls_labels.to(self.device)
                clip_memory = None
                _, cls_preds = self.model(images, captions, cls_labels, clip_memory, self.criterion_cls, mode='graphtrain')
                # val_acc += (cls_preds_s1 == cls_labels_s1).sum().item()

            # val_acc = val_acc / len(self.val_dataloader.dataset)
            # log.update({'val_acc': val_acc})
                val_prompt_gts.extend(cls_labels.cpu().numpy().tolist())
                val_prompt_res.extend(cls_preds)

            val_prompt_ce = self.chexbert_metrics.compute_label(val_prompt_gts, val_prompt_res)
            log.update(**{'val_' + k: v for k, v in val_prompt_ce.items()})

        self.model.eval()
        test_prompt_gts, test_prompt_res = [], []
        test_acc = 0
        for batch_idx, (images, captions, cls_labels) in enumerate(self.test_dataloader):
            images = images.to(self.device) 
            cls_labels = cls_labels.to(self.device)
            clip_memory = None
            _, cls_preds = self.model(images, captions, cls_labels, clip_memory, self.criterion_cls, mode='graphtrain')

            test_prompt_gts.extend(cls_labels.cpu().numpy().tolist())
            test_prompt_res.extend(cls_preds)

            # test_acc += (cls_preds_step1 == cls_labels_s1).sum().item()

        # test_acc = test_acc / len(self.test_dataloader.dataset)
        # log.update({'test_acc': test_acc})
        
        test_prompt_ce = self.chexbert_metrics.compute_label(test_prompt_gts, test_prompt_res)
        log.update(**{'test_' + k: v for k, v in test_prompt_ce.items()})

        return log

    def test_time_training(self, epoch):
        # aggregation_weight = torch.nn.Parameter(torch.FloatTensor(3), requires_grad=True)
        # aggregation_weight.data.fill_(1/3)
        # cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
        # # L1 distance for two tensor with shape [bs, 3, 14]
        # l1 = torch.nn.L1Loss()


        # aggregation_weight = aggregation_weight.view(3, 13)
        train_loss = 0
        self.model.train()
        for batch_idx, (images, captions, cls_labels, clip_memory) in enumerate(self.train_dataloader):
            self.lr_scheduler.step(cur_epoch=epoch, cur_step=batch_idx)
            images = images.to(self.device)
            clip_memory = clip_memory.to(self.device)
            cls_labels = cls_labels.to(self.device)
            ground_truths = captions
            loss_cls, cls_preds, weights = self.model(images, captions, cls_labels, clip_memory, self.criterion_cls, mode='second')

            loss = self.args.cls_weight*loss_cls
            if batch_idx%100 == 0:
                print("{}/{} loss: {}".format(batch_idx, len(self.train_dataloader), loss.item()))
            train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()
            self.optimizer.zero_grad()
        log = {'train_loss': train_loss / len(self.train_dataloader), 'weights': weights}

        self.model.eval()
        val_prompt_gts, val_prompt_res = [], []
        with torch.no_grad():
            for batch_idx, (idx, images, captions, cls_labels, clip_memory) in enumerate(self.val_dataloader):
                images = images.to(self.device) 
                cls_labels = cls_labels.to(self.device)
                clip_memory = clip_memory.to(self.device)
                ground_truths = captions
                _, cls_preds, weights = self.model(images, captions, cls_labels, clip_memory, self.criterion_cls, mode='second')

                if batch_idx%10 == 0:
                    print(f'val: {batch_idx}/{len(self.val_dataloader)}')
                val_prompt_gts.extend(cls_labels.cpu().numpy().tolist())
                val_prompt_res.extend(cls_preds)
            val_prompt_ce = self.chexbert_metrics.compute_label(val_prompt_gts, val_prompt_res)
            log.update(**{'val_' + k: v for k, v in val_prompt_ce.items()})

        self.model.eval()
        test_prompt_gts, test_prompt_res = [], []
        test_idx = []
        with torch.no_grad():
            for batch_idx, (idx, images, captions, cls_labels, clip_memory) in enumerate(self.test_dataloader):
                images = images.to(self.device) 
                cls_labels = cls_labels.to(self.device)
                clip_memory = clip_memory.to(self.device)
                ground_truths = captions
                _, cls_preds, weights = self.model(images, captions, cls_labels, clip_memory, self.criterion_cls, mode='second')

                if batch_idx%10 == 0:
                    print(f'test: {batch_idx}/{len(self.test_dataloader)}')
                test_prompt_gts.extend(cls_labels.cpu().numpy().tolist())
                test_prompt_res.extend(cls_preds)
                test_idx.extend(idx)
            test_prompt_ce = self.chexbert_metrics.compute_label(test_prompt_gts, test_prompt_res)
            log.update(**{'test_' + k: v for k, v in test_prompt_ce.items()})

        # save test_idx, test_prompt_res as json
        test_cls_json = {}
        for idx, cls_preds in zip(test_idx, test_prompt_res):
            test_cls_json[idx] = cls_preds
        
        with open (os.path.join(self.args.save_dir, 'test_cls_preds_3_14.json'), 'w') as f:
            json.dump(test_cls_json, f)
        
        return log
            
    def train_dynamic_logit_adj(self, epoch):
        log = {}
        self.model.train()
        train_loss = 0
        train_sim_loss = 0
        bla_loss, pos_loss, neg_loss, unc_loss = 0, 0, 0, 0
        logits = {'bla': [], 'pos': [], 'neg': [], 'unc': []}
        counts = {'bla': [], 'pos': [], 'neg': [], 'unc': []}

        # logits = {'0': {'bla': [], 'pos': [], 'neg': [], 'unc': []}, '1': {'bla': [], 'pos': [], 'neg': [], 'unc': []}, '2': {'bla': [], 'pos': [], 'neg': [], 'unc': []}, '3': {'bla': [], 'pos': [], 'neg': [], 'unc': []}}
        # counts = {'0': {'bla': [], 'pos': [], 'neg': [], 'unc': []}, '1': {'bla': [], 'pos': [], 'neg': [], 'unc': []}, '2': {'bla': [], 'pos': [], 'neg': [], 'unc': []}, '3': {'bla': [], 'pos': [], 'neg': [], 'unc': []}}

        for batch_idx, (ids, images, captions, cls_labels) in enumerate(self.train_dataloader):
            self.lr_scheduler.step(cur_epoch=epoch, cur_step=batch_idx)
            images = images.to(self.device)
            cls_labels = cls_labels.to(self.device)
            # cls_labels[cls_labels == 3] = 2
            # cls_labels[cls_labels == 3] = 0
            # cls_labels = cls_labels[:, :-1]
            # clip_memory = clip_memory.to(self.device)
            clip_memory = None
            loss_cls, loss_all, loss_base,  _ , cls_preds_logits = self.model(images, captions, cls_labels, clip_memory, self.criterion_cls, base_probs=self.base_probs, mode='pretrain')
            # if epoch < 10:
            #     loss = loss_base
            # else:
            #     # stop gradient of visual_encoder 
            #     self.model.module.visual_encoder.requires_grad_(False)
            #     self.model.module.cls_head.requires_grad_(False)
            loss = loss_cls
            if batch_idx%100 == 0:
                print("{}/{} loss: {}".format(batch_idx, len(self.train_dataloader), loss.item()))
            train_loss += loss.item()
            # train_sim_loss += loss_sim.item()
            # bla_loss += loss_all[3]
            pos_loss += loss_all[1]
            neg_loss += loss_all[2]
            unc_loss += loss_all[0]
            # bla_loss += loss_all[3]
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()
            self.optimizer.zero_grad()
            with torch.no_grad():
 
                cls_preds_logits_1, cls_preds_logits_2, cls_preds_logits_3 = cls_preds_logits

                cls_labels_1 = (cls_labels==1).float()
                logit_1 = cls_preds_logits_2[:, 1, :]*cls_labels_1
                logits['pos'].append(logit_1.cpu().numpy())
                counts['pos'].append(cls_labels_1.cpu().numpy())

                cls_labels_2 = (cls_labels==2).float()
                logit_2 = cls_preds_logits_3[:, 2, :]*cls_labels_2
                logits['neg'].append(logit_2.cpu().numpy())
                counts['neg'].append(cls_labels_2.cpu().numpy())

                cls_labels_3 = (cls_labels==3).float()
                logit_3 = cls_preds_logits_1[:, 3, :]*cls_labels_3
                logits['unc'].append(logit_3.cpu().numpy())
                counts['unc'].append(cls_labels_3.cpu().numpy())

                # cls_labels_0 = (cls_labels==0).float()
                # logit_0 = cls_preds_logits_4[:, 0, :]*cls_labels_0
                # logits['bla'].append(logit_0.cpu().numpy())
                # counts['bla'].append(cls_labels_0.cpu().numpy())

        # logit_b = np.concatenate(logits['bla'], axis=0)
        # count_b = np.concatenate(counts['bla'], axis=0)
        # logit_b = np.sum(logit_b, axis=0)
        # count_b = np.sum(count_b, axis=0)
        # # nan to 1
        # count_b[count_b==0] = 1.0
        # logit_b = logit_b/count_b

        logit_p = np.concatenate(logits['pos'], axis=0)
        count_p = np.concatenate(counts['pos'], axis=0)
        logit_p = np.sum(logit_p, axis=0)
        count_p = np.sum(count_p, axis=0)
        # nan to 1
        count_p[count_p==0] = 1.0
        logit_p = logit_p/count_p

        logit_n = np.concatenate(logits['neg'], axis=0)
        count_n = np.concatenate(counts['neg'], axis=0)
        
        logit_n = np.sum(logit_n, axis=0)
        count_n = np.sum(count_n, axis=0)
        # nan to 1
        count_n[count_n==0] = 1.0
        logit_n = logit_n/count_n

        logit_p2 = np.concatenate(logits['unc'], axis=0)
        count_p2 = np.concatenate(counts['unc'], axis=0)
        logit_p2 = np.sum(logit_p2, axis=0)
        count_p2 = np.sum(count_p2, axis=0)
        # nan to 1
        count_p2[count_p2==0] = 1.0
        logit_p2 = logit_p2/count_p2

        ratio = 0.0
        # self.base_probs[0] = ratio * self.base_probs[0] + (1-ratio) * logit_b
        self.base_probs[1] = ratio * self.base_probs[1] + (1-ratio) * logit_p
        self.base_probs[2] = ratio * self.base_probs[2] + (1-ratio) * logit_n
        self.base_probs[3] = ratio * self.base_probs[3] + (1-ratio) * logit_p2
        # self.base_probs[0] = ratio * self.base_probs[0] + (1-ratio) * logit_b

        # self.base_probs /= self.base_probs.sum(axis=1)
        # self.base_probs[:, -1] = 1.0
        self.base_probs[2, -1] = 1.0
        self.base_probs[3, -1] = 1.0
        print(self.base_probs)
        # logit /= np.max(logit)
        # print(f'{key}: {logit}')
        # self.base_probs[k] = logit  

        log = {'train_loss': train_loss / len(self.train_dataloader), 'pos_loss': pos_loss / len(self.train_dataloader), 'neg_loss': neg_loss / len(self.train_dataloader), 'unc_loss': unc_loss / len(self.train_dataloader)}

        self.model.eval()
        test_prompt_gts = [] 
        test_prompt_res = {0:[], 1:[], 2:[], 3:[], 4:[]}
        test_cls_preds_logits = {0:[], 1:[], 2:[], 3:[]}
        test_idx = []
        with torch.no_grad():
            for batch_idx, (ids, images, captions, cls_labels, cls_preds_fd) in enumerate(self.test_dataloader):
                
                images = images.to(self.device) 
                cls_labels = cls_labels.to(self.device)
                # cls_labels = cls_labels[:, :-1]
                # clip_memory = clip_memory.to(self.device)
                clip_memory = None
                _, _, _, cls_preds, cls_preds_logits = self.model(images, captions, cls_labels, clip_memory, self.criterion_cls, mode='pretrain')
                # cls_labels[cls_labels == 2] = 0
                # cls_labels[cls_labels == 3] = 0

                test_prompt_gts.extend(cls_labels.cpu().numpy().tolist())
                test_idx.extend(ids)
                for i, cls_preds_i in enumerate(cls_preds):
                    test_prompt_res[i].extend(cls_preds_i.cpu().numpy().tolist())

                # for i, cls_preds_logit_i in enumerate(cls_preds_logits):
                #     test_cls_preds_logits[i].extend(cls_preds_logit_i.cpu().numpy().tolist())
                # test_cls_preds_logits[3].extend(cls_labels.cpu().numpy().tolist())

                # test_prompt_gts.extend(cls_labels.cpu().numpy().tolist())
                # test_prompt_res.extend(cls_preds.cpu().numpy().tolist())

            test_prompt_ce_bla = self.chexbert_metrics.compute_label(test_prompt_gts, test_prompt_res[3], k=0)
            test_prompt_ce_unc = self.chexbert_metrics.compute_label(test_prompt_gts, test_prompt_res[3], k=3)
            test_prompt_ce_pos = self.chexbert_metrics.compute_label(test_prompt_gts, test_prompt_res[3], k=1)
            test_prompt_ce_neg = self.chexbert_metrics.compute_label(test_prompt_gts, test_prompt_res[3], k=2)
            test_prompt_ce = self.chexbert_metrics.compute_label(test_prompt_gts, test_prompt_res[3])
            log.update(**{f'test_bla_' + k: v for k, v in test_prompt_ce_bla.items()})
            log.update(**{f'test_unc_' + k: v for k, v in test_prompt_ce_unc.items()})
            log.update(**{f'test_pos_' + k: v for k, v in test_prompt_ce_pos.items()})
            log.update(**{f'test_neg_' + k: v for k, v in test_prompt_ce_neg.items()})
            log.update(**{f'test_' + k: v for k, v in test_prompt_ce.items()})
        # test_prompt_ce = self.chexbert_metrics.compute_label(test_prompt_gts, test_prompt_res)
        # log.update(**{'test_' + k: v for k, v in test_prompt_ce.items()})

        # save test_idx, test_prompt_res as json
        test_cls_json = {}
        for idx, preds in zip(test_idx, test_prompt_res[3]):
            test_cls_json[idx] = preds
        
        with open (os.path.join(self.args.save_dir, 'test_3ex_lb01_4.json'), 'w') as f:
            json.dump(test_cls_json, f)

        # # # save test_cls_preds_logits as json
        # with open (os.path.join(self.args.save_dir, 'test_cls_preds_3experts_unc_pos_neg_momentum01_update2_logits.json'), 'w') as f:
        #     json.dump(test_cls_preds_logits, f)

        return log

    def test_resize(self):
        log = {}
        self.model.eval()
        test_prompt_res = {0:[], 1:[], 2:[], 3:[]}
        test_prompt_gts = [] 
        test_idx = []
        with torch.no_grad():
            for batch_idx, (ids, images, captions, cls_labels) in enumerate(self.test_dataloader):
                
                images = images.to(self.device) 
                cls_labels = cls_labels.to(self.device)
                clip_memory = None
                _,  cls_preds, cls_preds_logits = self.model(images, captions, cls_labels, clip_memory, self.criterion_cls, mode='pretrain')
                cls_labels[cls_labels == 3] = 1    
                test_prompt_gts.extend(cls_labels.cpu().numpy().tolist())
                test_idx.extend(ids)
                for i, cls_preds_i in enumerate(cls_preds):
                    test_prompt_res[i].extend(cls_preds_i.cpu().numpy().tolist())

                # for i, cls_preds_logit_i in enumerate(cls_preds_logits):
                #     test_cls_preds_logits[i].extend(cls_preds_logit_i.cpu().numpy().tolist())
                # test_cls_preds_logits[3].extend(cls_labels.cpu().numpy().tolist())

                # test_prompt_gts.extend(cls_labels.cpu().numpy().tolist())
                # test_prompt_res.extend(cls_preds.cpu().numpy().tolist())
            # for i in range(3):
            test_prompt_ce_o = self.chexbert_metrics.compute_label(test_prompt_gts, test_prompt_res[3], k=0)
            test_prompt_ce_pos = self.chexbert_metrics.compute_label(test_prompt_gts, test_prompt_res[3], k=1)
            test_prompt_ce_neg = self.chexbert_metrics.compute_label(test_prompt_gts, test_prompt_res[3], k=2)
            test_prompt_ce = self.chexbert_metrics.compute_label(test_prompt_gts, test_prompt_res[3])
            log.update(**{f'test_bla_' + k: v for k, v in test_prompt_ce_o.items()})
            log.update(**{f'test_pos_' + k: v for k, v in test_prompt_ce_pos.items()})
            log.update(**{f'test_neg_' + k: v for k, v in test_prompt_ce_neg.items()})
            log.update(**{f'test_' + k: v for k, v in test_prompt_ce.items()})
        # test_prompt_ce = self.chexbert_metrics.compute_label(test_prompt_gts, test_prompt_res)
        # log.update(**{'test_' + k: v for k, v in test_prompt_ce.items()})

        # save test_idx, test_prompt_res as json
        test_cls_json = {}
        for idx, preds in zip(test_idx, test_prompt_res[3]):
            test_cls_json[idx] = preds

        for k, v in log.items():
            print(f'{k}: {v}')
        
        with open (os.path.join(self.args.save_dir, 'test_224_to_448.json'), 'w') as f:
            json.dump(test_cls_json, f)

    

    def train_mix_up(self, epoch):

        self.model.train()
        train_loss = 0
        for batch_idx, (images, captions, cls_labels, clip_memory) in enumerate(self.train_dataloader):
            images = images.to(self.device)
            cls_labels = cls_labels.to(self.device)
            # cls_labels = cls_labels[:, :-1]
            clip_memory = clip_memory.to(self.device)
            self.lr_scheduler.step(cur_epoch=epoch, cur_step=batch_idx)
            loss_cls, _ = self.model.module.mix_up_cls(images, captions, cls_labels, clip_memory, self.criterion_cls)
            loss =  self.args.cls_weight*loss_cls
            if batch_idx%10 == 0:
                print("{}/{} loss: {}".format(batch_idx, len(self.train_dataloader), loss.item()))
            train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()
            self.optimizer.zero_grad()
        log = {'train_loss': train_loss / len(self.train_dataloader)}

        self.model.eval()
        val_prompt_gts, val_prompt_res = [], []
        with torch.no_grad():
            for batch_idx, (images, captions, cls_labels, clip_memory) in enumerate(self.val_dataloader):
                images = images.to(self.device) 
                cls_labels = cls_labels.to(self.device)
                # cls_labels = cls_labels[:, :-1]
                clip_memory = clip_memory.to(self.device)
                _, cls_preds = self.model.module.mix_up_cls(images, captions, cls_labels, clip_memory, self.criterion_cls, mode='test')

                val_prompt_gts.extend(cls_labels.cpu().numpy().tolist())
                val_prompt_res.extend(cls_preds)
            
            val_prompt_ce = self.chexbert_metrics.compute_label(val_prompt_gts, val_prompt_res)
            log.update(**{f'val_' + k: v for k, v in val_prompt_ce.items()})


        self.model.eval()
        test_prompt_gts, test_prompt_res = [], []
        with torch.no_grad():
            for batch_idx, (images, captions, cls_labels, clip_memory) in enumerate(self.test_dataloader):
                images = images.to(self.device) 
                cls_labels = cls_labels.to(self.device)
                # cls_labels = cls_labels[:, :-1]
                clip_memory = clip_memory.to(self.device)
                _, cls_preds = self.model.module.mix_up_cls(images, captions, cls_labels, clip_memory, self.criterion_cls, mode='test')

                test_prompt_gts.extend(cls_labels.cpu().numpy().tolist())
                test_prompt_res.extend(cls_preds)
                   
        test_prompt_ce = self.chexbert_metrics.compute_label(test_prompt_gts, test_prompt_res)
        log.update(**{'test_' + k: v for k, v in test_prompt_ce.items()})

        return log

    def annotate_iu_xray(self):

        annoatate = []
        idx_all = []
        with torch.no_grad():
            train_gts = []
            for batch_idx, (ids, images, captions) in enumerate(self.train_dataloader):
                train_gts.extend(captions)
                idx_all.extend(ids)

                if batch_idx % 100 == 0:
                    print(f'train: {batch_idx}/{len(self.train_dataloader)}')

            scores, gts_annotate = self.chexbert_metrics.compute(train_gts, train_gts)
            annoatate['train'] = gts_annotate.tolist()

            val_gts = []
            for batch_idx, (ids, images, captions) in enumerate(self.val_dataloader):
                val_gts.extend(captions)

                if batch_idx % 100 == 0:
                    print(f'val: {batch_idx}/{len(self.val_dataloader)}')

            scores, gts_annotate = self.chexbert_metrics.compute(val_gts, val_gts)
            annoatate['val'] = gts_annotate.tolist()

            test_gts = []
            for batch_idx, (ids, images, captions) in enumerate(self.test_dataloader):
                test_gts.extend(captions)

                if batch_idx % 100 == 0:
                    print(f'test: {batch_idx}/{len(self.test_dataloader)}')

            scores, gts_annotate = self.chexbert_metrics.compute(test_gts, test_gts)
            annoatate['test'] = gts_annotate.tolist()

        # save the annotation
        with open(os.path.join(self.args.save_dir, 'annotate_iu_xray.json'), 'w') as f:
            json.dump(annoatate, f)

        
    def generate_test_label(self):
        self.model.eval()
        test_prompt_gts, test_prompt_res = [], []
        test_idx = []
        with torch.no_grad():
            for batch_idx, (ids, images, captions, cls_labels, _) in enumerate(self.test_dataloader):
                images = images.to(self.device) 
                cls_labels = cls_labels.to(self.device)
                clip_memory = clip_memory.to(self.device)
                # clip_memory = None
                
                cls_preds = self.model.module.predict_labels(images)

                test_prompt_gts.extend(cls_labels.cpu().numpy().tolist())
                test_prompt_res.extend(cls_preds.cpu().numpy().tolist())
                test_idx.extend(ids)

        test_prompt_ce = self.chexbert_metrics.compute_label(test_prompt_gts, test_prompt_res)
        
        for k, v in test_prompt_ce.items():
            print(f'{k}: {v}')

        # save the test label
        # save test_idx, test_prompt_res as json
        test_cls_json = {}
        for idx, preds in zip(test_idx, test_prompt_res):
            test_cls_json[idx] = preds
        with open(os.path.join(self.args.save_dir, 'iu_xray_3ex_lb01_3.json'), 'w') as f:
            json.dump(test_cls_json, f)

    def train_recap_cls(self, epoch):
        log = {}
        train_loss = 0
        self.model.train()
        for batch_idx, (ids, images, captions, cls_labels) in enumerate(self.train_dataloader):
            self.lr_scheduler.step(cur_epoch=epoch, cur_step=batch_idx)
            images = images.to(self.device)
            cls_labels = cls_labels.to(self.device)
            clip_memory = None
            loss_cls, _, _ = self.model(images, captions, cls_labels, clip_memory, self.criterion_cls, base_probs=self.base_probs, mode='recap')
            loss = loss_cls
            if batch_idx%100 == 0:
                print("{}/{} loss: {}".format(batch_idx, len(self.train_dataloader), loss.item()))
            train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()
            self.optimizer.zero_grad()

        log = {'train_loss': train_loss / len(self.train_dataloader)}

        self.model.eval()
        test_prompt_gts, test_prompt_res = [], []
        observation_cls_preds, observation_det_preds, observation_trues = [], [], []
        test_idx = []
        observation_preds = []
        with torch.no_grad():
            for batch_idx, (ids, images, captions, cls_labels, cls_preds_fd) in enumerate(self.test_dataloader):
                images = images.to(self.device) 
                cls_labels = cls_labels.to(self.device)
                cls_labels[cls_labels == 3] = 1
                clip_memory = None
                _, observation_det_logits, observation_cls_logits = self.model(images, captions, cls_labels, clip_memory, self.criterion_cls, mode='recap')
                print(observation_det_logits[0])
                print(observation_cls_logits[0])
                observation_det_pred = (observation_det_logits > 0).float().cpu().numpy()
                observation_cls_pred = (observation_cls_logits > 0).float().cpu().numpy()
                test_prompt_gts.extend(cls_labels.cpu().numpy().tolist())
                # test_prompt_res.extend(cls_preds.cpu().numpy())
                observation_cls_preds.append(observation_cls_pred)
                observation_det_preds.append(observation_det_pred)
                observation_trues.append(cls_labels.cpu().numpy())
                test_idx.extend(ids)
            observation_det_preds = np.concatenate(observation_det_preds, axis=0)
            observation_cls_preds = np.concatenate(observation_cls_preds, axis=0)
            observation_trues = np.concatenate(observation_trues, axis=0)
            num_observation = observation_cls_preds.shape[1]

            ce_scores = [0, 0, 0]
            observation_preds = []
            def get_pred(a, b):
                if a == 1 and b == 1:
                    return 1
                elif a == 1 and b == 0:
                    return 0
                else:
                    return 2

            print("--------------------------------------------------------------")
            for i in range(num_observation):
                y_cls_pred = observation_cls_preds[:, i]
                if i == num_observation - 1:
                    y_det_pred = np.ones_like(y_cls_pred)
                else:
                    y_det_pred = observation_det_preds[:, i]

                y_pred = [1 if a == 1 and b == 1 else 0 for a, b in zip(y_det_pred, y_cls_pred)]
                y_true = (observation_trues[:, i] == 1) + 0.0
                observation_preds.append(
                    [get_pred(a, b) for a, b in zip(y_det_pred, y_cls_pred)]
                )
                i_ce_score = precision_recall_fscore_support(
                    y_pred=y_pred,
                    y_true=y_true,
                    pos_label=1,
                    average="binary",
                )[:-1]
                print(
                    "%s\t\t\t Prec. %0.4f\tRec. %0.4f\tF1 %0.4f"
                    % (pad_observation_labels[i], *i_ce_score)
                )
                ce_scores = [
                    ce_scores[i] + i_ce_score[i] / len(pad_observation_labels)
                    for i in range(len(ce_scores))
                ]
                # print(f'precision: {i_ce_score[0]}, recall: {i_ce_score[1]}, f1: {i_ce_score[2]}')
            # observation_preds = np.stack(observation_preds, axis=1)
            print("--------------------------------------------------------------")
            print(
                "Abnormal CE Scores\t\t\t Prec. %0.4f\tRec. %0.4f\tF1 %0.4f"
                % (ce_scores[0], ce_scores[1], ce_scores[2])
            )

            # for i in range(num_observation):
            #     y_cls_pred = observation_cls_preds[:, i]
            #     if i == num_observation - 1:
            #         y_det_pred = np.ones_like(y_cls_pred)
            #     else:
            #         y_det_pred = observation_det_preds[:, i]

            #     y_pred = [1 if a == 1 and b == 1 else 0 for a, b in zip(y_det_pred, y_cls_pred)]
            #     observation_preds.append([get_pred(a, b) for a, b in zip(y_det_pred, y_cls_pred)])
            observation_preds = np.array(observation_preds).T
            observation_preds = observation_preds.tolist()
            
            # test_prompt_ce_bla = self.chexbert_metrics.compute_label(test_prompt_gts, observation_preds, k=0)
            # # test_prompt_ce_unc = self.chexbert_metrics.compute_label(test_prompt_gts, observation_preds, k=3)
            # test_prompt_ce_pos = self.chexbert_metrics.compute_label(test_prompt_gts, observation_preds, k=1)
            # test_prompt_ce_neg = self.chexbert_metrics.compute_label(test_prompt_gts, observation_preds, k=2)
            # test_prompt_ce = self.chexbert_metrics.compute_label(test_prompt_gts, observation_preds)
            # log.update(**{f'test_bla_' + k: v for k, v in test_prompt_ce_bla.items()})
            # # log.update(**{f'test_unc_' + k: v for k, v in test_prompt_ce_unc.items()})
            # log.update(**{f'test_pos_' + k: v for k, v in test_prompt_ce_pos.items()})
            # log.update(**{f'test_neg_' + k: v for k, v in test_prompt_ce_neg.items()})
            # log.update(**{f'test_' + k: v for k, v in test_prompt_ce.items()})

        test_cls_json = {}
        for idx, preds in zip(test_idx, observation_preds):
            test_cls_json[idx] = preds
        
        with open (os.path.join(self.args.save_dir, 'test_recap2.json'), 'w') as f:
            json.dump(test_cls_json, f)

        return log

    def train_focal_cls(self, epoch):
        self.model.train()
        train_loss = 0

        for batch_idx, (idx, images, captions, cls_labels, _) in enumerate(self.train_dataloader):
            images = images.to(self.device)
            cls_labels = cls_labels.to(self.device)
            clip_memory = None
            self.lr_scheduler.step(cur_epoch=epoch, cur_step=batch_idx)
            cls_labels[cls_labels==3] = 1
            loss_cls, _ = self.model(images, captions, cls_labels, clip_memory, self.criterion_cls, base_probs=self.base_probs, mode='focal')
            loss =  self.args.cls_weight*loss_cls
            if batch_idx%100 == 0:
                print("{}/{} loss: {}".format(batch_idx, len(self.train_dataloader), loss.item()))
            train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()
            self.optimizer.zero_grad()

        log = {'train_loss': train_loss / len(self.train_dataloader)}

        self.model.eval()
        test_prompt_gts, test_prompt_res = [], []
        test_idx = []
        with torch.no_grad():
            for batch_idx, (ids, images, captions, cls_labels, cls_preds_fd) in enumerate(self.test_dataloader):
                images = images.to(self.device) 
                cls_labels = cls_labels.to(self.device)
                clip_memory = None
                _, cls_preds = self.model(images, captions, cls_labels, clip_memory, self.criterion_cls, mode='focal')

                cls_labels[cls_labels==3] = 1
                test_prompt_gts.extend(cls_labels.cpu().numpy().tolist())
                test_prompt_res.extend(cls_preds)
                test_idx.extend(ids)
            test_prompt_ce = self.chexbert_metrics.compute_label(test_prompt_gts, test_prompt_res)
            log.update(**{'test_' + k: v for k, v in test_prompt_ce.items()})

            test_cls_json = {}
            for idx, preds in zip(test_idx, test_prompt_res):
                test_cls_json[idx] = preds
            
            with open (os.path.join(self.args.save_dir, 'test_focalloss_cls3.json'), 'w') as f:
                json.dump(test_cls_json, f)

        return log

    def train_side_cls(self, epoch):
        self.model.train()
        train_loss = 0
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

        for batch_idx, (idx, images1, images2, captions, cls_labels, _) in enumerate(self.train_dataloader):
            images1 = images1.to(self.device)
            images2 = images2.to(self.device)
            cls_labels = cls_labels.to(self.device)
            clip_memory = None
            self.lr_scheduler.step(cur_epoch=epoch, cur_step=batch_idx)
            # loss_cls, _ = self.model(images1, captions, cls_labels, clip_memory, self.criterion_cls, base_probs=self.base_probs, mode='side')
            # loss = loss_cls
            
            cls_preds_logits1, ratio = self.model(images1, captions, cls_labels, clip_memory, self.criterion_cls, base_probs=self.base_probs, mode='side')
            cls_preds_logits2, ratio = self.model(images2, captions, cls_labels, clip_memory, self.criterion_cls, base_probs=self.base_probs, mode='side')
            
            cos_similarity = cos(cls_preds_logits1, cls_preds_logits2).mean()
            ssl_loss = cos_similarity
            loss = -ssl_loss 
            
            # loss =  self.args.cls_weight*loss_cls
            if batch_idx%100 == 0:
                print("{}/{} loss: {}".format(batch_idx, len(self.train_dataloader), loss.item()))
            train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()
            self.optimizer.zero_grad()

        log = {'train_loss': train_loss / len(self.train_dataloader), 'ratio': ratio}

        self.model.eval()
        test_prompt_gts, test_prompt_res = [], []
        test_idx = []
        with torch.no_grad():
            for batch_idx, (ids, images, captions, cls_labels, cls_preds_fd) in enumerate(self.test_dataloader):
                images = images.to(self.device) 
                cls_labels = cls_labels.to(self.device)
                clip_memory = None
                cls_preds = self.model(images, captions, cls_labels, clip_memory, self.criterion_cls, mode='side')
                test_prompt_gts.extend(cls_labels.cpu().numpy().tolist())
                test_prompt_res.extend(cls_preds)
                test_idx.extend(ids)
            test_prompt_ce = self.chexbert_metrics.compute_label(test_prompt_gts, test_prompt_res)
            log.update(**{'test_' + k: v for k, v in test_prompt_ce.items()})

            test_cls_json = {}
            for idx, preds in zip(test_idx, test_prompt_res):
                test_cls_json[idx] = preds
            
            with open (os.path.join(self.args.save_dir, 'test_side2.json'), 'w') as f:
                json.dump(test_cls_json, f)

        return log

    def get_dinov2_feats(self):
        self.model.eval()

        with torch.no_grad():
            for batch_idx, (ids, images, captions, cls_labels) in tqdm(enumerate(self.train_dataloader)):
                print(ids)
                images = images.to(self.device)
                patch_embeds, avg_embeds = self.model.module.get_visual_repr(images)
                patch_embeds = patch_embeds.cpu().numpy()
                avg_embeds = avg_embeds.cpu().numpy()

                for i in range(len(ids)):
                    np.save(os.path.join(self.args.save_dir, f'{ids[i]}_patch.npy'), patch_embeds[i])

            for batch_idx, (ids, images, captions, cls_labels) in tqdm(enumerate(self.val_dataloader)):
                images = images.to(self.device)
                patch_embeds, avg_embeds = self.model.module.get_visual_repr(images)
                patch_embeds = patch_embeds.cpu().numpy()
                avg_embeds = avg_embeds.cpu().numpy()

                for i in range(len(ids)):
                    np.save(os.path.join(self.args.save_dir, f'{ids[i]}_patch.npy'), patch_embeds[i])

            for batch_idx, (ids, images, captions, cls_labels) in tqdm(enumerate(self.test_dataloader)):
                images = images.to(self.device)
                patch_embeds, avg_embeds = self.model.module.get_visual_repr(images)
                patch_embeds = patch_embeds.cpu().numpy()
                avg_embeds = avg_embeds.cpu().numpy()

                for i in range(len(ids)):
                    np.save(os.path.join(self.args.save_dir, f'{ids[i]}_patch.npy'), patch_embeds[i])