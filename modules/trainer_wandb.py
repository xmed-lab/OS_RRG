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
import torch.nn.functional as F
import json
import wandb
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

        # init wandb here
        if self.is_main_process:
            wandb.init(
                project="OSRRG",
                name=args.mode,
                config=vars(args)
            )


        self.chexbert_metrics = CheXbertMetrics('./checkpoints/chexbert/chexbert.pth', args.batch_size, device)

        self.criterion_cls = criterion_cls

        self.base_probs = base_probs
        self.metric_ftns = metric_ftns
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
                self.train_dataloader.sampler.set_epoch(epoch)
            if self.mode == 'SBD':
                result = self.train_SBD(epoch)
                # save logged information 
                log = {'epoch': epoch}
                log.update(result)
                # print logged information 
                for key, value in log.items():
                    print('\t{:15s}: {}'.format(str(key), value))
                
                last_path = os.path.join(self.checkpoint_dir, 'SBD.pth')
                torch.save(self.model.module.state_dict(), last_path)
                print("Saving current best to {}".format(last_path))
                continue
            
            result = self._train_epoch_blip(epoch)
            dist.barrier()
            result = self.eval_blip(result, test=False)

            # save logged information 
            log = {'epoch': epoch}
            log.update(result)

            # record best
            if self.is_main_process:
                if log[self.mnt_metric] >= self.mnt_best:
                    best_path = os.path.join(self.checkpoint_dir, f'osrrg_best.pth')
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
        train_loss_lm = 0
        self.model.train()
        for batch_idx, (ids, images, captions, cls_labels) in enumerate(self.train_dataloader):
            images = images.to(self.device)
            cls_labels = cls_labels.to(self.device)
            self.lr_scheduler.step(cur_epoch=epoch, cur_step=batch_idx)
            loss_lm = self.model(images, captions, cls_labels)
            loss = loss_lm
            if batch_idx%10 == 0:
                print("{}/{} loss: {}, loss_lm: {}".format(batch_idx, len(self.train_dataloader), loss.item(), loss_lm.item()))
                # Log to wandb every 10 batches
                if self.is_main_process:
                    wandb.log({
                        "train/batch_loss": loss.item(),
                        "train/learning_rate": self.optimizer.param_groups[0]['lr'],
                        "epoch": epoch,
                        "batch": batch_idx
                    })
            train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()
            self.optimizer.zero_grad()

        log = {'train_loss': train_loss / len(self.train_dataloader)}
        
        # Log epoch metrics to wandb
        if self.is_main_process:
            wandb.log({
                "train/epoch_loss": train_loss / len(self.train_dataloader),
                "epoch": epoch
            })

        return log

    def eval_blip(self, log, test=True):
        self.model.module.eval()

        if test is not True:
            with torch.no_grad():
                val_gts, val_res = [], []
                val_prompt_gts, val_prompt_res = [], []
                for batch_idx, (ids, images, captions, cls_labels) in enumerate(self.val_dataloader):
                    images = images.to(self.device) 
                    cls_labels = cls_labels.to(self.device)
                    ground_truths = captions
                    reports, cls_preds = self.model.module.generate(images, cls_labels, sample=False, num_beams=self.args.beam_size, 
                                                                    max_length=self.args.gen_max_len, min_length=self.args.gen_min_len, mode="train")
                    
                    if batch_idx%10 == 0:
                        print(f'val: {batch_idx}/{len(self.val_dataloader)}')
                    val_res.extend(reports)
                    val_gts.extend(ground_truths)


                val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                        {i: [re] for i, re in enumerate(val_res)})
                val_ce, _ = self.chexbert_metrics.compute(val_gts, val_res)
                log.update(**{'val_' + k: v for k, v in val_met.items()})
                log.update(**{'val_' + k: v for k, v in val_ce.items()})
                
                # Log validation metrics to wandb
                if self.is_main_process:
                    # Create a dictionary with all validation metrics
                    val_metrics = {}
                    for k, v in val_met.items():
                        val_metrics[f"val/nlg_{k}"] = v
                    for k, v in val_ce.items():
                        val_metrics[f"val/chexbert_{k}"] = v
                    
                    # Add a few example predictions for qualitative analysis
                    if len(val_gts) > 0 and len(val_res) > 0:
                        examples = []
                        for i in range(min(3, len(val_gts))):
                            examples.append([val_gts[i], val_res[i]])
                        val_metrics["val/examples"] = wandb.Table(
                            columns=["Ground Truth", "Prediction"],
                            data=examples
                        )
                    
                    wandb.log(val_metrics)
            return log
        
        elif test is True:
            with torch.no_grad():
                test_gts, test_res = [], []
                test_prompt_gts, test_prompt_res = [], []
                logits_level_all = []
                cls_labels_all, cls_preds_all = [], []
                for batch_idx, (ids, images, captions, cls_labels) in enumerate(self.test_dataloader):
                    images = images.to(self.device) 
                    cls_labels = cls_labels.to(self.device)
                    ground_truths = captions
                    reports, cls_preds = self.model.module.generate(images, cls_labels, sample=False, num_beams=self.args.beam_size, 
                                                                    max_length=self.args.gen_max_len, min_length=self.args.gen_min_len, mode="train")

                    if batch_idx%10 == 0:
                        print(f'test: {batch_idx}/{len(self.test_dataloader)}')

                    test_res.extend(reports)
                    test_gts.extend(ground_truths)


                test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                            {i: [re] for i, re in enumerate(test_res)})
                test_ce, _ = self.chexbert_metrics.compute(test_gts, test_res)

                try:
                    test_res, test_gts = pd.DataFrame(test_res), pd.DataFrame(test_gts)
                    test_gts.to_csv(os.path.join(self.args.save_dir, "gts.csv"), index=False, header=False)
                    test_res.to_csv(os.path.join(self.args.save_dir, "res.csv"), index=False, header=False)
                    
                    # Convert back to lists for logging examples
                    test_gts_list = test_gts.values.flatten().tolist()
                    test_res_list = test_res.values.flatten().tolist()
                except:
                    test_gts_list = test_gts
                    test_res_list = test_res
                    
                log.update(**{'test_' + k: v for k, v in test_met.items()})
                log.update(**{'test_' + k: v for k, v in test_ce.items()})
                
                # Log test metrics to wandb
                if self.is_main_process:
                    # Create a dictionary with all test metrics
                    test_metrics = {}
                    for k, v in test_met.items():
                        test_metrics[f"test/nlg_{k}"] = v
                    for k, v in test_ce.items():
                        test_metrics[f"test/chexbert_{k}"] = v
                    
                    # Add a few example predictions for qualitative analysis
                    if len(test_gts_list) > 0 and len(test_res_list) > 0:
                        examples = []
                        for i in range(min(3, len(test_gts_list))):
                            examples.append([test_gts_list[i], test_res_list[i]])
                        test_metrics["test/examples"] = wandb.Table(
                            columns=["Ground Truth", "Prediction"],
                            data=examples
                        )
                    
                    wandb.log(test_metrics)
            return log

            
    def train_SBD(self, epoch):
        log = {}
        self.model.train()
        train_loss = 0
        pos_loss, neg_loss, unc_loss = 0, 0, 0
        logits = {'pos': [], 'neg': [], 'unc': []}
        counts = {'pos': [], 'neg': [], 'unc': []}
        for batch_idx, (ids, images, captions, cls_labels) in enumerate(self.train_dataloader):
            self.lr_scheduler.step(cur_epoch=epoch, cur_step=batch_idx)
            images = images.to(self.device)
            cls_labels = cls_labels.to(self.device)
            loss_cls, loss_all, cls_preds, cls_preds_logits = self.model.module.forward_sbd(images, cls_labels, self.criterion_cls, base_probs=self.base_probs)
            loss = loss_cls
            if batch_idx%100 == 0:
                print("{}/{} loss: {}".format(batch_idx, len(self.train_dataloader), loss.item()))
                # Log to wandb every 100 batches
                if self.is_main_process:
                    wandb.log({
                        "SBD/batch_loss": loss.item(),
                        "SBD/batch_pos_loss": loss_all[1],
                        "SBD/batch_neg_loss": loss_all[2],
                        "SBD/batch_unc_loss": loss_all[0],
                        "SBD/learning_rate": self.optimizer.param_groups[0]['lr'],
                        "epoch": epoch,
                        "batch": batch_idx
                    })
            
            train_loss += loss.item()
            pos_loss += loss_all[1]
            neg_loss += loss_all[2]
            unc_loss += loss_all[0]

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

        logit_unc = np.concatenate(logits['unc'], axis=0)
        count_unc = np.concatenate(counts['unc'], axis=0)
        logit_unc = np.sum(logit_unc, axis=0)
        count_unc = np.sum(count_unc, axis=0)
        # nan to 1
        count_unc[count_unc==0] = 1.0
        logit_unc = logit_unc/count_unc

        self.base_probs[1] = logit_p
        self.base_probs[2] = logit_n
        self.base_probs[3] = logit_unc

        # No Finding only have pos and bla states
        self.base_probs[2, -1] = 1.0
        self.base_probs[3, -1] = 1.0

        log = {'train_loss': train_loss / len(self.train_dataloader), 'pos_loss': pos_loss / len(self.train_dataloader), 'neg_loss': neg_loss / len(self.train_dataloader), 'unc_loss': unc_loss / len(self.train_dataloader)}

        # Log training metrics to wandb
        if self.is_main_process:
            wandb.log({
                "SBD/epoch_loss": train_loss / len(self.train_dataloader),
                "SBD/epoch_pos_loss": pos_loss / len(self.train_dataloader),
                "SBD/epoch_neg_loss": neg_loss / len(self.train_dataloader),
                "SBD/epoch_unc_loss": unc_loss / len(self.train_dataloader),
                "epoch": epoch
            })
            
            # Log base_probs as histograms
            wandb.log({
                "base_probs/pos": wandb.Histogram(self.base_probs[1]),
                "base_probs/neg": wandb.Histogram(self.base_probs[2]),
                "base_probs/unc": wandb.Histogram(self.base_probs[3]),
                "epoch": epoch
            })

        self.model.eval()
        test_prompt_gts = [] 
        test_prompt_res = {0:[], 1:[], 2:[], 3:[], 4:[]}
        test_cls_preds_logits = {0:[], 1:[], 2:[], 3:[]}
        test_idx = []
        with torch.no_grad():
            for batch_idx, (ids, images, captions, cls_labels) in enumerate(self.test_dataloader):
                
                images = images.to(self.device) 
                cls_labels = cls_labels.to(self.device)
                _, _, cls_preds, cls_preds_logits = self.model.module.forward_sbd(images, cls_labels, self.criterion_cls, base_probs=self.base_probs)

                test_prompt_gts.extend(cls_labels.cpu().numpy().tolist())
                test_idx.extend(ids)
                for i, cls_preds_i in enumerate(cls_preds):
                    test_prompt_res[i].extend(cls_preds_i.cpu().numpy().tolist())


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
            
            # Log test metrics to wandb
            if self.is_main_process:
                test_metrics = {}
                for k, v in test_prompt_ce_bla.items():
                    test_metrics[f"SBD/test_bla_{k}"] = v
                for k, v in test_prompt_ce_unc.items():
                    test_metrics[f"SBD/test_unc_{k}"] = v
                for k, v in test_prompt_ce_pos.items():
                    test_metrics[f"SBD/test_pos_{k}"] = v
                for k, v in test_prompt_ce_neg.items():
                    test_metrics[f"SBD/test_neg_{k}"] = v
                for k, v in test_prompt_ce.items():
                    test_metrics[f"SBD/test_{k}"] = v
                wandb.log(test_metrics)

        # save test_idx, test_prompt_res as json
        test_cls_json = {}
        for idx, preds in zip(test_idx, test_prompt_res[3]):
            test_cls_json[idx] = preds
        
        with open (os.path.join(self.args.save_dir, f'SBD_epoch{epoch}.json'), 'w') as f:
            json.dump(test_cls_json, f)
            
        # Also save the results to wandb for each epoch
        if self.is_main_process:
            # Save the model predictions to wandb with epoch in filename
            json_path = os.path.join(self.args.save_dir, f'SBD_epoch{epoch}.json')
            with open(json_path, 'w') as f:
                json.dump(test_cls_json, f)
            wandb.save(json_path)
            wandb.log({"epoch": epoch})


        return log
