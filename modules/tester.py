import logging
import os
from abc import abstractmethod
import numpy as np
import time

import cv2
import torch

from .metrics_clinical import CheXbertMetrics
from models.grad_cam import GradCAM
import json
class BaseTester(object):
    def __init__(self, model, criterion_cls, metric_ftns, args, device):
        self.args = args
        self.model = model
        self.device = device

        self.chexbert_metrics = CheXbertMetrics('/home/xmli/hl_yang/OS_RRG/checkpoints/chexbert/chexbert.pth', args.batch_size, device)

        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
        self.logger = logging.getLogger(__name__)


        self.criterion_cls = criterion_cls
        self.metric_ftns = metric_ftns

        self.epochs = self.args.epochs
        self.save_dir = self.args.save_dir

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    @abstractmethod
    def test(self):
        raise NotImplementedError

    @abstractmethod
    def plot(self):
        raise NotImplementedError

import pandas as pd
from tqdm import tqdm
class Tester(BaseTester):
    def __init__(self, model, criterion_cls, metric_ftns, args, device, test_dataloader):
        super(Tester, self).__init__(model, criterion_cls, metric_ftns, args, device)
        self.test_dataloader = test_dataloader

    def test_blip(self, split='test'):
        data_loader = self.test_dataloader
        self.logger.info('Start to evaluate in the test set.')
        log = dict()
        self.model.eval()
        epoch_iter = tqdm(data_loader, desc="Iteration")
        with torch.no_grad():
            test_gts, test_res = [], []
            cls_preds_all, cls_gts_all = [], []
            idx_all = []
            cls_labels_all, cls_preds_all = [], []
            for batch_idx, (ids, images, captions, cls_labels) in enumerate(epoch_iter):
                images = images.to(self.device) 
                cls_labels = cls_labels.to(self.device)
                reports, cls_preds = self.model.generate(images, cls_labels, sample=False, num_beams=self.args.beam_size, max_length=self.args.gen_max_len, min_length=self.args.gen_min_len)
                
                test_res.extend(reports)
                test_gts.extend(captions)
                cls_gts_all.extend(cls_labels.cpu().numpy().tolist())
                idx_all.extend(ids)
                cls_labels_all.extend(cls_labels.cpu().numpy().tolist())
                cls_preds_all.extend(cls_preds.cpu().numpy().tolist())

                if batch_idx % 10 == 0:
                    print('{}/{}'.format(batch_idx, len(self.test_dataloader)))

            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            test_ce, chexbert_ann = self.chexbert_metrics.compute(test_gts, test_res)
            gts_chexbert, res_chexbert = chexbert_ann[0].tolist(), chexbert_ann[1].tolist()

            log.update(**{f'{split}_' + k: v for k, v in test_met.items()})
            log.update(**{f'{split}_' + k: v for k, v in test_ce.items()})

            test_res_df = pd.DataFrame({'id': idx_all,
                                        'gts chexbert': gts_chexbert,
                                        'res chexbert': res_chexbert,
                                        'gts label': cls_labels_all,
                                        'res label': cls_preds_all,
                                        'predict_report': test_res,
                                        'reference_report': test_gts})
            
            test_res_df.to_csv(os.path.join(self.args.save_dir, "outputs.csv"), index=False)

        return log
    