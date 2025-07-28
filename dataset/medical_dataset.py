import json
import os
import torch
import numpy as np

from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from .utils import my_pre_caption
import os
from glob import glob

CONDITIONS = [
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

SCORES = [
'[BLA]',
'[POS]',
'[NEG]',
'[UNC]'
]


class generation_train(Dataset):
    def __init__(self, transform, image_root, ann_root, tokenizer, dataset='mimic_cxr', args=None):
        
        self.annotation = json.load(open(os.path.join(ann_root),'r'))
        self.ann = self.annotation['train']
        self.transform = transform
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.max_words = args.gen_max_len
        self.dataset = dataset
        self.args = args

    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.ann[index]
        ids = ann['id']
        image_path = ann['image_path']

        image = Image.open(os.path.join(self.image_root, image_path[0])).convert('RGB')
        image = self.transform(image)
        
        cls_labels = ann['labels']
        cls_labels = np.array(cls_labels)[:14]

        prompt = [SCORES[l] for l in cls_labels]
        prompt = ' '.join(prompt)+' '
    
        original_caption = my_pre_caption(ann['report'], self.max_words, dataset=self.dataset)
        caption = prompt + original_caption
        
        cls_labels = torch.from_numpy(cls_labels)

        return ids, image, caption, cls_labels
    
class generation_eval(Dataset):
    def __init__(self, transform, image_root, ann_root, tokenizer, split='val', dataset='mimic_cxr', args=None):
        self.annotation = json.load(open(os.path.join(ann_root), 'r'))
        if dataset == 'mimic_cxr':
            self.ann = self.annotation[split]
        else: # IU
            self.ann = self.annotation
        self.split = split
        self.transform = transform
        self.max_words = args.gen_max_len
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.args = args
    
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.ann[index]
        ids = ann['id']
        image_path = ann['image_path']

        image = Image.open(os.path.join(self.image_root, image_path[0])).convert('RGB')
        image = self.transform(image)
        
        caption = my_pre_caption(ann['report'], self.max_words, dataset=self.dataset)
        cls_labels = ann['labels']
        cls_labels = torch.from_numpy(np.array(cls_labels)[:14])

        return ids, image, caption, cls_labels