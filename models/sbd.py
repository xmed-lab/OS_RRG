from models.resnet import blip_resnet
import torch
import torch.nn as nn
import torch.nn.functional as F

class SBD(nn.Module):
    def __init__(self, cls_width=2048, dropout=0.1, args=None):
        super(SBD, self).__init__()

        self.visual_encoder = blip_resnet(args, name='resnet50')

        self.cls_head_ex1 = nn.Linear(cls_width, 4*14)
        self.cls_head_ex2 = nn.Linear(cls_width, 4*14)
        self.cls_head_ex3 = nn.Linear(cls_width, 4*14)
        self.dropout = nn.Dropout(dropout)

        # # init 1, 2, 3
        nn.init.normal_(self.cls_head_ex1.weight, std=0.001)
        nn.init.normal_(self.cls_head_ex2.weight, std=0.001)
        nn.init.normal_(self.cls_head_ex3.weight, std=0.001)

    def forward(self, image, cls_labels, criterion_cls, base_probs):
        image_embeds, avg_embeds = self.visual_encoder(image)
        cls_preds_ex1 = self.cls_head_ex1(avg_embeds)
        cls_preds_ex2 = self.cls_head_ex2(avg_embeds)
        cls_preds_ex3 = self.cls_head_ex3(avg_embeds)

        # # dropout
        cls_preds_ex1 = self.dropout(cls_preds_ex1)
        cls_preds_ex2 = self.dropout(cls_preds_ex2)
        cls_preds_ex3 = self.dropout(cls_preds_ex3)

        # # # BCE loss
        cls_preds_ex1 = cls_preds_ex1.view(-1, 4, 14)
        cls_preds_ex2 = cls_preds_ex2.view(-1, 4, 14)
        cls_preds_ex3 = cls_preds_ex3.view(-1, 4, 14)

        extra_info = {'logits': [cls_preds_ex1, cls_preds_ex2, cls_preds_ex3]}

        if base_probs is not None:
            base_probs = torch.from_numpy(base_probs).to(image.device)
            loss_cls, loss_all = criterion_cls(cls_labels, extra_info=extra_info, base_probs=base_probs)
            loss_all = [l.item() for l in loss_all]
        else:
            loss_cls = 0
            loss_all = 0

        cls_preds_ex1 = F.softmax(cls_preds_ex1, dim=1)
        cls_preds_ex1_logits = cls_preds_ex1
        cls_preds_ex1 = torch.argmax(cls_preds_ex1, dim=1) # [bs, 13]

        cls_preds_ex2 = F.softmax(cls_preds_ex2, dim=1)
        cls_preds_ex2_logits = cls_preds_ex2
        cls_preds_ex2 = torch.argmax(cls_preds_ex2, dim=1) # [bs, 4, 14]

        cls_preds_ex3 = F.softmax(cls_preds_ex3, dim=1)
        cls_preds_ex3_logits = cls_preds_ex3
        cls_preds_ex3 = torch.argmax(cls_preds_ex3, dim=1)

        cls_preds_logits = torch.cat([cls_preds_ex1_logits, cls_preds_ex2_logits, cls_preds_ex3_logits], 1)
        cls_preds_ex = torch.argmax(cls_preds_logits, dim=1)
        cls_preds_ex %= 4

        cls_preds_ex = cls_preds_ex

        cls_preds = [cls_preds_ex1, cls_preds_ex2, cls_preds_ex3, cls_preds_ex]


        return loss_cls, loss_all, cls_preds, [cls_preds_ex1_logits, cls_preds_ex2_logits, cls_preds_ex3_logits]
    
    def get_cls_preds(self, image):
        image_embeds, avg_embeds = self.visual_encoder(image)
        cls_preds_ex1 = self.cls_head_ex1(avg_embeds).view(-1, 4, 14)
        cls_preds_ex2 = self.cls_head_ex2(avg_embeds).view(-1, 4, 14)
        cls_preds_ex3 = self.cls_head_ex3(avg_embeds).view(-1, 4, 14)

        cls_preds_ex1 = F.softmax(cls_preds_ex1, dim=1)
        cls_preds_ex1_logits = cls_preds_ex1
        cls_preds_ex1 = torch.argmax(cls_preds_ex1, dim=1) 
        # print(f'cls_preds_ex1: {cls_preds_ex1}')

        cls_preds_ex2 = F.softmax(cls_preds_ex2, dim=1)
        cls_preds_ex2_logits = cls_preds_ex2
        cls_preds_ex2 = torch.argmax(cls_preds_ex2, dim=1)
        # print(f'cls_preds_ex2: {cls_preds_ex2}')

        cls_preds_ex3 = F.softmax(cls_preds_ex3, dim=1)
        cls_preds_ex3_logits = cls_preds_ex3
        cls_preds_ex3 = torch.argmax(cls_preds_ex3, dim=1)

        cls_preds_logits = torch.cat([cls_preds_ex1_logits, cls_preds_ex2_logits, cls_preds_ex3_logits], 1)
        cls_preds_ex = torch.argmax(cls_preds_logits, dim=1)
        cls_preds_ex %= 4

        return cls_preds_ex
        
        
        
