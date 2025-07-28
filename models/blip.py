import os
import warnings
warnings.filterwarnings("ignore")

from models.med import BertConfig, BertModel, BertLMHeadModel
from transformers import BertTokenizer, AutoModel,BertConfig
from models.resnet import blip_resnet
from models.sbd import SBD
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from models.transformer import Transformer
from bert_score import BERTScorer

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


class BLIP_Decoder(nn.Module):
    def __init__(self,                 
                 args,
                 tokenizer=None,
                 image_size = 224,
                 prompt = '',
                 ):
        super().__init__()
        self.args = args
        
        vision_width = 2048
        self.visual_encoder = blip_resnet(args, name='resnet101')
        self.sbd = SBD(cls_width=vision_width, dropout=0.1, args=args)
        
        self.cls_head = nn.Linear(vision_width, 14*4)
        nn.init.normal_(self.cls_head.weight, std=0.001)
        if self.cls_head.bias is not None:
            nn.init.constant_(self.cls_head.bias, 0)
            
        self.tokenizer = tokenizer   
        
        decoder_config = BertConfig.from_json_file('configs/bert_config.json')
        decoder_config.encoder_width = vision_width
        decoder_config.add_cross_attention = True
        decoder_config.is_decoder = True

        
        self.text_decoder = BertLMHeadModel.from_pretrained('bert-base-uncased',config=decoder_config)
        
        self.text_decoder.resize_token_embeddings(len(self.tokenizer))

        self.embed_func = self.text_decoder.get_input_embeddings()
        
        self.prompt = prompt
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids)-1


    def forward_sbd(self, image, cls_labels, criterion_cls, base_probs=None):
        return self.sbd(image, cls_labels, criterion_cls, base_probs)

    def forward(self, image, caption, cls_labels):
        image_embeds, avg_embeds = self.visual_encoder(image) # [bs, 4, 2048]

        try:
            text = self.tokenizer(caption, padding='longest', truncation=True, return_tensors="pt").to(image.device)
        except:
            print(caption)
        text.input_ids[:,0] = self.tokenizer.bos_token_id
        
        decoder_targets = text.input_ids.masked_fill(text.input_ids == self.tokenizer.pad_token_id, -100) 
        decoder_targets[:,:self.prompt_length] = -100
        decoder_output = self.text_decoder(text.input_ids, 
                                           attention_mask = text.attention_mask, 
                                           encoder_hidden_states = image_embeds,
                                           labels = decoder_targets,
                                           return_dict = True,   
                                           reduction = 'mean',
                                           state_embed = None)
        loss_lm = decoder_output.loss   

        return loss_lm
    

    def generate(self, image, cls_labels, sample=False, num_beams=3, max_length=100, min_length=10, top_p=0.9, repetition_penalty=1.0, mode="train"):
        
        image_embeds, avg_embeds = self.visual_encoder(image) 
        if mode == 'train':
            cls_preds = self.sbd.get_cls_preds(image)
        else:
            cls_preds = cls_labels
        
        prompts = []
        for j in range(len(cls_preds)):
            prompt = ' '.join([SCORES[c] for c in cls_preds[j]])+' '
            prompts.append(prompt)

        if not sample:   
            image_embeds = image_embeds.repeat_interleave(num_beams,dim=0)
            
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image_embeds.device)
        model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask":image_atts}
                
        text = self.tokenizer(prompts, return_tensors="pt")
        input_ids = text.input_ids.to(image_embeds.device)
        attn_masks = text.attention_mask.to(image_embeds.device)
        input_ids[:,0] = self.tokenizer.bos_token_id
        input_ids = input_ids[:, :-1] 
        attn_masks = attn_masks[:, :-1] 

        # cls_labels with value 1,2,3 equal to 1 and 0 equal to 0
        cls_labels_mask = torch.where(cls_preds == 0, 0, 1).to(image.device)
        cls_labels_mask = torch.concat([attn_masks[:, :1], cls_labels_mask], 1)
        if not sample:
            cls_labels_mask = cls_labels_mask.repeat_interleave(num_beams,dim=0)
        # add cls_labels_mask to the model_kwargs
        model_kwargs['cls_label_mask'] = cls_labels_mask

   
        outputs = self.text_decoder.generate(input_ids=input_ids,
                                            min_length=min_length, # 4.25 Transformers
                                            max_new_tokens=max_length,
                                            num_beams=num_beams,
                                            eos_token_id=self.tokenizer.sep_token_id,
                                            pad_token_id=self.tokenizer.pad_token_id, 
                                            repetition_penalty=repetition_penalty,
                                            attention_mask = attn_masks,
                                            output_scores=True,
                                            output_attentions = True,
                                            use_cache=True,
                                            return_dict_in_generate = True,
                                            **model_kwargs)   

        captions = []    
        for i, output in enumerate(outputs.sequences):
            caption = self.tokenizer.decode(output, skip_special_tokens=True) 
            captions.append(caption[len(prompts[i]):])

        return captions, cls_preds


def blip_decoder(args, tokenizer, **kwargs):
    model = BLIP_Decoder(args, tokenizer, **kwargs)
    return model    
    