import sys
sys.path.append('/home/xmli/hl_yang/TOG_RRG')
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor import Meteor
from pycocoevalcap.rouge import Rouge
from pycocoevalcap.cider import Cider

def compute_scores(gts, res):
    """
    Performs the MS COCO evaluation using the Python 3 implementation (https://github.com/salaniz/pycocoevalcap)

    :param gts: Dictionary with the image ids and their gold captions,
    :param res: Dictionary with the image ids ant their generated captions
    :print: Evaluation score (the mean of the scores of all the instances) for each measure
    """
    # post-processing, make format consistent
    for k in res.keys():
        res[k][0] = (res[k][0]+' ').replace('. ', ' . ').replace(' - ', '-')

    # Set up scorers
    scorers = [
        (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    eval_res = {}
    # Compute score for each metric
    for scorer, method in scorers:
        try:
            score, scores = scorer.compute_score(gts, res, verbose=0)
        except TypeError:
            score, scores = scorer.compute_score(gts, res)
        if type(method) == list:
            for sc, m in zip(score, method):
                eval_res[m] = sc
        else:
            eval_res[method] = score
    return eval_res

def compute_bleu(gts, res):
    for k in res.keys():
        res[k][0] = (res[k][0]+' ').replace('. ', ' . ').replace(' - ', '-')
    
    # Set up scorers
    scorers = [
        (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L")
    ]

    eval_res = {}
    for scorer, method in scorers:
        try:
            score, scores = scorer.compute_score(gts, res, verbose=0)
        except TypeError:
            score, scores = scorer.compute_score(gts, res)
        if type(method) == list:
            for sc, m in zip(scores, method):
                eval_res[m] = sc
        else:
            eval_res[method] = scores

    return eval_res

def compute_cider(gts, res):
    for k in res.keys():
        res[k][0] = (res[k][0]+' ').replace('. ', ' . ').replace(' - ', '-')
    
    # Set up scorers
    scorers = [
        (Cider(), "CIDEr")
    ]

    eval_res = {}
    for scorer, method in scorers:
        try:
            score, scores = scorer.compute_score(gts, res, verbose=0)
        except TypeError:
            score, scores = scorer.compute_score(gts, res)
        if type(method) == list:
            for sc, m in zip(scores, method):
                eval_res[m] = sc
        else:
            eval_res[method] = scores

    return eval_res['CIDEr']

def compute_meteor(gts, res):
    for k in res.keys():
        res[k][0] = (res[k][0]+' ').replace('. ', ' . ').replace(' - ', '-')
    
    # Set up scorers
    scorers = [
        (Meteor(), "METEOR")
    ]

    eval_res = {}
    for scorer, method in scorers:
        try:
            score, scores = scorer.compute_score(gts, res, verbose=0)
        except TypeError:
            score, scores = scorer.compute_score(gts, res)
        if type(method) == list:
            for sc, m in zip(scores, method):
                eval_res[m] = sc
        else:
            eval_res[method] = scores

    return eval_res['METEOR']

    

import re
def clean_report_mimic_cxr(report):
        report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
            .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
            .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
            .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
            .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                        .replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report

def clean_report_iu_xray(report):
        report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
            .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                        replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report

def my_pre_caption(caption, max_words=128, dataset='mimic_cxr'):
    if dataset == 'mimic_cxr':
        caption = clean_report_mimic_cxr(caption)
    elif dataset == 'iu_xray':
        caption = clean_report_iu_xray(caption)
    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
    return caption

def cleanStr(token):
    parts = re.split(r'[,:.\n]', token.lower())
    filtered_parts = []
    for part in parts:
        part = part.lstrip().rstrip()
        if part != '' and '_' not in part:
            filtered_parts.append(part)
        
    token = '.'.join(filtered_parts)
    token = re.sub(r'\s+', ' ', token).lstrip().rstrip()
    
    return token

import pandas as pd
import json
def main():

    # results = json.loads(open('/home/xmli/hl_yang/TOG_RRG/results/MLLM/mimic_cxr_train2_4_gmaiv1_3_VEdinov2_dynamic_rexrank.json').read())
    # gts = [clean_report_mimic_cxr(r['gt_answer']).lower() for r in results]
    # res = [clean_report_mimic_cxr(r['pred_answer'].lower()) for r in results]   


    # results = json.loads(open('/home/xmli/hl_yang/TOG_RRG/results/MLLM/iuxray_train2_4_gmaiv1_3_VEdinov2_dynamic.json').read())
    # gts = [clean_report_iu_xray(r['gt_answer']).lower() for r in results]
    # res = [clean_report_iu_xray(r['pred_answer'].lower()) for r in results] 
    
    results = json.loads(open('/home/xmli/hl_yang/TOG_RRG/results/MLLM/mimic_cxr_train3_8_radfm.json').read())
    gts = [clean_report_mimic_cxr(r['gt_answer']) for r in results]
    res = [clean_report_mimic_cxr(r['pred_answer']) for r in results]  

    # gts = pd.read_csv('/home/xmli/hl_yang/MRG/PromptMRG_vs/test_gts.csv')['report'].tolist()
    # res = pd.read_csv('/home/xmli/hl_yang/MRG/PromptMRG_vs/test_res.csv')['report'].tolist()
    # gts = json.loads(open('/home/xmli/hl_yang/TOG_RRG/results/mimic_cxr/20240907_shanshan/gts.json').read())
    # gts = pd.read_csv('/home/xmli/hl_yang/TOG_RRG/results/mimic_cxr/gts_maxlenfull.csv')['report'].tolist()
    # res = pd.read_csv('/home/xmli/hl_yang/MRG/FITA_v2/results/20240807_contextual_v1/res_nocfe_nollm_bs16.csv')['report'].tolist()
    # res = pd.read_csv('/home/xmli/hl_yang/TOG_RRG/results/mimic_cxr/20240903_softprompt/train/res_add.csv')['report'].tolist()
    # res = json.loads(open('/home/xmli/hl_yang/TOG_RRG/results/mimic_cxr/20240907_shanshan/res.json').read())
    # res = [re.replace('[sum]', '').replace('  ', ' ') for re in res]
    # print(res[0])
    # gts = json.loads(open('/home/xmli/hl_yang/MRG/FoundationModel/LLaVA/playground/data/iu_xray_icl/iu_xray_gt.json').read())
    # gts = [g['report'] for g in gts]
    # res = json.loads(open('/home/xmli/hl_yang/MRG/FoundationModel/LLaVA/playground/data/iu_xray_icl/iu_xray_togrrg.json').read())
    # res = [r['report'] for r in res]
    # print(len(gts))
    # res = [json.loads(q)['text'] for q in open(('/home/xmli/hl_yang/MRG/FoundationModel/LLaVA/playground/data/iu_xray_icl/iu_xray_icl_ans.jsonl'), "r")]
    # print(len(res))
    # res = [json.loads(q)['text'] for q in open(('/home/xmli/hl_yang/MRG/FoundationModel/LLaVA/playground/data/iu_xray_qa_eval/iuxray_test_answer_hypo_max256.jsonl'), "r")]
    # print(res[0])
    # res = pd.read_csv('/home/xmli/hl_yang/MRG/FoundationModel/InternVL/Results/20240821_iuxray/10_shot_predict_dignosis.csv')['report'].tolist()
    # res = [r.lower().replace('\"', '').strip() for r in res]
    # len_res = len(res)
    # # len_res = 458
    # print(len_res)
    # res = res[:len_res]
    # gts = gts[:len_res]
    # res = [r.replace('.', ' .').replace(' - ', '-').replace('re-', 're- ') for r in res]

    # gts_view = []
    # res_view = []
    # ann_file = '/home/xmli/hlyang/MRG/PromptMRG_V2/data/mimic_cxr/mimic_annotation_promptmrg.json'
    # ann_test = json.loads(open(ann_file).read())['test']
    # for i in range(len(ann_test)):
    #     view = ann_test[i]['view']
    #     if view == 3:
    #         gts_view.append(gts[i])
    #         res_view.append(res[i])
    
    # res_pd = pd.DataFrame(res)
    # res_pd.to_csv('/home/xmli/hlyang/MRG/Results/M2KT_res.csv', index=False)
    # import re
    # gts = [re.sub(' +', ' ', text.lower().replace(".", " .").replace("\"", '')) for text in gts]
    # res = [re.sub(' +', ' ', text.lower().replace(".", " .").replace("\"", '')) for text in res]
    # gts = [g.replace(' -', '-').replace('- ', '-').replace(' - ', '-').replace('  .', ' .') for g in gts]
    # res = [r.replace('.', ' .') for r in res]
    # res = [r + ' .' if r[-1] != '.' else r for r in res]
    # gts = [my_pre_caption(gt) for gt in gts]
    # res = [my_pre_caption(re) for re in res]

    # print(gts[0])
    # print([res[0]])
    # gts = [g.lower().replace('.', ' .') for g in gts]
    # res = [r.lower().replace('.', ' .') for r in res]
    scores = compute_scores({i: [gt] for i, gt in enumerate(gts)}, {i: [re] for i, re in enumerate(res)})
    print(scores)
    
    # from bert_score import BERTScorer
    # scorer = BERTScorer(
    #     model_type="distilroberta-base",
    #     batch_size=256,
    #     lang="en",
    #     rescale_with_baseline=True)

    # _, _, f1 = scorer.score(gts, res)
    # print(f1.mean().item())
    
    # from fast_bleu import BLEU
    # weights = {"bigram": (0.5, 0.5)}
    # scores = 0
    # for i in range(len(gts)):
    #     gts_i = gts[i].split()
    #     res_i = res[i].split()
    #     bleu = BLEU([gts_i], weights)
    #     score = bleu.get_score([res_i])["bigram"]
    #     scores += score[0]
    # print(scores/len(gts))

if __name__ == '__main__':
    main()