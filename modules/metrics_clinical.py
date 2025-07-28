import sys 
# sys.path.append('/home/xmli/hlyang/MRG/PromptMRG_V2/modules')
# from chexbert import CheXbert

import os
from modules.chexbert import CheXbert
import numpy as np

"""
0 = blank/not mentioned
1 = positive
2 = negative
3 = uncertain
"""

CONDITIONS = [
    'enlarged_cardiomediastinum',
    'cardiomegaly',
    'lung_opacity',
    'lung_lesion',
    'edema',
    'consolidation',
    'pneumonia',
    'atelectasis',
    'pneumothorax',
    'pleural_effusion',
    'pleural_other',
    'fracture',
    'support_devices',
    'no_finding',
]

class CheXbertMetrics():
    def __init__(self, checkpoint_path, mbatch_size, device):
        self.checkpoint_path = checkpoint_path
        self.mbatch_size = mbatch_size
        self.device = device
        self.chexbert = CheXbert(self.checkpoint_path, self.device,).to(self.device)

    def mini_batch(self, gts, res, mbatch_size=16):
        length = len(gts)
        assert length == len(res)
        for i in range(0, length, mbatch_size):
            yield gts[i:min(i + mbatch_size, length)], res[i:min(i + mbatch_size, length)]

    def annotate(self, res):
        res_chexbert = []
        for re, re in self.mini_batch(res, res, self.mbatch_size):
            re_chexbert = self.chexbert(list(re)).tolist()
            res_chexbert += re_chexbert
        return res_chexbert

    def compute(self, gts, res):
        gts_chexbert = []
        res_chexbert = []
        for gt, re in self.mini_batch(gts, res, self.mbatch_size):
            gt_chexbert = self.chexbert(list(gt)).tolist()
            re_chexbert = self.chexbert(list(re)).tolist()
            gts_chexbert += gt_chexbert
            res_chexbert += re_chexbert

        gts_chexbert = np.array(gts_chexbert)
        res_chexbert = np.array(res_chexbert)

        gts_chexbert_ = gts_chexbert.copy()
        res_chexbert_ = res_chexbert.copy()

        # res_chexbert[res_chexbert == 3] = 1
        # gts_chexbert[gts_chexbert == 3] = 1

        res_chexbert = (res_chexbert == 1) 
        gts_chexbert = (gts_chexbert == 1)

        # count = 0
        # for i in range(len(gts_chexbert)):
        #     if count > 20:
        #         break
        #     elif gts_chexbert[i][7] == 0 and res_chexbert[i][7] == 1:
        #         print(i)
        #         print(gts[i])
        #         print('\n')
        #         print(res[i])
        #         count += 1

        
        tp = (res_chexbert * gts_chexbert).astype(float) # [batch_size, 14]

        fp = (res_chexbert * ~gts_chexbert).astype(float)
        fn = (~res_chexbert * gts_chexbert).astype(float)

        tn = (~res_chexbert * ~gts_chexbert).astype(float)

        # calculate confusion matrix for each class
        tp_cls = tp.sum(0) # [14]
        fp_cls = fp.sum(0)
        fn_cls = fn.sum(0)
        tn_cls = tn.sum(0)

        # for i in range(14):
        #     # table \t
        #     print(f'{CONDITIONS[i]}\t{tp_cls[i]}\t{fp_cls[i]}\t{fn_cls[i]}\t{tn_cls[i]}')

        tp_eg = tp.sum(1) # [batch_size]
        fp_eg = fp.sum(1)
        fn_eg = fn.sum(1)

        precision_class = np.nan_to_num(tp_cls / (tp_cls + fp_cls))
        recall_class = np.nan_to_num(tp_cls / (tp_cls + fn_cls))
        f1_class = np.nan_to_num(tp_cls / (tp_cls + 0.5 * (fp_cls + fn_cls)))

        for i in range(14):
            # table \t
            print(f'{CONDITIONS[i]}\t{precision_class[i]}\t{recall_class[i]}\t{f1_class[i]}')

        scores = {
            # example-based CE metrics
            'ce_precision': np.nan_to_num(tp_eg / (tp_eg + fp_eg)).mean(),
            'ce_recall': np.nan_to_num(tp_eg / (tp_eg + fn_eg)).mean(),
            'ce_f1': np.nan_to_num(tp_eg / (tp_eg + 0.5 * (fp_eg + fn_eg))).mean(),
            'ce_num_examples': float(len(res_chexbert)),
            'ce_precision_class': precision_class.mean(),
            'ce_recall_class': recall_class.mean(),
            'ce_f1_class': f1_class.mean(),
        }
        return scores, (gts_chexbert_, res_chexbert_)
    
    def compute_label(self, gts, res, k=1):
        gts_chexbert = np.array(gts)
        res_chexbert = np.array(res)

        # res_chexbert[res_chexbert == 3] = 1
        # gts_chexbert[gts_chexbert == 3] = 1

        res_chexbert = (res_chexbert == k)
        gts_chexbert = (gts_chexbert == k)

        tp = (res_chexbert * gts_chexbert).astype(float)

        fp = (res_chexbert * ~gts_chexbert).astype(float)
        fn = (~res_chexbert * gts_chexbert).astype(float)

        tp_cls = tp.sum(0)
        fp_cls = fp.sum(0)
        fn_cls = fn.sum(0)

        tp_eg = tp.sum(1)
        fp_eg = fp.sum(1)
        fn_eg = fn.sum(1)

        precision_class = np.nan_to_num(tp_cls / (tp_cls + fp_cls))
        recall_class = np.nan_to_num(tp_cls / (tp_cls + fn_cls))
        f1_class = np.nan_to_num(tp_cls / (tp_cls + 0.5 * (fp_cls + fn_cls)))

        for i in range(14):
            # table \t
            print(f'{CONDITIONS[i]}\t{precision_class[i]}\t{recall_class[i]}\t{f1_class[i]}')

        scores = {
            # example-based CE metrics
            'ce_precision': np.nan_to_num(tp_eg / (tp_eg + fp_eg)).mean(),
            'ce_recall': np.nan_to_num(tp_eg / (tp_eg + fn_eg)).mean(),
            'ce_f1': np.nan_to_num(tp_eg / (tp_eg + 0.5 * (fp_eg + fn_eg))).mean(),
            'ce_num_examples': float(len(res_chexbert)),
            'ce_precision_class': precision_class.mean(),
            'ce_recall_class': recall_class.mean(),
            'ce_f1_class': f1_class.mean(),
        }
        return scores

    def compute_step1(self, gts, res):
        gts_chexbert = np.array(gts)
        res_chexbert = np.array(res)

        res_chexbert = (res_chexbert == 2)
        gts_chexbert = (gts_chexbert == 2)

        tp = (res_chexbert * gts_chexbert).astype(float)
        fp = (res_chexbert * ~gts_chexbert).astype(float)
        fn = (~res_chexbert * gts_chexbert).astype(float)

        tp_cls = tp.sum(0)
        fp_cls = fp.sum(0)
        fn_cls = fn.sum(0)

        try:
            tp_eg = tp.sum(1)
            fp_eg = fp.sum(1)
            fn_eg = fn.sum(1)

            precision_example = np.nan_to_num(tp_eg / (tp_eg + fp_eg)).mean()
            recall_example = np.nan_to_num(tp_eg / (tp_eg + fn_eg)).mean()
            f1_example = np.nan_to_num(tp_eg / (tp_eg + 0.5 * (fp_eg + fn_eg))).mean()
        except:
            print('Error in computing the example-based metrics')

        precision = np.nan_to_num(tp_cls / (tp_cls + fp_cls))
        recall = np.nan_to_num(tp_cls / (tp_cls + fn_cls))
        f1 = np.nan_to_num(tp_cls / (tp_cls + 0.5 * (fp_cls + fn_cls)))

        scores = {
            'precision_example': precision_example,
            'recall_example': recall_example,
            'f1_example': f1_example,
            'precision_class': precision.mean(),
            'recall_class': recall.mean(),
            'f1_class': f1.mean(),
        }
        return scores

    def cal_rewards(self, res, gts):
        res_chexbert = []
        gts_chexbert = []
        for gt, re in self.mini_batch(gts, res, self.mbatch_size):
            re_chexbert = self.chexbert(list(re)).tolist()
            res_chexbert += re_chexbert

            gt_chexbert = self.chexbert(list(gt)).tolist()
            gts_chexbert += gt_chexbert

        res_chexbert = np.array(res_chexbert)
        gts_chexbert = np.array(gts_chexbert)

        rewards = 0
        for k in [1, 2, 3]:
            res_chexbert_k = (res_chexbert == int(k))
            gts_chexbert_k = (gts_chexbert == int(k))

            tp = (res_chexbert_k * gts_chexbert_k).astype(float)

            fp = (res_chexbert_k * ~gts_chexbert_k).astype(float)
            fn = (~res_chexbert_k * gts_chexbert_k).astype(float)

            tp_eg = tp.sum(1)
            fp_eg = fp.sum(1)
            fn_eg = fn.sum(1)

            rewards += np.nan_to_num(tp_eg / (tp_eg + 0.5 * (fp_eg + fn_eg)))

        return rewards



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

def my_pre_caption(caption, max_words=500, dataset='mimic_cxr'):
    if dataset == 'mimic_cxr':
        caption = clean_report_mimic_cxr(caption)
    elif dataset == 'iu_xray':
        caption = clean_report_iu_xray(caption)
    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
    return caption

import pandas as pd
import json
def main():
    checkpoint_path = '/home/xmli/hl_yang/TOG_RRG/checkpoints/stanford/chexbert/chexbert.pth'
    # checkpoint_path = '/home/xmli/hl_yang/TOG_RRG/checkpoints/model_mixed.ckpt'
    mbatch_size = 16
    device = 'cuda'
    chexbert_metrics = CheXbertMetrics(checkpoint_path, mbatch_size, device)

    # results = json.loads(open('/home/xmli/hl_yang/TOG_RRG/results/MLLM/postprocess_rl/mimic_cxr_trainf_1.json').read())
    # results = json.loads(open('/home/xmli/hl_yang/TOG_RRG/results/MLLM/postprocess_rl/mimic_cxr_trainf_1_reason.json').read())
    # results = json.loads(open('/home/xmli/hl_yang/TOG_RRG/results/MLLM/postprocess_rl/mimic_cxr_trainf_1_reason_rl_2.json').read())
    # gts = [my_pre_caption(r['gt_answer']) for r in results]
    # res = [my_pre_caption(r['pred_answer'], 300) for r in results]

    # scores, f1 = chexbert_metrics.compute(gts, res)
    # for k, v in scores.items():
    #     print(f'{k}: {v}')


    results_oo = json.loads(open('/home/xmli/hl_yang/TOG_RRG/results/MLLM/mimic_cxr_trainf_1.json').read())
    results_o = json.loads(open('/home/xmli/hl_yang/TOG_RRG/results/MLLM/mimic_cxr_trainf_1_reason.json').read())
    # gts = [clean_report_mimic_cxr(r['gt_answer']) for r in results][:500]
    # res = [clean_report_mimic_cxr(r['pred_answer']) for r in results][:500]  

    # results = json.loads(open('/home/xmli/hl_yang/TOG_RRG/results/MLLM/mimic_cxr_trainf_1_reason_2.json').read())
    results = json.loads(open('/home/xmli/hl_yang/TOG_RRG/results/MLLM/mimic_cxr_trainf_1_reason_rl_2.json').read())
    outputs = []
    gts = []
    res = []

    gts_o = []
    res_o = []
    gts_oo = []
    res_oo = []
    for i in range(len(results)):
        re = results[i]['pred_answer']
        q_id = results[i]['question_id']

        # # # find the result_o with the same question_id
        result_o = [r for r in results_o if r['question_id'] == q_id][0]
        gt_o = result_o['gt_answer']
        re_o = result_o['pred_answer']
        re_o_ = result_o['pred_answer']
        if "Step 5" not in re or 'findings' not in re_o:
            continue
        re_o = re_o.split('the findings are: ')[1]
        re_o = 'Findings: ' + re_o
        re_o = my_pre_caption(re_o, max_words=300)

        if "Step 5" not in re or 'findings' not in re:
            continue
        else:
            re_ = re.split('the findings are: ')[1]
            re_ = 'Findings: ' + re_
            re_length = len(re_.split(' '))
            if re_length > 500:
                continue
            re_ = my_pre_caption(re_, max_words=300)
            res.append(re_)
            gts_oo.append(my_pre_caption(results_oo[i]['gt_answer']))
            res_oo.append(my_pre_caption(results_oo[i]['pred_answer']))
            gts_o.append(gt_o)
            res_o.append(re_o)
            gts.append(my_pre_caption(results[i]['gt_answer']))
            outputs.append({
                'question_id': results[i]['question_id'],
                'pred_answer_reason_rl': re,
                'pred_answer_reason': re_o_,
                'pred_answer': my_pre_caption(results_oo[i]['pred_answer']),
                'gt_answer': results[i]['gt_answer']
            })

        # if i > 220:
        #     break

    scores, f1 = chexbert_metrics.compute(gts, res)
    for k, v in scores.items():
        print(f'{k}: {v}')

    scores_o, f1_o = chexbert_metrics.compute(gts_o, res_o)
    for k, v in scores_o.items():
        print(f'{k}: {v}')

    scores_oo, f1_oo = chexbert_metrics.compute(gts_oo, res_oo)
    for k, v in scores_oo.items():
        print(f'{k}: {v}')

    for i in range(len(outputs)):
        outputs[i]['f1_reason_rl'] = f1[i]
        outputs[i]['f1_reason'] = f1_o[i]
        outputs[i]['f1'] = f1_oo[i]

    # save the outputs to csv file
    df = pd.DataFrame(outputs)
    df.to_csv('/home/xmli/hl_yang/TOG_RRG/results/MLLM/mimic_cxr_trainf_1_reason_rl_2.csv', index=False)


    # # save outputs to json file
    # with open('/home/xmli/hl_yang/TOG_RRG/results/MLLM/mimic_cxr_trainf_1_reason_filtered.json', 'w') as f:
    #     json.dump(outputs, f, ensure_ascii=False, indent=4)
        
    # gts = [clean_report_mimic_cxr(gt) for gt in gts]
    # res = [clean_report_mimic_cxr(re) for re in res]

    # results = json.loads(open('/home/xmli/hl_yang/TOG_RRG/results/MLLM/iuxray_train2_4_gmaiv1_3_VEdinov2_dynamic.json').read())
    # gts = [clean_report_iu_xray(r['gt_answer']).lower() for r in results]
    # res = [clean_report_iu_xray(r['pred_answer'].lower()) for r in results] 


    # gts = pd.read_csv('/home/xmli/hl_yang/MRG/PromptMRG/results/promptmrg/20241206/gts_report.csv')['report'].tolist()
    # res = pd.read_csv('/home/xmli/hl_yang/MRG/PromptMRG/results/promptmrg/20241206/res_l7_14label.csv')['report'].tolist()

    # gts = pd.read_csv('/home/xmli/hl_yang/MRG/R2GenCMN/results/iu_xray_1207/gts.csv')
    # # 2nd column is the report
    # gts = gts.iloc[:, 1].tolist()
    # res = pd.read_csv('/home/xmli/hl_yang/MRG/R2GenCMN/results/iu_xray_1207/res.csv')
    # res = res.iloc[:, 1].tolist()

    # gts = pd.read_csv('/home/xmli/hl_yang/TOG_RRG/results/iu_xray/gts.csv')['report'].tolist()
    # gts = pd.read_csv('/home/xmli/hl_yang/TOG_RRG/results/mimic_cxr/gts_maxlenfull.csv')['report'].tolist()
    # gts = json.loads(open('/home/xmli/hl_yang/TOG_RRG/results/mimic_cxr/20240907_shanshan/gts.json').read())

    # res = pd.read_csv('/home/xmli/hl_yang/TOG_RRG/results/mimic_cxr/20240903_softprompt/train/res_add.csv')['report'].tolist()
    # res = pd.read_csv('/home/xmli/hl_yang/TOG_RRG/results/20240606_baseline/test/res_3ex_pos_posneg_posunc_cfe_lb01.csv')['report'].tolist()
    # res = json.loads(open('/home/xmli/hl_yang/TOG_RRG/results/mimic_cxr/20240907_shanshan/res.json').read())

    # res = [re.replace('[sum]', '') for re in res]    # read each line as dict
    # res = [json.loads(q)['text'] for q in open(('/home/xmli/hl_yang/MRG/FoundationModel/LLaVA/playground/data/20240817/mimic_cxr_qa_eval/mimic_cxr_test_answer_hypo_max256.jsonl'), "r")]
    # gts = json.loads(open('/home/xmli/hl_yang/MRG/FoundationModel/LLaVA/playground/data/iu_xray_icl/iu_xray_gt.json').read())
    # gts = [g['report'] for g in gts]

    # res = json.loads(open('/home/xmli/hl_yang/MRG/FoundationModel/LLaVA/playground/data/iu_xray_icl/iu_xray_togrrg.json').read())
    # res = [r['report'] for r in res]
    # res = [json.loads(q)['text'] for q in open(('/home/xmli/hl_yang/MRG/FoundationModel/LLaVA/playground/data/iu_xray_icl/iu_xray_icl_ans.jsonl'), "r")]
    # res = [json.loads(q)['text'] for q in open(('/home/xmli/hl_yang/MRG/FoundationModel/LLaVA/playground/data/iu_xray_qa_eval/iuxray_test_answer_hypo_max256.jsonl'), "r")]
    # res = pd.read_csv('/home/xmli/hl_yang/MRG/FoundationModel/InternVL/Results/20240821_iuxray/10_shot_predict_dignosis.csv')['report'].tolist()
    # res = [r.lower().replace('\"', '').strip() for r in res]
    # res = res[:len_res]
    # gts = gts[:len_res]
    # ann_file = '/home/xmli/hlyang/MRG/PromptMRG_V2/data/mimic_cxr/mimic_annotation_promptmrg.json'
    # ann_test = json.loads(open(ann_file).read())['test']

    # gts_view = []
    # res_view = []
    # ann_file = '/home/xmli/hlyang/MRG/PromptMRG_V2/data/mimic_cxr/mimic_annotation_promptmrg.json'
    # ann_test = json.loads(open(ann_file).read())['test']
    # for i in range(len(ann_test)):
    #     view = ann_test[i]['view']
    #     if view == 3:
    #         gts_view.append(gts[i])
    #         res_view.append(res[i])
        
    
    # iu_xray
    # gts = pd.read_csv('/home/xmli/hlyang/MRG/PromptMRG_V2/results/iu_xray/gts.csv')['report'].tolist()
    # res = pd.read_csv('/home/xmli/hlyang/MRG/PromptMRG_V2/results/iu_xray/baseline/20240609IOSM/res_3ex_pos_posneg_posunc_nofreeze.csv')['report'].tolist()
    


    # # make a new csv file, with the same format as the original csv file and add the new column 'f1'
    # res_save = pd.read_csv('/home/xmli/hl_yang/MRG/PromptMRG_vs/test_res.csv')
    # res_save['f1'] = f1
    # res_save.to_csv('/home/xmli/hl_yang/MRG/PromptMRG_vs/test_res_f1.csv', index=False)


    # json_file = '/home/xmli/hlyang/MRG/PromptMRG_V2/results/iu_xray/baseline/20240614IOSM/test_3ex_posbla_pos_posunc_cfe_lb01_orignialimage.json'
    # ann_file = '/home/xmli/hlyang/MRG/PromptMRG_V2/data/iu_xray/iu_annotation_promptmrg.json'
    # cls_preds = json.loads(open(json_file).read())
    # ann_test = json.loads(open(ann_file).read())
    
    # json_file = '/home/xmli/hlyang/MRG/PromptMRG_V2/results/mimic_cxr/baseline/20240531_onestage/test/test_bs16_cls.json'
    # ann_file = '/home/xmli/hlyang/MRG/PromptMRG_V2/data/mimic_cxr/mimic_annotation_promptmrg.json'
    # cls_preds = json.loads(open(json_file).read())
    # ann_test = json.loads(open(ann_file).read())['test']
    # # print(ann_test[244])

    # gts, res = [], []
    # Acc = 0
    # # print(ann_test[0])
    # for i in range(len(ann_test)):
    #     # view = ann_test[i]['view']
    #     # if view == 3:
    #     ids = ann_test[i]['id']
    #     # if view == 3:
    #     preds_label = np.array(cls_preds[ids])
    #     # preds_label[preds_label == 3] = 1
    #     cls_label = np.array(ann_test[i]['labels'])[:14]

    #     acc = (preds_label == cls_label).mean()
    #     Acc += acc

    #     # cls_label[cls_label == 3] = 1
    #     gts.append(cls_label)
    #     res.append(preds_label)
    
    # scores = chexbert_metrics.compute_label(gts, res, k=1)
    # for k, v in scores.items():
    #     print(f'{k}: {v}')

    # print(f'Acc: {Acc / len(ann_test)}')

    # gts_file = '/home/xmli/hlyang/MRG/PromptMRG_V2/results/mimic_cxr/gts_maxlenfull.csv'
    # ours_file = '/home/xmli/hlyang/MRG/PromptMRG_V2/results/mimic_cxr/20240606_baseline/test/res_3ex_pos_posneg_posunc_lb01.csv'
    # promptmrg_file = '/home/xmli/hlyang/MRG/PromptMRG/results/promptmrg/20240418/res_report.csv'
    # organ_file = '/home/xmli/hlyang/MRG/ORGan/result/20240523_mimic_cxr/res_report.csv'
    # gts = pd.read_csv(gts_file)
    # ours = pd.read_csv(ours_file)
    # promptmrg = pd.read_csv(promptmrg_file)
    # organ = pd.read_csv(organ_file)

    # import ast
    # gts_label = gts['label'].tolist()
    # ours_label = ours['label'].tolist()
    # promptmrg_label = promptmrg['label'].tolist()
    # organ_label = organ['label'].tolist()

    # print(gts_label[244])
    # print(ours_label[244])
    # print(promptmrg_label[244])

    # for i in range(len(gts_label)):
    #     # string to array
    #     gts_label_i = np.array(ast.literal_eval(gts_label[i]))
    #     ours_label_i = np.array(ast.literal_eval(ours_label[i]))
    #     promptmrg_label_i = np.array(ast.literal_eval(promptmrg_label[i]))
    #     organ_label_i = np.array(ast.literal_eval(organ_label[i]))

    #     ours_acc_i = (ours_label_i == gts_label_i).mean()
    #     promptmrg_acc_i = (promptmrg_label_i == gts_label_i).mean()
    #     organ_acc_i = (organ_label_i == gts_label_i).mean()


if __name__ == '__main__':
    main()