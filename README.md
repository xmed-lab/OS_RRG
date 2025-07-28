<div align='center'>

<h2><a href="https://github.com/xmed-lab/OS_RRG">OS-RRG: Observation State-Aware Radiology Report Generation With Balanced Diagnosis and Attention Intervention</a></h2>

</div>

## 📋 Overview
[IEEE TNNLS 2025] OS-RRG is a two-stage approach that incorporates a State-aware Balancing Diagnosis (SBD) module to alleviate inter- and intra-class imbalances in medical report generation and employs a State-guided Attention Intervention (SAI) technique to dynamically adjusts focus on key diagnostic features through targeted filtering and enhancement mechanisms.

## 🔨 Installation

Clone this repository and install the required packages:

```shell
git clone https://github.com/xmed-lab/OS_RRG.git
cd OS_RRG

conda create -n osrrg python=3.8
conda activate osrrg
pip install -r requirements.txt
```

## 🍹 Preparation

### Data Acquisition

**MIMIC-CXR**: The images can be downloaded from either [physionet](https://www.physionet.org/content/mimic-cxr-jpg/2.0.0/) or [R2Gen](https://github.com/zhjohnchan/R2Gen). Note that the physionet version requires a license for download. We use the [R2Gen](https://github.com/zhjohnchan/R2Gen) version for both training and evaluation. The annotation file can be downloaded from [Google Drive](https://drive.google.com). Please place all downloaded files under the `data/mimic_cxr/` folder.

**IU-Xray**: Download the images from [R2Gen](https://github.com/zhjohnchan/R2Gen) and the annotation file from [Google Drive](https://drive.google.com/file/d/your_iu_xray_annotation_link). Please place both the images and annotation files under the `data/iu_xray/` folder.

Moreover, you need to download the `chexbert.pth` from [here](https://drive.google.com) for evaluating clinical efficacy and put it under `checkpoints/chexbert/`.

## 🚀 Training

### Two-Stage Training Pipeline

OS_RRG employs a two-stage training approach to align fine-grained visual observations to high-quality medical reports:

#### Stage 1: State-to-Description Alignment Training
This stage focuses on aligning observation states with textual descriptions.

```shell
bash train_step1_Align.sh
```

#### Stage 2: Imbalanced Observation and State Mitigation (SBD)
This stage employs the SBD (State-aware Balancing Diagnosis) module to handle imbalanced observations states.

```shell
bash train_step2_SBD.sh
```

## 📊 Evaluation

### Pre-trained Model Usage

You can directly use our pre-trained models for evaluation:

* **Step 1 - State-to-Description Alignment(https://drive.google.com/drive/folders/your_alignment_weights_folder)** - Put at ./checkpoints/osrrg/
* **Step 2 - Imbalanced Observation and State Mitigation (SBD)(https://drive.google.com/drive/folders/your_sbd_weights_folder)** - Put at ./checkpoints/osrrg/

### Testing Commands

```shell
# For MIMIC-CXR dataset
bash test_mimic.sh

# For IU-XRay dataset
bash test_iuxray.sh
```

## 💙 Acknowledgement

OS_RRG is built upon the [BLIP](https://github.com/salesforce/BLIP), [PromptMRG](https://github.com/jhb86253817/PromptMRG/), and [SADE](https://github.com/Vanint/SADE-AgnosticLT).

## 📄 Citation

If you use this work in your research, please cite:

```bibtex
@ARTICLE{11095809,
  author={Yang, Honglong and Tang, Hui and Song, Shanshan and Li, Xiaomeng},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={OS-RRG: Observation State-Aware Radiology Report Generation With Balanced Diagnosis and Attention Intervention}, 
  year={2025},
  volume={},
  number={},
  pages={1-15},
  keywords={Accuracy;Medical diagnostic imaging;Radiology;Diseases;Telecommunication traffic;MIMICs;Linguistics;Heavily-tailed distribution;Communication switching;Training;Natural language processing;observation state (OS)-aware generation;observation-guided generation (OGG);radiology report generation (RRG)},
  doi={10.1109/TNNLS.2025.3589103}}

```

## 📧 Contact

For questions and issues, please use the GitHub issue tracker or contact [hyangdh@connect.ust.hk]. 