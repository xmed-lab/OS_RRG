CUDA_VISIBLE_DEVICES=1 python main_test.py \
    --n_gpu 1 \
    --image_dir your_image_dir \
    --ann_path ./data/mimic_cxr/mimiccxr_annotation.json \
    --dataset_name mimic_cxr \
    --gen_max_len 150 \
    --gen_min_len 100 \
    --batch_size 32 \
    --save_dir ./results/mimic_cxr/infer \
    --seed 456789 \
    --beam_size 3 \
    --image_size 224 \
    --load_pretrained ./checkpoints/osrrg/osrrg_alignment.pth \
    --load_sbd_pretrained ./checkpoints/osrrg/osrrg_sbd.pth
   