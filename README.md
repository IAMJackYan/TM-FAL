# TM-FAL
official code of MICCAI2025: TM-FAL

1. Install

`pip install -r requirements.txt`

2. Prepare datasets

Download and prepare Fed-ISIC dataset following [FEAL](https://github.com/JiayiChen815/FEAL)

3. Train

`CUDA_VISIBLE_DEVICES=0 python main_cls_ours.py --dataset FedISIC --al_method ours --query_model both --query_ratio 0 --budget 500 --al_round 5 --max_round 100 --batch_size 32 --base_lr 5e-4 --kl_weight 1e-2 --display_freq 20`
