export CUDA_VISIBLE_DEVICES=0


python train_HybridCBM.py \
    --dataset HAM10000 \
    --data_root /kaggle/input/datasets/loinguyen57/ham10000-merge/ham10000_images \
    --exp_root /kaggle/working/experiments/HAM10000_run_1 \
    --device cuda:0 \
    --clip_model ViT-L/14 \
    --translator_path /kaggle/input/models/loinguyen57/translators/pytorch/default/1/translators-best.pt \
    \
    --weight_init_method rand \
    --train_mode joint \
    --scale 0.1 \
    --num_class 7 \
    --num_static_concept 50 \
    --num_dynamic_concept 50 \
    --concept_select_fn submodular \
    --submodular_weights 1e7 0.1\
    \
    --batch_size 128 \
    --num_workers 4 \
    --use_img_features \
    --max_epochs 1 \
    --lr 0.001 \
    --concept_lr 0.001 \
    --n_shots all \
    \
    --lambda_l1 0.001 \
    --lambda_discri 1.0 \
    --lambda_discri_alpha 0.5 \
    --lambda_discri_beta 0.5 \
    --lambda_ort 0.1 \
    --lambda_align 0.01