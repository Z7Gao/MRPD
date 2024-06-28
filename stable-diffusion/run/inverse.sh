export CUDA_VISIBLE_DEVICES='0'
python scripts/inverse.py \
    --task_config='configs/super_resolution_config_psld.yaml' \
    --outdir='outputs/psld-samples-sr' \
    --ddim_step=50