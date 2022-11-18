export CUDA_VISIBLE_DEVICE=0,1
python -m torch.distributed.launch \
--nproc_per_node=2 \
train_seg.py \
--num-worker 8 \
--des-dir \
