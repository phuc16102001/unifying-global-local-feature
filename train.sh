export CUDA_VISIBLE_DEVICES=0
python3 train_e2e.py "soccernet" \
	"/ext_drive/data/soccernet_720p_2fps" \
	-s "results/BCE_800MF_GRU" \
	-m "rny008_gsm" \
	-mgpu \
	--learning_rate 0.0001 \
	--num_epochs 150 \
	--start_val_epoch 149 \
	--temporal_arch "gru" \
	--warm_up_epochs 3 \
	--batch_size 6 \
	--crop_dim -1 \
	--label_type "one_hot" \
	--num_workers 8 \

# srun python3 train_e2e.py soccernetv2 soccerNet_outdir -s /cm/archive/kimth1/spot/save_soccernet_p2 -m rny008_gsm -mgpu --num_epochs 50 --batch_size 8 --crop_dim -1 --resume
# add --resume to train from checkpoint
# crop_dim -1 will not crop image
