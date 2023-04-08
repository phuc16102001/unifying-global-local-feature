export CUDA_VISIBLE_DEVICES=1
python3 train_e2e.py "test_glip" \
	"/ext_drive/data/soccernet_720p_2fps" \
	-s "results/800MF_GRU_GSM_Integer_AlphaNo_Gamma5" \
	-m "rny008_gsm" \
	-mgpu \
	--learning_rate 1e-3 \
	--num_epochs 150 \
	--start_val_epoch 149 \
	--temporal_arch "gru" \
	--warm_up_epochs 3 \
	--batch_size 8 \
	--clip_len 100 \
	--crop_dim -1 \
	--label_type "one_hot" \
	--num_workers 4 \
	--mixup \
	--alpha -1 \
	--gamma 5 
#	--glip_dir "/ext_drive/data/glip_feat" 

# srun python3 train_e2e.py soccernetv2 soccerNet_outdir -s /cm/archive/kimth1/spot/save_soccernet_p2 -m rny008_gsm -mgpu --num_epochs 50 --batch_size 8 --crop_dim -1 --resume
# add --resume to train from checkpoint
# crop_dim -1 will not crop image
