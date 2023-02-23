
export CUDA_VISIBLE_DEVICES=0
python3 test_e2e.py "results/BCE_800MF_GRU" \
	"/ext_drive/data/soccernet_720p_2fps" \
	-s "test" \
	--save

# python3 test_e2e.py <model_dir> <frame_dir> -s <split> --save
# default using map metric
# --criterion_key val: use val loss metric
