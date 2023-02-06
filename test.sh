
export CUDA_VISIBLE_DEVICES=3
python3 test_e2e.py "save_checkpoints_of_author/soccer_challenge_rny008gsm_gru_rgb" \
	"/extdrive/data/soccernet_720p_2fps" \
	-s challenge \
	--save

# python3 test_e2e.py <model_dir> <frame_dir> -s <split> --save
# default using map metric
# --criterion_key val: use val loss metric
