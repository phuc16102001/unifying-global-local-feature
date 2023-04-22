
export CUDA_VISIBLE_DEVICES=0
python3 eval_soccernetv2.py "/Users/JLJXDXCFK7/Prediction/pred-test.149.recall.json" \
	--split "test" \
	--eval_dir "/Users/JLJXDXCFK7/Prediction/output" \
	--soccernet_path "/Users/JLJXDXCFK7/Data/soccernet-label" \
	--nms_window 2 \
	--filter_score 0.05 \
	--allow_remove

# python3 test_e2e.py <model_dir> <frame_dir> -s <split> --save
# default using map metric
# --criterion_key val: use val loss metric
