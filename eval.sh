python3 eval_soccernetv2.py "/Users/JLJXDXCFK7/Prediction/pred-test.149.recall.json" \
	--split "test" \
	--eval_dir "/Users/JLJXDXCFK7/Prediction/output" \
	--soccernet_path "/Users/JLJXDXCFK7/Data/soccernet-label" \
	--nms_window 2 \
	--filter_score 0.05 \
	--allow_remove
