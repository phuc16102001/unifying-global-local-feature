Dataset size is sum all frames of dataset
clip_len is length of window
train:
val:
test:
challenge:
dataset size = 6328323 frames = EPOCH_NUM_FRAMES
clip_len = 100 frames
num_window = 6328323 // 100 = 63283 windows
batch size = 32 windows

nohup python3 -u frames_as_jpg_soccernet.py /extdrive/data/soccernet_224p -o soccernet_224p_2fps > logs/extract_frames_log.txt &
python3 frames_as_jpg_soccernet.py /extdrive/data/soccernet_224 -o soccernet_224p_5fps --sample_fps 5
nohup bash train.sh > logs/train_protocol3_log.txt &
nohup bash test.sh > logs/test_protocol3_log.txt &