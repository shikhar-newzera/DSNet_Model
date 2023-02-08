import cv2
import numpy as np
import torch

from helpers import init_helper, vsumm_helper, bbox_helper, video_helper
from modules.model_zoo import get_model
from moviepy.editor import *
import os


def main(source_video, index, folder_name):
    args = init_helper.get_arguments()

    # load model
    print('Loading DSNet model ...')
    model = get_model(args.model, **vars(args))
    model = model.eval().to(args.device)
    state_dict = torch.load(args.ckpt_path,
                            map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)

    # load video
    print('Preprocessing source video ...')
    video_proc = video_helper.VideoPreprocessor(args.sample_rate)
    n_frames, seq, cps, nfps, picks = video_proc.run(source_video)
    seq_len = len(seq)

    print('Predicting summary ...')
    with torch.no_grad():
        seq_torch = torch.from_numpy(seq).unsqueeze(0).to(args.device)

        pred_cls, pred_bboxes = model.predict(seq_torch)

        pred_bboxes = np.clip(pred_bboxes, 0, seq_len).round().astype(np.int32)

        pred_cls, pred_bboxes = bbox_helper.nms(pred_cls, pred_bboxes, args.nms_thresh)
        pred_summ = vsumm_helper.bbox2summary(
            seq_len, pred_cls, pred_bboxes, cps, n_frames, nfps, picks)

    print('Writing summary video ...')

    # load original video
    cap = cv2.VideoCapture(source_video)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # create summary video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    frame_idx = 0
    curr_st = -1
    curr_st_time = -1
    curr_en = -1
    prev = -1
    prev_time = -1
    hello = True
    end_time = 0
    timestamps = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if pred_summ[frame_idx]:
            if curr_st == -1:
              curr_st = frame_idx
              curr_st_time = cap.get(cv2.CAP_PROP_POS_MSEC)
            elif frame_idx != prev+1:
              curr_en = prev
              print('frames chosen', curr_st, curr_en, ' timeframe',curr_st_time, prev_time)
              timestamps.append([curr_st_time, prev_time])
              curr_st = -1

            prev = frame_idx
            prev_time = cap.get(cv2.CAP_PROP_POS_MSEC)
            end_time = max(prev_time, end_time)

        frame_idx += 1
    if(curr_st != -1):
        print('frames chosen', curr_st, curr_en, ' timeframe',curr_st_time, end_time)
        timestamps.append([curr_st_time, end_time])
    
    print(timestamps)
    clip = VideoFileClip(source_video)
    clips = []
    for i in range(len(timestamps)):
        clips.append(clip.subclip(timestamps[i][0]/1000, timestamps[i][1]/1000))
    final = concatenate_videoclips(clips)
    final.write_videofile(folder_name + "/outputClip_" + str(index) + ".mp4")

    cap.release()


if __name__ == '__main__':
    args = init_helper.get_arguments()
    # dataset_used = args.ckpt-path.split('/')[-1].split('.')[0]
    dataset_used = "youtube"
    folder_name = args.source.split('/')[-1].split('.')[0] + "_" + args.model + "_" + dataset_used
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    print(folder_name)
    cap_duration = cv2.VideoCapture(args.source)
    fps_cap_duration = cap_duration.get(cv2.CAP_PROP_FPS)
    frame_count_cap_duration = int(cap_duration.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_val = frame_count_cap_duration/fps_cap_duration
    print('video duration', duration_val)
    cap_duration.release()

    video_chunk_counts = 10
    video_chunks = []
    chunk_duration = duration_val/video_chunk_counts
    st_mark = 0.0

    clip1 = VideoFileClip(args.source)
    for i in range(video_chunk_counts):
        if i == video_chunk_counts - 1:
            video_clip_extracted = clip1.subclip(st_mark)
            video_clip_extracted.write_videofile("./input_" + str(i+1)+".mp4")
            main("./input_" + str(i+1)+".mp4",i+1,folder_name)
        else:
            video_clip_extracted = clip1.subclip(st_mark, st_mark + chunk_duration)
            st_mark = st_mark + chunk_duration
            video_clip_extracted.write_videofile("./input_" + str(i+1)+".mp4")
            main("./input_" + str(i+1)+".mp4",i+1,folder_name)
"""
for split
python /home/shikhar/mansion/DSNet/DSNet/src/make_split.py --dataset /home/shikhar/mansion/DSNet/DSNet/datasets/eccv16_dataset_youtube_google_pool5.h5 --train-ratio 0.67 --save-path /home/shikhar/mansion/DSNet/DSNet/splits/youtube.yml
python /home/shikhar/mansion/DSNet/DSNet/src/make_split.py --dataset /home/shikhar/mansion/DSNet/DSNet/datasets/eccv16_dataset_ovp_google_pool5.h5 --train-ratio 0.67 --save-path /home/shikhar/mansion/DSNet/DSNet/splits/ovp.yml

for training
python /home/shikhar/mansion/DSNet/DSNet/src/train.py anchor-based --model-dir /home/shikhar/mansion/DSNet/DSNet/models/ab_youtube --splits /home/shikhar/mansion/DSNet/DSNet/splits/youtube.yml
python /home/shikhar/mansion/DSNet/DSNet/src/train.py anchor-based --model-dir /home/shikhar/mansion/DSNet/DSNet/models/ab_ovp --splits /home/shikhar/mansion/DSNet/DSNet/splits/ovp.yml
"""