"""
Organize the tti dataset in the same way the ultralytics requirements for
training a yolo model have been set.

Tasks:
Grab all frames per video with TTI.
Use the polygon shapes to create bounding boxes needed for this architecture.
"""
# TODO: take previously made splits so that we can compare different models
# with the same training, validation and testing splits.

import os
import shutil as sh
import json
import ast
import pandas as pd
import numpy as np
import cv2


def cv2Rect_to_yolo(rect, frame):
    (x_min, y_min) = (rect[0], rect[1])
    (w, h) = (rect[2], rect[3])
    # calculate maximum pixel
    (x_max, y_max) = (x_min + w, y_min + h)

    # calculate normalized center
    (x_c, y_c) = (((x_max + x_min)/2)/frame.shape[1],
                   ((y_max + y_min)/2)/frame.shape[0])

    # normalize width and height
    (w, h) = w/frame.shape[1], h/frame.shape[0]

    return (x_c, y_c, w, h)


def yolo_to_cv2Rect(rect, frame):
    (x_c, y_c, w, h) = rect
    w = w*frame.shape[1]
    h = h*frame.shape[0]
    x_min = (frame.shape[1]*x_c) - w/2
    y_min = (frame.shape[0]*y_c) - h/2

    return (round(x_min), round(y_min),
            round(w), round(h))


def main(args):
    raw_dir = args.raw_dir
    seg_path = args.seg_path
    out_dir = args.out_dir
    videos_dirs = ['./LC 5 sec clips 30fps']
                # '/cluster/projects/madanigroup/CLAIM/subvideos/raw']
    yaml_dir = './yaml'

    #if args.exclude_non_TTI_frames:
    #    out_dir = os.path.join(base_dir, 'datasets', 'TTI_noTN')
    #elif args.only_confirmed_non_TTI:
    #    out_dir = os.path.join(base_dir, 'datasets', 'TTI_conf')
    #else:
    #    out_dir = os.path.join(base_dir, 'datasets', 'TTI')

    train_dir = os.path.join(out_dir, 'train')
    valid_dir = os.path.join(out_dir, 'valid')
    test_dir = os.path.join(out_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    seg_df = pd.read_csv(seg_path)
    poly_df = seg_df[seg_df['shape'] == 'polygon']

    if not args.video_inclusions is None:
        #TODO: Some videos have no segmentations (despite clearly having 
        # TTI's...). For now these shouldn't be sampled from. But, the five videos
        #  Adnanset-Lc 79-010.mp4, Adnanset-Lc 34-003.mp4, Adnanset-Lc 42-003.mp4,
        # Adnanset-Lc 3-001.mp4, Hokkaidoset-Lc 2-007.mp4 in the file 
        # 'November 13 Videos to be sent to Jaryd (Nov 13).txt'
        # import data_utils as du
        incl_df = pd.read_csv(args.video_inclusions, names=['bns'])
        # incl_df['title'] = incl_df['bns'].apply(du.vidpath2title)
        seg_df = seg_df[seg_df['title'].isin(incl_df['title'])]


    #Split into train, valid and test videos
    if args.split_dir is None:
        seg_df['surgery'] = seg_df['title'].str.split('-').str[0:2].str.join('-')
        seg_df['surgery'] = seg_df['surgery'].str.lower().str.replace(' ', '_')
        # skip videos with no TTI present in case those are incorrect omissions...
        rel_surgs = seg_df[(seg_df['segmentation name'] == 
                            'TTI Free Drawing')]['surgery']
        rel_surgs = np.unique(rel_surgs)
        # randomly split the surgeries up into the splits
        rand_s = np.random.choice(rel_surgs, len(rel_surgs), replace=False)
        splits = {}
        splits['train'] = rand_s[:int(rand_s.shape[0]*0.6)]
        splits['valid'] = rand_s[int(rand_s.shape[0]*0.6):int(rand_s.shape[0]*0.8)]
        splits['test'] = rand_s[int(rand_s.shape[0]*0.8):]
    else:
        msg = f"Copying of previous splits has not been implement."
        raise NotImplementedError(msg)

    # Iterate over the videos since it's easier to reason about
    videos = {}
    videos['train'] = np.unique(seg_df[seg_df['surgery'].isin(splits['train'])]['title'])
    videos['valid'] = np.unique(seg_df[seg_df['surgery'].isin(splits['valid'])]['title'])
    videos['test'] = np.unique(seg_df[seg_df['surgery'].isin(splits['test'])]['title'])
    for sn, vids in videos.items():
        split_dir = os.path.join(out_dir, sn)
        os.makedirs(os.path.join(split_dir, 'images'))
        os.makedirs(os.path.join(split_dir, 'labels'))
        for v in vids:
            print(v)
            sub_df = poly_df[(poly_df['title'] == v)]

            # sample a maximum of pervid_maxsamp tti frames per video
            tti_df = sub_df[sub_df['segmentation name'] == 'TTI Free Drawing']
            tti_frames = np.unique(tti_df['frame_num'])
            if len(tti_frames) > args.pervid_maxsamp:
                samp_tti_frames = np.random.choice(tti_frames,
                                                   args.pervid_maxsamp,
                                                   replace=False)
                samp_tti_df = tti_df[tti_df['frame_num'].isin(samp_tti_frames)]
            else:
                samp_tti_frames = tti_frames
                samp_tti_df = tti_df
            for i, row in samp_tti_df.iterrows():
                frame_name = v + f"-{row['frame_num']}.png"
                print(f'TTI: {frame_name}')
                old_path = os.path.join(raw_dir, frame_name)
                new_path = os.path.join(split_dir, 'images', frame_name)
                sh.copy(old_path, new_path)
                frame = cv2.imread(old_path)
                points = ast.literal_eval(row['points'])
                coord_array = []
                for coords in points.values():
                    coord_array.append((int(coords['x']*frame.shape[1]),
                                        int(coords['y']*frame.shape[0])))
                coord_array = np.array(coord_array,
                                       dtype=np.int32).reshape((-1,1,2))
                rect = cv2.boundingRect(coord_array)
                yolo_rect = cv2Rect_to_yolo(rect, frame)
                with open(os.path.join(split_dir, 'labels',
                                    v + f"-{row['frame_num']}.txt"),
                                    'a') as f:
                    f.write(f"0 {yolo_rect[0]} {yolo_rect[1]}"
                            f" {yolo_rect[2]} {yolo_rect[3]}\n")

            # sample pervid_maxsamp non-tti frames per video
            if args.exclude_non_TTI_frames:
                continue
            # Get frames with a 'No interaction' segmentation
            non_df = sub_df[sub_df['segmentation name'] == 'No Interaction']
            # Exclude frames that have both a 'No Interaction' and 'TTI Free
            # Drawing' segmentation.
            non_frames = set(non_df['frame_num']).difference(set(tti_frames))
            non_frames = np.array(list(non_frames)) # array to sample from it
            # Drop the excluded frames from non_df
            non_df = non_df[non_df['frame_num'].isin(non_frames)]
            print(f"# non_frames: {len(non_frames)}")
            if len(non_frames) > args.pervid_maxsamp:
                samp_non_frames = np.random.choice(non_frames, 3, replace=False)
                samp_non_df = non_df[non_df['frame_num'].isin(samp_non_frames)]
            else:
                samp_non_frames = non_frames
                samp_non_df = non_df

            if len(samp_non_frames) < args.pervid_maxsamp and not args.only_confirmed_non_TTI:
                print('get frames outside of TTI regions?')
                vid_name = v + '.mp4'
                if os.path.exists(os.path.join(videos_dirs[0], vid_name)):
                    vid_path = os.path.join(videos_dirs[0], vid_name)
                else:
                    vid_path = os.path.join(videos_dirs[1], vid_name)

                cap = cv2.VideoCapture(vid_path)
                max_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                tti_range = (tti_df['frame_num'].min(),
                             tti_df['frame_num'].max())
                try:
                    pos_frames = [i for i in range(0, tti_range[0])] + \
                                [i for i in range(tti_range[1]+1, max_frame)]
                except:
                    import pdb
                    pdb.set_trace()
                if pos_frames:
                    print(f"found possible non-tti regions {v}.")
                    print(f"    {tti_range}")
                    num_choose = min(len(pos_frames),
                                     args.pervid_maxsamp - non_df.shape[0])
                    print(f'choose {num_choose} non frames')
                    add_frames = np.random.choice(pos_frames, size=num_choose,
                                                replace=False)
                    for af in add_frames:
                        if af in non_frames:
                            print(f'sampled duplicate frame_num {af}')
                            continue
                        cap.set(cv2.CAP_PROP_POS_FRAMES, af)
                        check, add_frame = cap.read()
                        if not check:
                            import pdb
                            pdb.set_trace()
                            print('problem reading frame')
                        af_path = os.path.join(split_dir, 'images',
                                            v + f"-{af}.png")
                        check = cv2.imwrite(af_path, add_frame)
                        if not check:
                            import pdb
                            pdb.set_trace()
                            print('problem writing frame')
                        print(f"added non-tti frame"
                              f" {os.path.basename(af_path)}")
                        f = open(os.path.join(split_dir, 'labels',
                                            v + f"-{af}.txt"), 'w')
                        f.close()
                else:
                    print(f"no possible non-tti frames {v}")

            for i, row in samp_non_df.iterrows():
                frame_name = v + f"-{row['frame_num']}.png"
                print(f'Non: {frame_name}')
                old_path = os.path.join(raw_dir, frame_name)
                new_path = os.path.join(split_dir, 'images', frame_name)
                sh.copy(old_path, new_path)
                #Write empty set of labels, since they are True Negatives.
                f = open(os.path.join(split_dir, 'labels',
                                    v + f"-{row['frame_num']}.txt"), 'w')
                f.close()

        ## Make an additional split_dir/check that draws all the bounding boxes on
        ## the background frames in split_dir/images and split_dir/labels for easy
        ## visualization.
        # All basenames should be shared between images and labels
        bns = os.listdir(os.path.join(split_dir, 'images'))

        #TODO: implement writing the visualization checks of the bounding boxes
        # on the original background frame.
        check_dir = os.path.join(split_dir, 'check')
        os.makedirs(check_dir, exist_ok=True)
        for bn in bns:
            img_path = os.path.join(split_dir, 'images', bn)
            lbl_path = os.path.join(split_dir, 'labels', bn)
            lbl_path = os.path.splitext(lbl_path)[0] + '.txt'

            frame = cv2.imread(img_path)
            with_bb = frame.copy()
            with open(lbl_path, 'r') as f:
                for l in f.readlines():
                    coords = [float(s) for s in l.split(' ')[1:]]
                    rect = yolo_to_cv2Rect(coords, frame)
                    with_bb = cv2.rectangle(with_bb, rect, (256, 256, 256))
            cv2.imwrite(os.path.join(check_dir, bn), with_bb)

    yaml_bn = f"{os.path.basename(args.out_dir)}.yaml"
    with open(os.path.join(yaml_dir, yaml_bn), 'w') as f:
        f.write(f"path: '{args.out_dir}'\n")
        f.write(f"train: 'train/images'\n")
        f.write(f"val: 'valid/images'\n")
        f.write(f"test: 'test/images'\n\n")
        f.write(f"names:\n")
        f.write(f"  0: 'TTI'\n")

    return None


if __name__ == '__main__':
    base_dir = './'

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', type=str, required=True,
                        help="""
                        Path to directory containing the raw frames.
                        """)
    parser.add_argument('--seg_path', type=str, required=True,
                        help="""
                        Path to csv file with information about all segmentations in
                        the frames.
                        """)
    parser.add_argument('--out_dir', type=str, required=True,
                        help="""
                        Path to directory where the output dataset will be written,
                        and a json of the given arguments. This script will not
                        adjust the given out_dir, so if there is any human
                        legibility wanted in this for ease of access then that
                        must be done manually.
                        """)
    parser.add_argument('--yaml_path', type=str, required=True,
                        help="""
                        Path where the yolo yaml should be written.
                        """)
    parser.add_argument('--split_dir', type=str, default=None,
                        help="""
                        Path to directory that contains at least three csv 
                        files, training_frames.csv, validation_frames.csv, and 
                        test_frames.csv. All frames in the csv seg_path will be
                        split based on their membership of these three files,
                        and if they are in none of them then they will be
                        excluded. Reports about which frames are in none of the
                        splits, and which frames are in the splits but not in
                        seg_path will be written to out_dir.
                        """)
    parser.add_argument('--pervid_maxsamp', type=int, default=3,
                        help="""
                        Per video maximum samples; if there are more annotated 
                        frames then pervid_maxsamp, down sample the number of 
                        annotated frames to this maximum. This applies
                        separately to non-TTI frames and TTI frames.
                        """)
    excl_grp = parser.add_argument_group('exclusions')
    excl_grp.add_argument('--exclude_non_TTI_frames', action='store_true',
                        help="""
                        If given do not sample frames that are outside the labelled
                        TTI. Default is we want the model to be able to distinguish
                        when a TTI is not happening so non-TTI frames are necessary.
                        Assumed exclusive with other arguments.
                        """)
    excl_grp.add_argument('--only_confirmed_non_TTI', action='store_true',
                        help="""
                        If given do not sample frames outside the labelled TTI, but
                        do use the labelled Non-tti frames as provided. Assumed
                        exclusive with other arguments.
                        """)
    excl_grp.add_argument('--video_inclusions', type=str, default=None,
                          help="""
                          Path to a text file of video basenames that are the
                          only videos to include from the segmentations
                          dataframe. If not given then all files are included.
                          Assumes each line is the basename for a single video.
                          """)
    args = parser.parse_args()

    os.makedirs(args.out_dir)
    with open(os.path.join(args.out_dir, 'cli_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    main(args)


