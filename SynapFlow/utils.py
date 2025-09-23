import os
import json
from glob import glob

import cv2
import pandas as pd
import numpy as np

def generate_coco_from_spine(data_root, split, out_dir, csv_fp=None):
    """
    Generate COCO data from spine dataset.
    """
    annotations = {}
    annotations['type'] = 'instances'
    annotations['images'] = []
    annotations['categories'] = [{"supercategory": "spine",
                                  "name": "spine",
                                  "id": 1}]
    annotations['annotations'] = []
    if csv_fp is not None:
        annotation_file = os.path.join(out_dir, os.path.basename(csv_fp).replace('.csv', '.json'))
    else:
        annotation_file = os.path.join(out_dir, f'{split}.json')
    os.makedirs(out_dir, exist_ok=True)
    # annotation_file = os.path.join(data_root, f'annotations/{split}.json')
    # os.makedirs(os.path.join(data_root, 'annotations'), exist_ok=True)

    # IMAGES
    imgs_list_dir = sorted([os.path.basename(f) for f in glob(os.path.join(data_root, split, '*.png'))])
    seqs_names = sorted(np.unique([img.split('_layer')[0] for img in imgs_list_dir]))
    seqs_lengths = [len([img for img in imgs_list_dir if seq in img]) for seq in seqs_names]
    first_frame_image_id = 0
    for i, img in enumerate(sorted(imgs_list_dir)):
        im = cv2.imread(os.path.join(data_root, split, img))
        h, w, _ = im.shape
        seq_name = img.split('_layer')[0]
        if i > 0 and seq_name != imgs_list_dir[i-1].split('_layer')[0]:
            first_frame_image_id = i

        annotations['images'].append({
            "file_name": img,
            "height": h,
            "width": w,
            "id": i, 
            "first_frame_image_id": first_frame_image_id,
            "seq_length": seqs_lengths[seqs_names.index(seq_name)],
            "frame_id": i - first_frame_image_id,
        })

    # GT
    annotation_id = 0
    img_file_name_to_id = {
        os.path.splitext(img_dict['file_name'])[0]: img_dict['id']
        for img_dict in annotations['images']}

    # for split in ['train', 'valid', 'test']:
    #     if split not in split:
    #         continue
    if csv_fp is not None:
        # csv_file = os.path.join(data_root, f'annotations/{split}.csv')
        df = pd.read_csv(csv_fp)

        xmins = np.around(df['xmin']).astype(int).values
        ymins = np.around(df['ymin'].values).astype(int)
        xmaxs = np.around(df['xmax'].values).astype(int)
        ymaxs = np.around(df['ymax'].values).astype(int)
        bbox_widths = xmaxs - xmins
        bbox_heights = ymaxs - ymins
        areas = (bbox_widths * bbox_heights).astype(int).tolist()
        if 'id' not in df.columns:
            df['id'] = np.arange(len(df))
        track_ids = df['id'].values

        gtboxes = np.array([xmins, ymins, bbox_widths, bbox_heights]).T # (x, y, width, height)
        for i in range(len(gtboxes)):
            visibility = 1.0
            filename = os.path.basename(df['filename'].values[i])[:-4]
            if filename not in img_file_name_to_id:
                continue

            annotation = {
                "id": annotation_id,
                "bbox": gtboxes[i].tolist(),
                "image_id": img_file_name_to_id[filename],
                "segmentation": [],
                "ignore": 0,
                "visibility": visibility,
                "area": areas[i],
                "iscrowd": 0,
                "category_id": annotations['categories'][0]['id'],
                "seq": filename.split('_layer')[0],
                "track_id": int(track_ids[i]),
            }

            annotation_id += 1
            annotations['annotations'].append(annotation)

    # add "sequences" and "frame_range"
    annotations['sequences'] = seqs_names

    frame_range = {'start': 0.0, 'end': 1.0}
    annotations['frame_range'] = frame_range

    with open(annotation_file, 'w') as anno_file:
        json.dump(annotations, anno_file, indent=4)
    
    return annotations