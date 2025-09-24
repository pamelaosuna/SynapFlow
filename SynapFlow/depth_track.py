import os
import json
from glob import glob
import argparse

import torch
import pandas as pd
import numpy as np
import cv2

from SynapFlow.utils import generate_coco_from_spine

from SynapFlow.depth_tracking.tracking_manager import TrackingManager
from SynapFlow.depth_tracking.utils import (
    generalized_box_iou,
    hungarian_matching,
    draw_boxes_and_save,
    extract_embeddings
)
from SynapFlow.depth_tracking.model import SiameseNetwork

COLS = ['id', 'filename', 'width', 'height', 'xmin', 'ymin', 'xmax', 'ymax', 'class', 'score']

def compute_cost_matrix(
        boxes_t1: torch.Tensor, 
        boxes_t2: torch.Tensor, 
        embeddings_t1: torch.Tensor, 
        embeddings_t2: torch.Tensor,
        lambdas: dict
        ) -> np.ndarray:
    c_giou = -generalized_box_iou(boxes_t1, boxes_t2).to('cpu')
    cost_matrix = lambdas['spatial'] * c_giou

    if lambdas['appearance'] > 0.0:
        c_app = torch.cdist(embeddings_t1, embeddings_t2, p=2).to('cpu')
        cost_matrix += lambdas['appearance'] * c_app

    return cost_matrix.cpu().detach().numpy()

def track_in_depth(
        df_all: pd.DataFrame, 
        img_info: dict, 
        img_dir: str, 
        outdir: str, 
        thresh: float,
        lambdas: dict,
        model: SiameseNetwork = None,
        config: dict = None,
        device: torch.device = torch.device('cpu'),
        draw: bool = False,
        ):
    if lambdas['appearance'] > 0.0:
        patch_size = config['dataset']['patch_size']
    
    for sn in img_info['sequences']:
        df = df_all[df_all['filename'].str.contains(sn)]

        if os.path.exists(os.path.join(outdir, "{sn}.csv")):
            continue

        if len(df) == 0:
            print(f"No detections found for {sn}")
            # still create the csv file
            df_stack = pd.DataFrame(columns=COLS)
            df_stack.to_csv(os.path.join(outdir, f"{sn}.csv"), index=False)
            continue
            
        df = df.reset_index(drop=True)
        df['filename'] = [os.path.basename(f) for f in df['filename'].values]

        images_in_stack = [im for im in img_info['images'] if sn in im['file_name']]

        tracker = TrackingManager(inactive_wait=1)
        df_stack = pd.DataFrame(columns=COLS)

        for i_im in range(1, len(images_in_stack)):
            act_tracks = tracker.get_active_tracks()

            prev_im = images_in_stack[i_im-1]
            im = images_in_stack[i_im]

            df_img = df[df['filename'] == im['file_name']].copy()
            df_prev_img = df[df['filename'] == prev_im['file_name']].copy()

            if len(df_prev_img) == 0 or len(df_img) == 0:
                continue

            if len(act_tracks) == 0:
                boxes_prev_orig = df_prev_img[['xmin', 'ymin', 'xmax', 'ymax']].values
                w, h = prev_im['width'], prev_im['height']
                boxes_prev = boxes_prev_orig.copy() / np.array([w, h, w, h])
                boxes_prev = torch.as_tensor(boxes_prev.astype(np.float32))

                if lambdas['appearance'] > 0.0:
                    emb_prev = extract_embeddings(
                        img_dir, prev_im, patch_size, boxes_prev_orig, model, device
                        )
                else:
                    emb_prev = [None] * len(boxes_prev)
            else:
                boxes_prev_orig = boxes_orig

                if lambdas['appearance'] > 0.0:
                    emb_prev, boxes_prev = tracker.get_emb_box_active_tracks()
                else:
                    boxes_prev = tracker.get_box_active_tracks()

            # get the bounding boxes
            w, h = prev_im['width'], prev_im['height']

            boxes_orig = df_img[['xmin', 'ymin', 'xmax', 'ymax']].values
            boxes = boxes_orig.copy() / np.array([w, h, w, h])

            # convert boxes to torch tensors
            boxes = torch.as_tensor(boxes.astype(np.float32))

            if lambdas['appearance'] > 0.0:
                emb = extract_embeddings(
                    img_dir, im, patch_size, boxes_orig, model, device
                    )
            else:
                emb = [None] * len(boxes)

            cost_matrix = compute_cost_matrix(
                boxes_prev,
                boxes,
                emb_prev,
                emb,
                lambdas
                )
            matches = hungarian_matching(cost_matrix)

            tracker.update_tracks(
                matches, 
                cost_matrix,
                emb_prev, 
                boxes_prev,
                emb, 
                boxes,
                prev_im['frame_id'], 
                im['frame_id'],
                threshold=thresh,
            )

            if i_im == 1 or len(df_stack) == 0:
                boxes_idx_to_id_prev = tracker.get_boxes_idx_to_id(prev_im['frame_id'])
                if draw:
                    img_prev = cv2.imread(os.path.join(img_dir, prev_im['file_name']))

                draw_boxes_and_save(
                    img_prev,
                    boxes_prev_orig,
                    boxes_idx_to_id_prev,
                    os.path.join(outdir, 'images', prev_im['file_name'])
                    )
                
                df_prev_img['id'] = [boxes_idx_to_id_prev[i] for i in range(len(boxes_prev_orig))]
                df_stack = pd.concat([df_stack, df_prev_img])
            
            boxes_idx_to_id = tracker.get_boxes_idx_to_id(im['frame_id'])
            if draw:
                img = cv2.imread(os.path.join(img_dir, im['file_name']))

                draw_boxes_and_save(
                    img, 
                    boxes_orig, 
                    boxes_idx_to_id,
                    os.path.join(outdir, 'images', im['file_name'])
                    )
            df_img['id'] = [boxes_idx_to_id[i] for i in range(len(boxes_orig))]
            df_stack = pd.concat([df_stack, df_img])

        df_stack = df_stack.reset_index(drop=True)
        df_stack.to_csv(os.path.join(outdir, f"{sn}.csv"), index=False)

def run_track_in_depth(
        input_dir: str, 
        out_dir: str, 
        img_dir: str, 
        det_thresh: float, 
        track_thresh: float, 
        sp_cost: float,
        app_cost: float,
        checkpoint_dir: str,
        draw: bool
        ):
    
    # load the image information
    img_info = generate_coco_from_spine(
        data_root=os.path.dirname(img_dir),
        split=os.path.basename(img_dir),
        out_dir=img_dir,
    )
    
    os.makedirs(os.path.join(out_dir), exist_ok=True)
    if draw:
        os.makedirs(os.path.join(out_dir, 'images'), exist_ok=True)

    # load the detections
    df_list = [pd.read_csv(f) for f in sorted(glob(f"{input_dir}/*.csv"))]
    df_list = [tmp_df for tmp_df in df_list if len(tmp_df) > 0]
    df_all = pd.concat(df_list)

    # filter out low-score detecions
    df_all = df_all[df_all['score'] >= det_thresh]
    df_all = df_all.reset_index(drop=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if app_cost > 0.0:
        model = SiameseNetwork(contra_loss=True).to(device)
        model.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'best_model.pth'), map_location=device))
        model.eval()

        config = json.load(open(os.path.join(checkpoint_dir, 'config.json')))
    else:
        model = None
        config = None
    
    lambdas = {'spatial': sp_cost, 'appearance': app_cost}

    track_in_depth(
        df_all,
        img_info,
        img_dir,
        out_dir,
        track_thresh,
        lambdas,
        model,
        config,
        device,
        draw
    )

def run_track_in_depth_args(args):
    run_track_in_depth(
        args.input_dir,
        args.out_dir,
        args.img_dir,
        args.det_thresh,
        args.track_thresh,
        args.sp_cost,
        args.app_cost,
        args.checkpoint_dir,
        args.draw
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Track detected spines across depth using bounding box IoU"
        )
    parser.add_argument("--input_dir", type=str, required=True,
        help="Directory containing the csv files with detections")
    parser.add_argument("--out_dir", type=str, required=True,
        help="Directory to save the depth-tracked files and images with drawn boxes")
    parser.add_argument("--img_dir", type=str, required=True,
        help="Directory containing the preprocessed images (from which to extract patches).")
    parser.add_argument("--det_thresh", type=float, default=0.5,
        help="Minimum score to consider a detection valid.")
    parser.add_argument("--track_thresh", type=float, default=0.0,
        help="Threshold for the cost matrix to consider a match valid.")
    parser.add_argument("--draw", action='store_true',
        help="Whether to draw the boxes on the images and save them.")
    # spatial cost, appearance cost weights, checkpoint dir
    parser.add_argument("--sp_cost", type=float, default=1.0,
        help="Weight for the spatial cost in the cost matrix.")
    parser.add_argument("--app_cost", type=float, default=0.0,
        help="Weight for the appearance cost in the cost matrix.")
    parser.add_argument("--checkpoint_dir", type=str,
        help="Directory containing the trained model for encoding appearance features.")
    args = parser.parse_args()
    
    run_track_in_depth(
        args.input_dir,
        args.out_dir,
        args.img_dir,
        args.det_thresh,
        args.track_thresh,
        args.sp_cost,
        args.app_cost,
        args.checkpoint_dir,
        args.draw
    )