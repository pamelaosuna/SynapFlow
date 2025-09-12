import os
from glob import glob
import argparse

import numpy as np
import pandas as pd

from skimage import io
from scipy.optimize import linear_sum_assignment
import cv2

import matplotlib

from SynapFlow.time_tracking.utils import (
    comp_best_shift_z,
    correspondences_in_xy,
    correspondences_in_z
    )

from SynapFlow.depth_tracking.utils import generalized_box_iou
import torch

class SpinesVolumeMatcher:
    def __init__(
        self,
        fp_t1: str,
        fp_t2: str,
        mip_t1: str,
        mip_t2: str,
        corres_dir: str,
        img_dir: str,
        out_dir: str,
        lambdas: dict,
        thresh: float
        ):
        self.fp_t1 = fp_t1
        self.fp_t2 = fp_t2
        self.mip_t1 = mip_t1
        self.mip_t2 = mip_t2
        self.corres_dir = corres_dir
        self.img_dir = img_dir
        self.out_dir = out_dir
        self.lambdas = lambdas
        self.thresh = thresh

        self.best_shift_z = None
        self.img_width = None
        self.img_height = None

    def load_corres(self):
        corres_t1_fixed_fp = os.path.join(
            self.corres_dir, f"fixed={os.path.basename(self.mip_t1).replace('.png', '')}_moving={os.path.basename(self.mip_t2).replace('.png', '')}_dense.npy"
            )
        corres_t2_fixed_fp = os.path.join(
            self.corres_dir, f"fixed={os.path.basename(self.mip_t2).replace('.png', '')}_moving={os.path.basename(self.mip_t1).replace('.png', '')}_dense.npy"
            )
        
        corres_t1_fixed = np.load(corres_t1_fixed_fp, allow_pickle=True)['corres']
        corres_t2_fixed = np.load(corres_t2_fixed_fp, allow_pickle=True)['corres']

        return corres_t1_fixed, corres_t2_fixed

    def load_dfs(self):
        df_t1 = pd.read_csv(self.fp_t1)
        df_t2 = pd.read_csv(self.fp_t2)

        return df_t1, df_t2
    
    def get_out_of_bounds_info(self):
        return self.oob_t1_fixed, self.oob_t2_fixed

    def time_track(self):
        df_t1, df_t2 = self.load_dfs()

        corres_t1_fixed, corres_t2_fixed = self.load_corres()

        self.vol_name_t1 = os.path.basename(self.fp_t1).replace('.csv', '')
        self.vol_name_t2 = os.path.basename(self.fp_t2).replace('.csv', '')

        # Get image filepaths for each timepoint
        self.img_files_t1 = sorted(glob(os.path.join(self.img_dir, self.vol_name_t1 + '_layer*.png')))
        self.img_files_t2 = sorted(glob(os.path.join(self.img_dir, self.vol_name_t2 + '_layer*.png')))

        nlayers_t1 = len(self.img_files_t1)
        nlayers_t2 = len(self.img_files_t2)

        fixed_vol = np.array([io.imread(f) for f in self.img_files_t1])
        moving_vol = np.array([io.imread(f) for f in self.img_files_t2])

        self.best_shift_z = -6 # comp_best_shift_z(fixed_vol, moving_vol) # -8 or -6 for test case

        # Clear fixed_vol and moving_vol once they have been used
        del fixed_vol
        del moving_vol

        self.img_width = df_t1.iloc[0]['width']
        self.img_height = df_t1.iloc[0]['height']

        medboxes_t1, medboxes_t2, oob_xy_t1_fixed, oob_xy_t2_fixed = correspondences_in_xy(
            df_t1,
            df_t2,
            corres_t1_fixed,
            corres_t2_fixed,
            (self.img_height, self.img_width)
        )

        zmin_max_t1, zmin_max_t2, oob_depth_t1_fixed, oob_depth_t2_fixed = correspondences_in_z(
            self.best_shift_z,
            df_t1,
            df_t2,
            nlayers_t1-1, # because layers are 0-indexed
            nlayers_t2-1
        )

        self.oob_t1_fixed = oob_xy_t1_fixed.union(oob_depth_t1_fixed)
        self.oob_t2_fixed = oob_xy_t2_fixed.union(oob_depth_t2_fixed)

        cost_matrix = comp_cost_matrix(
            medboxes_t1,
            medboxes_t2,
            zmin_max_t1,
            zmin_max_t2,
            self.lambdas
        )

        ids_t1 = sorted(df_t1['id'].unique())
        ids_t2 = sorted(df_t2['id'].unique())

        matches = match_from_cost_matrix(
            cost_matrix,
            ids_t1,
            ids_t2,
            self.thresh
        )

        df_t2['id'] = df_t2['id'].map(matches)
        
        self.df_t1 = df_t1
        self.df_t2 = df_t2

        return df_t2
    
    def draw_ids_in_tp(self, tp2: bool, out_dir: str):
        cmap = matplotlib.cm.get_cmap('tab20')
        colors = [(int(cmap(i)[0]*255), int(cmap(i)[1]*255), int(cmap(i)[2]*255)) for i in range(20)]

        if tp2:
            img_files = self.img_files_t2
            df = self.df_t2
        else:
            img_files = self.img_files_t1
            df = self.df_t1

        for img_fp in img_files:
            img = io.imread(img_fp)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            tmp_df = df[df['filename'].str.contains(os.path.basename(img_fp))]
            
            boxes = tmp_df[['xmin', 'ymin', 'xmax', 'ymax']].values
            ids = tmp_df['id'].values

            for id, box in zip(ids, boxes):
                color = colors[int(id)%20]
                cv2.rectangle(
                    img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 1
                    )
                cv2.putText(
                    img, str(id), (int(box[0])+2, int(box[1])-2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1
                    )
            
            out_fp = os.path.join(out_dir, os.path.basename(img_fp))
            cv2.imwrite(out_fp, img)

def cost_of_match(
        box_t1,
        box_t2,
        zminmax_t1,
        zminmax_t2,
        lambdas: dict):
    if np.all(box_t1 == -1) or np.all(box_t2 == -1):
        return 10
    
    if zminmax_t1[0] > zminmax_t2[1] or zminmax_t2[0] > zminmax_t1[1]:
        # Distance between the two depth ranges
        z_cost = max(zminmax_t1[0] - zminmax_t2[1], zminmax_t2[0] - zminmax_t1[1])
    else:
        # IoM in depth
        inter = np.intersect1d(
            np.arange(zminmax_t1[0], zminmax_t1[1]+1),
            np.arange(zminmax_t2[0], zminmax_t2[1]+1)
            ).shape[0]
        min_len = min(zminmax_t1[1]-zminmax_t1[0], zminmax_t2[1]-zminmax_t2[0]) + 1
        z_cost = 1 - inter / min_len
    
    box_t1 = torch.tensor(box_t1).unsqueeze(0)
    box_t2 = torch.tensor(box_t2).unsqueeze(0)
    giou_cost = -generalized_box_iou(box_t1, box_t2).item()

    total_cost = lambdas['giou_cost'] * giou_cost + lambdas['depth_cost'] * z_cost

    return total_cost
    
def comp_cost_matrix(
        medboxes_t1: dict,
        medboxes_t2: dict,
        zmin_max_t1: dict,
        zmin_max_t2: dict,
        lambdas: dict
        ) -> np.ndarray:
    """
    Compute cost matrix for matching spines in timepoint 1 and timepoint 2
    based on weighted combination of GIoU and depth difference.
    """
    len_t1 = len(medboxes_t1)
    len_t2 = len(medboxes_t2)

    cost_matrix = np.ones((len_t1, len_t2)) * 10

    ids_t1 = sorted(medboxes_t1.keys())
    ids_t2 = sorted(medboxes_t2.keys())

    for id1 in ids_t1:
        for id2 in ids_t2:
            cost = cost_of_match(
                medboxes_t1[id1],
                medboxes_t2[id2],
                zmin_max_t1[id1],
                zmin_max_t2[id2],
                lambdas
            )
            cost_matrix[ids_t1.index(id1), ids_t2.index(id2)] = cost
    
    return cost_matrix

def match_from_cost_matrix(
        cost_matrix: np.ndarray,
        ids_t1: np.ndarray,
        ids_t2: np.ndarray,
        thresh: float
        ) -> dict:
    unmatched_ids_t2 = set(ids_t2)
    matches = {}
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    next_new_id = max(ids_t1) + 1

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] <= thresh:
            matches[ids_t2[c]] = ids_t1[r]
            unmatched_ids_t2.remove(ids_t2[c])

    for uid2 in unmatched_ids_t2:
        matches[uid2] = next_new_id
        next_new_id += 1

    return matches

def run_track_in_time(
        fp_t1: str, 
        fp_t2: str, 
        mip_t1_fp: str, 
        mip_t2_fp: str, 
        corres_dir: str, 
        img_dir: str, 
        out_dir: str,
        giou_cost: float = 1.5,
        depth_cost: float = 0.3,
        thresh: float = 0.5
        ):
    os.makedirs(out_dir, exist_ok=True)
    
    lambdas = {
        'giou_cost': giou_cost,
        'depth_cost': depth_cost
    }
    # assert os.path.exists(corres_fp), f"Correspondence file not found: {corres_fp}"
    matcher = SpinesVolumeMatcher(
        fp_t1,
        fp_t2,
        mip_t1_fp,
        mip_t2_fp,
        corres_dir,
        img_dir,
        out_dir,
        lambdas,
        thresh
    )

    df_t2 = matcher.time_track()

    out_fp = os.path.join(out_dir, os.path.basename(fp_t2))
    df_t2.to_csv(out_fp, index=False)

    out_dir_draw = os.path.join(out_dir, 'images')
    os.makedirs(out_dir_draw, exist_ok=True)

    matcher.draw_ids_in_tp(tp2=False, out_dir=out_dir_draw)
    matcher.draw_ids_in_tp(tp2=True, out_dir=out_dir_draw)

    oof_t1_fixed, oob_t2_fixed = matcher.get_out_of_bounds_info()
    # TODO: save this information

def run_track_in_time_args(args):
    run_track_in_time(
        args.input_t1,
        args.input_t2,
        args.mip_t1,
        args.mip_t2,
        args.corres_dir,
        args.img_dir,
        args.out_dir
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Match spines between two consecutive timepoints")
    parser.add_argument("--input_t1", type=str, required=True,
        help="Filepath to the input data at first timepoint.")
    parser.add_argument("--input_t2", type=str, required=True,
        help="Filepath to the input data at second timepoint.")
    parser.add_argument("--mip_t1", type=str, required=True,
            help="Filepath to the MIP image of volume at first timepoint.")
    parser.add_argument("--mip_t2", type=str, required=True,
            help="Filepath to the MIP image of volume at second timepoint.")
    parser.add_argument("--corres_dir", type=str, required=True,
        help="Directory containing the dense correspondences between timepoints.")
    parser.add_argument("--img_dir", type=str, required=True,
        help="Directory containing the images (previously aligned within volume).")
    parser.add_argument("--out_dir", type=str, required=True,
        help="Output directory for the time-tracked results.")
    # TODO: add giou_cost, depth_cost, thresh
    
    args = parser.parse_args()

    run_track_in_time(
        args.input_t1,
        args.input_t2,
        args.mip_t1,
        args.mip_t2,
        args.corres_dir,
        args.img_dir,
        args.out_dir
    )
    