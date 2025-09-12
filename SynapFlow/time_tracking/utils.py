import numpy as np
import pandas as pd

from skimage.exposure import rescale_intensity
from skimage.transform import resize
from scipy.ndimage import shift

def comp_best_shift_z(
        fixed_vol: np.ndarray,
        moving_vol: np.ndarray,
        downs: float = 1.,
        max_abs_shift: int = 100
        ) -> int:
    """
    Compute the best shift in z-axis (depth) to align two volumes based on
    maximizing the mean 2D cross-correlation (across depth).
    """
    orig_h, orig_w = fixed_vol.shape[1], fixed_vol.shape[2]

    # Downsample volumes (only in xy) for faster computation
    if downs != 1.:
        fixed_vol = resize(
            fixed_vol, 
            (fixed_vol.shape[0], int(orig_h/downs), int(orig_w/downs)),
            anti_aliasing=True,
            preserve_range=True
            )
        moving_vol = resize(
            moving_vol,
            (moving_vol.shape[0], int(orig_h/downs), int(orig_w/downs)),
            anti_aliasing=True,
            preserve_range=True
            )

        fixed_vol = rescale_intensity(
            fixed_vol,
            out_range = (0, 255)
        )
        moving_vol = rescale_intensity(
            moving_vol,
            out_range = (0, 255)
        )
    
    corr_scores = []
    shift_range = range(-max_abs_shift, max_abs_shift + 1)

    for shift_z in shift_range:
        moving_vol_shifted = shift(
            moving_vol,
            shift=(shift_z, 0, 0),
            order=1,
            mode='nearest'
        )
        min_layers = min(fixed_vol.shape[0], moving_vol_shifted.shape[0])
        corr_per_layer = [
            np.corrcoef(
                fixed_vol[i].flatten(), 
                moving_vol_shifted[i].flatten()
                )[0, 1] if np.std(moving_vol_shifted[i]) > 0 else 0
            for i in range(min_layers)
        ]
        mean_corr = np.mean(corr_per_layer)
        corr_scores.append(mean_corr)
    
    best_shift_z = shift_range[np.argmax(corr_scores)]
    return best_shift_z


def corres_box(box: np.ndarray, corres: np.ndarray, shape: "tuple[int, int]") -> "list[int]":
    """
    Given a bounding box and dense correspondences, compute the new bounding box
    after applying the correspondences.
    """
    box = np.round(box).astype(int)
    box = np.clip(box, [0, 0, 0, 0], [shape[1]-1, shape[0]-1, shape[1]-1, shape[0]-1])
    xmin, ymin, xmax, ymax = box

    new_xmin, new_ymin = corres[ymin*shape[1] + xmin, 2:4]
    new_xmax, new_ymax = corres[ymax*shape[1] + xmax, 2:4]

    # Check if out-of-bounds
    if new_xmin < 0 or new_ymin < 0 or new_xmax >= shape[1] or new_ymax >= shape[0]:
        return [-1]*4
    
    # Most likely incorrect correspondence
    if new_xmin >= new_xmax or new_ymin >= new_ymax:
        return [-1]*4
    
    new_box = [new_xmin, new_ymin, new_xmax, new_ymax]

    return new_box

def correspondences_in_xy(
        df_t1: pd.DataFrame,
        df_t2: pd.DataFrame,
        corres_t1_to_t2: np.ndarray,
        corres_t2_to_t1: np.ndarray,
        img_shape: "tuple[int, int]"
    ):
    """
    Project boxes of a timepoint into the other, and keep record of boxes 
    that are out-of-bounds after applying correspondences.
    """
    ids_t1 = df_t1['id'].unique()
    ids_t2 = df_t2['id'].unique()

    out_of_bounds_t2_fixed = set()
    out_of_bounds_t1_fixed = set()

    median_boxes_t1 = {}
    median_boxes_t2 = {}

    for id1 in ids_t1:
        df_id1 = df_t1[df_t1['id'] == id1]
        boxes_id1 = df_id1[['xmin', 'ymin', 'xmax', 'ymax']].values

        # Project boxes of t1 into t2 using dense correspondences
        
        expected_boxes_id1 = np.asarray([
            corres_box(
                box,
                corres_t1_to_t2,
                img_shape)
                if box[0] >= 0 else [-1]*4
            for box in boxes_id1
            ])
        
        # Check if the number of boxes was initially > 0 but all are out-of-bounds
        len_boxes_id1 = len(boxes_id1)

        # Remove boxes that are out-of-bounds in the next timepoint
        expected_boxes_id1 = expected_boxes_id1[
            (expected_boxes_id1 != -1).all(axis=1)
            ]
        
        if len_boxes_id1 > 0 and len(expected_boxes_id1) == 0:
            # Add to out-of-bounds record
            out_of_bounds_t2_fixed.add(id1)
        
        if len(expected_boxes_id1) > 0:
            median_box_id1 = np.median(expected_boxes_id1, axis=0)
            median_boxes_t1[id1] = median_box_id1
        else:
            median_boxes_t1[id1] = [-1]*4
        
    for id2 in ids_t2:
        df_id2 = df_t2[df_t2['id'] == id2]
        boxes_id2 = df_id2[['xmin', 'ymin', 'xmax', 'ymax']].values

        # Check if object is out of bounds (in xy) in the previous timepoint
        expected_boxes_id2 = np.asarray([
            corres_box(
                box,
                corres_t2_to_t1,
                img_shape)
                if box[0] >= 0 else [-1]*4
            for box in boxes_id2
            ])
        
        len_boxes_id2 = len(boxes_id2)
        expected_boxes_id2 = expected_boxes_id2[
            (expected_boxes_id2 != -1).all(axis=1)
            ]
        
        if len_boxes_id2 > 0 and len(expected_boxes_id2) == 0:
            out_of_bounds_t1_fixed.add(id2)

        if len(expected_boxes_id2) > 0:
            median_box_id2 = np.median(expected_boxes_id2, axis=0)
            median_boxes_t2[id2] = median_box_id2
        else:
            median_boxes_t2[id2] = [-1]*4

    return median_boxes_t1, median_boxes_t2, out_of_bounds_t1_fixed, out_of_bounds_t2_fixed


def correspondences_in_z(z_shift, df_t1, df_t2, max_layer_t1, max_layer_t2):
    ids_t1 = df_t1['id'].unique()
    ids_t2 = df_t2['id'].unique()

    depth_min_max_t1 = {}
    depth_min_max_t2 = {}

    out_of_bounds_t2_fixed = set()
    out_of_bounds_t1_fixed = set()

    for id1 in ids_t1:
        filenames_id1 = df_t1[df_t1['id'] == id1]['filename']

        zrange_id1 = [int(f.split('_layer')[1].split('.png')[0]) for f in filenames_id1]
        zmin1, zmax1 = min(zrange_id1), max(zrange_id1)

        # Check if object is out-of-bounds (in depth) after applying z_shift
        if (zmax1 < z_shift) or (zmin1 + z_shift > max_layer_t2):
            out_of_bounds_t2_fixed.add(id1)
        
        depth_min_max_t1[id1] = (zmin1, zmax1)

    for id2 in ids_t2:
        filenames_id2 = df_t2[df_t2['id'] == id2]['filename']

        zrange2 = [int(f.split('_layer')[1].split('.png')[0]) for f in filenames_id2]
        zmin2, zmax2 = min(zrange2) + z_shift, max(zrange2) + z_shift

        # Check if object is out-of-bounds (in depth) after applying -z_shift
        if zmax2 < 0 or zmin2 > max_layer_t1:
            out_of_bounds_t1_fixed.add(id2)

        depth_min_max_t2[id2] = (zmin2, zmax2)

    return depth_min_max_t1, depth_min_max_t2, out_of_bounds_t1_fixed, out_of_bounds_t2_fixed