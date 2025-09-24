import argparse
import os
from glob import glob

import numpy as np
import pandas as pd
from skimage import io
from skimage.filters import (
     threshold_otsu,
     threshold_multiotsu
)
from skimage.morphology import binary_dilation
import cv2

from SynapFlow.compute_size import median_of_inliers

OP_DISPATCHER = {
    'mean': np.mean,
    'median': np.median,
    'max': np.max,
    'median_of_inliers': median_of_inliers}

class SpineMorphologyEstimator:
    def __init__(
            self, 
            img: np.ndarray, 
            bbox: np.ndarray, 
            margin: int = 10, 
            kernel_size: int = 3,
            max_spine_len: int = 300
            ): 
        self.img = img
        self.bbox = bbox
        self.margin = margin # for bbox extension
        self.kernel_size = kernel_size # for dilation
        self.max_spine_len = max_spine_len

        self.img_width = img.shape[1]
        self.img_height = img.shape[0]

        self.bbox_ext = self.extend_bbox()
        self.img_spine = self.img[
            int(self.bbox[1]):int(self.bbox[3]),
            int(self.bbox[0]):int(self.bbox[2])
        ]
        self.img_spine_ext = self.img[
            int(self.bbox_ext[1]):int(self.bbox_ext[3]),
            int(self.bbox_ext[0]):int(self.bbox_ext[2])
        ]

        self.thresh = None # for binarization
        self.n_classes_otsu = 2

        self.head_center = np.array([np.nan]*2)
        self.junction_center = np.array([np.nan]*2)

    def extend_bbox(self):
        xmin, ymin, xmax, ymax = self.bbox

        # define the coordinates of the bbox with margins
        bbox_ext = np.array([
            xmin - self.margin, 
            ymin - self.margin, 
            xmax + self.margin, 
            ymax + self.margin]
            )
        
        # Clip to image boundaries
        bbox_ext = np.clip(
            bbox_ext, [0, 0, 0, 0], [self.img_width, self.img_height, self.img_width, self.img_height]).astype(int)

        return bbox_ext
    
    def get_junction_center(self):
        return self.junction_center
    
    def get_head_center(self):
        return self.head_center
    
    def find_junction(self, thresh: float):
        binary_ext, binary = binarize_spine_subreg(
            self.img, self.img_spine, self.bbox_ext, thresh
        )

        # Morphological dilation to find the dendritic shaft
        spine_mask, dend_mask = overlay_spine_and_dendrite(
            self.img, self.bbox, self.bbox_ext, binary_ext, binary, kernel_size=self.kernel_size
        )

        overlap = spine_mask & dend_mask
        if np.sum(overlap) == 0:
            return
        
        junction_center = np.where(overlap)
        cx = int(np.median(junction_center[1])) + self.bbox_ext[0]
        cy = int(np.median(junction_center[0])) + self.bbox_ext[1]

        self.junction_center = np.array([cx, cy])
    
    def find_spine_head(self, thresh: float):
        largest_contour = find_largest_contour(
            self.img_spine, thresh
        )
        if largest_contour is not None:
            return
        
        ellipse = cv2.fitEllipse(largest_contour)
        if np.any(np.isnan(ellipse[1])) or np.any(np.isnan(ellipse[0])):
            return
        
        bbox_width = self.bbox[2] - self.bbox[0]
        bbox_height = self.bbox[3] - self.bbox[1]

        # Start and end of minor and major axes
        ellipse_width = ellipse[1][0]
        ellipse_height = ellipse[1][1]

        cx, cy = ellipse[0]
        angle_rad = np.deg2rad(ellipse[2])

        major_axis_start = (
            int(cx - (ellipse_width / 2) * np.cos(angle_rad)),
            int(cy - (ellipse_width / 2) * np.sin(angle_rad))
        )
        major_axis_end = (
            int(cx + (ellipse_width / 2) * np.cos(angle_rad)),
            int(cy + (ellipse_width / 2) * np.sin(angle_rad))
        )
        minor_axis_start = (
            int(cx - (ellipse_height / 2) * np.sin(angle_rad)),
            int(cy + (ellipse_height / 2) * np.cos(angle_rad))
        )
        minor_axis_end = (
            int(cx + (ellipse_height / 2) * np.sin(angle_rad)),
            int(cy - (ellipse_height / 2) * np.cos(angle_rad))
        )

        # Clip the coordinates to the bounding box boundaries
        major_axis_start = tuple(
            np.clip(major_axis_start, 0, [bbox_width, bbox_height]))
        major_axis_end = tuple(
            np.clip(major_axis_end,   0, [bbox_width, bbox_height]))
        minor_axis_start = tuple(
            np.clip(minor_axis_start, 0, [bbox_width, bbox_height]))
        minor_axis_end = tuple(
            np.clip(minor_axis_end,   0, [bbox_width, bbox_height]))
        
        # Compute ellipse center in the original image coordinates
        cx = (major_axis_start[0] + major_axis_end[0]) // 2 + self.bbox[0]
        cy = (major_axis_start[1] + major_axis_end[1]) // 2 + self.bbox[1]

        ellipse_width = np.linalg.norm(np.array(major_axis_start) - np.array(major_axis_end))
        ellipse_height = np.linalg.norm(np.array(minor_axis_start) - np.array(minor_axis_end))

        if ellipse_width < 1 or \
            ellipse_height < 1 or \
            ellipse_width > self.max_spine_len or \
            ellipse_height > self.max_spine_len:
            return
        
        self.head_center = np.array([cx, cy])

def overlay_spine_and_dendrite(
        img: np.ndarray, 
        bbox: np.ndarray, 
        bbox_ext: np.ndarray, 
        binary_ext: np.ndarray, 
        binary: np.ndarray, 
        kernel_size: int = 3
        ):

    # Prepare binary mask for spine region
    binary_spine = np.zeros_like(img, dtype=np.uint8)
    binary_spine[bbox[1]:bbox[3], bbox[0]:bbox[2]] = binary.astype(np.uint8)

    # Dilate spine region within extended box
    footprint = np.ones((kernel_size, kernel_size))
    spine_dilated = binary_dilation(
        binary_spine[bbox_ext[1]:bbox_ext[3], bbox_ext[0]:bbox_ext[2]], footprint=footprint)

    # Binary mask for dendrite region
    dendrite_mask = binary_ext.astype(np.uint8)

    # Overlay masks: 0=background, 1=spine, 2=dendrite, 3=overlap
    overlay = dendrite_mask * 2 + spine_dilated.astype(np.uint8)

    # Masks for spine and dendrite
    mask_spine = (overlay == 1) | (overlay == 3)
    mask_dendrite = (overlay == 2) | (overlay == 3)

    return mask_spine.astype(np.uint8), mask_dendrite.astype(np.uint8)

def binarize_spine_subreg(
        img: np.ndarray,
        img_spine: np.ndarray,
        bbox_ext: np.ndarray,
        thresh: float
        ):
    binary = img_spine > thresh
    binary_ext = img[
        int(bbox_ext[1]):int(bbox_ext[3]),
        int(bbox_ext[0]):int(bbox_ext[2])
    ] > thresh
    return binary_ext, binary

def find_largest_contour(
        img_spine: np.ndarray, thresh: float):
    binary = img_spine > thresh
    contours, _ = cv2.findContours(
        binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Largest contour inside the spine patch is assumed to be the spine head
    largest_contour = max(contours, key=cv2.contourArea)

    # Check that there are enough points to fit an ellipse
    if len(largest_contour) < 5:
        return None
    
    return largest_contour
    
def estimate_head2dend(
        dets_file: str, img_dir: str, max_classes_otsu: int = 5) -> pd.DataFrame:
    df = pd.read_csv(dets_file)
    feat_labels = ['spine_to_dendrite_distance']

    if df.empty:
        # add new column names and return
        df = pd.DataFrame(
             columns=df.columns.tolist() + feat_labels)
        return df
    feats = {k: [] for k in feat_labels}

    for _, row in df.iterrows():
        filename = row['filename']

        bbox = row[['xmin', 'ymin', 'xmax', 'ymax']].values
        img = io.imread(os.path.join(img_dir, filename.replace('.csv', '.png')))

        # Make sure image is grayscale
        img = img[:,:,0] if img.ndim == 3 else img

        img_spine = img[
            int(bbox[1]):int(bbox[3]), 
            int(bbox[0]):int(bbox[2])
            ]
        thresh = threshold_otsu(img_spine) # for binarization

        feat_estimator = SpineMorphologyEstimator(img, bbox)

        for n_classes_otsu in range(2, max_classes_otsu):
            thresh = threshold_multiotsu(img_spine, classes=n_classes_otsu)[0]
            feat_estimator.find_junction(thresh)
            if not np.any(np.isnan(feat_estimator.get_junction_center())):
                break

        junction = feat_estimator.get_junction_center()
        if np.any(np.isnan(junction)):
            feats['spine_to_dendrite_distance'].append(np.nan)
            continue

        for n_classes_otsu in range(2, max_classes_otsu):
            thresh = threshold_multiotsu(img_spine, classes=n_classes_otsu)[0]
            feat_estimator.find_spine_head(thresh)
            if not np.any(np.isnan(feat_estimator.get_head_center())):
                break

        spine_head = feat_estimator.get_head_center()
        if np.any(np.isnan(spine_head)):
            feats['spine_to_dendrite_distance'].append(np.nan)
            continue

        # Compute distance 
        feats['spine_to_dendrite_distance'].append(
            np.linalg.norm(spine_head - junction)
        )

    for k in feat_labels:
        df[k] = feats[k]
        
    return df

def integrate_2d_to_3d(filepath: str, operator_3d: str = 'median'):
    operator_dispatcher = {
        'mean': np.mean,
        'median': np.median,
        'max': np.max,
        'median_of_inliers': median_of_inliers
    }
    df = pd.read_csv(filepath)
    ids = df['id'].unique()

    for spine_id in ids:
        ind = df[df['id'] == spine_id].index
        head2dend_2d = df['spine_to_dendrite_distance'].iloc[ind].values

        head2dend_3d = operator_dispatcher[operator_3d](head2dend_2d)
        df.loc[ind, 'spine_to_dendrite_distance'] = head2dend_3d

    return df

def run_head2dend_estimation(
        input_dir: str, img_dir: str, out_dir: str, operator_3d: str
        ) -> None:
        out_dir_2d = os.path.join(out_dir, '2D')
        os.makedirs(out_dir_2d, exist_ok=True)

        files = sorted(glob(os.path.join(input_dir, '*.csv')))
        for f in files:
            out_fp = os.path.join(out_dir_2d, os.path.basename(f))

            if os.path.exists(out_fp):
                print(f'>>> File {out_fp} already exists, skipping...')
                continue

            head2dend_df = estimate_head2dend(f, img_dir)
            head2dend_df.to_csv(out_fp, index=False)
        
        # Now integrate 2D information into 3D
        files_2d = sorted(glob(os.path.join(out_dir_2d, '*.csv')))
        out_dir_3d = os.path.join(out_dir, '3D')
        os.makedirs(out_dir_3d, exist_ok=True)

        for f in files_2d:
            out_fp = os.path.join(out_dir_3d, os.path.basename(f))
            if os.path.exists(out_fp):
                print(f'>>> File {out_fp} already exists, skipping...')
                continue
            head2dend_df = integrate_2d_to_3d(f, operator_3d)
            head2dend_df.to_csv(out_fp, index=False)

def run_head2dend_estimation_args(args):
    run_head2dend_estimation(
        args.input_dir,
        args.img_dir,
        args.out_dir,
        args.operator_3d
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Estimate spine head-to-dendrite distance.'
    )
    parser.add_argument('--input_dir', type=str, required=True,
        help='Input directory with spines with identities resolved in 3D (i.e. at least depth-tracked)')
    parser.add_argument('--img_dir', type=str, required=True,
        help='Directory with images used for detection')
    parser.add_argument('--out_dir', type=str, required=True,
        help='Output directory to save the csv files with spine-to-dendrite distances')
    parser.add_argument('--operator_3d', type=str, default='median', 
        choices=['mean', 'median', 'max', 'median_of_inliers'],
        help='Operator to use for aggregating the 2D distances into a single value per spine. Default: median')
    args = parser.parse_args()

    run_head2dend_estimation(
        args.input_dir,
        args.img_dir,
        args.out_dir,
        args.operator_3d
    )