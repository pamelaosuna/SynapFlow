import os
from glob import glob
import argparse
from typing import List, Union

import math
import numpy as np
import cv2
import pandas as pd

class SpineSizeEstimator3D:
    def __init__(self, imgs, bboxes, params, norm=True):
        self.imgs = imgs
        self.bboxes = bboxes
        self.params = params
        self.norm = norm

        self.operator_dispatcher = {
            'mean': np.mean,
            'median': np.median,
            'max': np.max,
            'min': np.min
        }
    
    def comp_integ_fluo_3d(self, operator_3d):
        assert operator_3d in self.operator_dispatcher, \
            f'Operator {operator_3d} not recognized. Choose from {self.operator_dispatcher.keys()}'

        ifluos = []
        for (img, bbox) in zip(self.imgs, self.bboxes):
            ifluo = SpineSizeEstimator2D(
                img, bbox, self.params, norm=self.norm,
                ).comp_integ_fluo_2d()
            ifluos.append(ifluo)

        return self.operator_dispatcher[operator_3d](ifluos)
    
class SpineSizeEstimator2D:
    def __init__(
            self, 
            img: np.ndarray, 
            bbox: List[int],
            params: dict,
            norm: bool = True, 
            ):
        self.bbox = bbox
        self.height, self.width = img.shape[:2]
        self.img = img
        self.params = params
        self.norm = norm

        self.bg_thresh = None
        self.norm_val = None
        self.masked_subreg = None

    def extend_bbox(self, bbox: List[int], margin: int) -> np.ndarray:
        xmin, ymin, xmax, ymax = bbox

        # define the coordinates of the bbox with margins
        xmin_ext = max(0, xmin - margin)
        xmax_ext = min(self.width, xmax + margin)
        ymin_ext = max(0, ymin - margin)
        ymax_ext = min(self.height, ymax + margin)

        return np.around(np.array([xmin_ext, ymin_ext, xmax_ext, ymax_ext])).astype(int)

    def set_background_thresh(self, val: Union[float, None] = None) -> None:
        """
        Removes background from an image. Background is defined as the pixels whose
        intensity is lower than a certain threshold. This threshold is calculated
        taking a bigger box but centered on the original one, and averaging the 10% of
        the lowest pixels (that are not zero? or disregarding of this?).
        """
        if val:
            self.bg_thresh = val
            return
        
        xmin_ext, ymin_ext, xmax_ext, ymax_ext = self.extend_bbox(self.bbox, self.params['margin_bg'])

        # extract the subregion from the image
        subreg_ext = self.img[ymin_ext:ymax_ext, xmin_ext:xmax_ext]

        # find the 10% pixels that have the lowest value and average them
        darkest_pixels = np.sort(subreg_ext.ravel())[:int(subreg_ext.size*self.params['perc_bg'])]
        bg_thresh = darkest_pixels.mean()
        self.bg_thresh = bg_thresh
    
    def get_background_thresh(self) -> Union[float, None]:
        """
        Returns the background threshold.
        """
        return self.bg_thresh

    def get_norm_val(self) -> Union[float, None]:
        """
        Returns the normalization value. - approximation of the dendrite brightness
        """
        return self.norm_val
    
    def get_masked_subreg(self) -> np.ndarray:
        return self.masked_subreg
    
    def remove_background(self) -> np.ndarray:
        xmin, ymin, xmax, ymax = self.bbox

        orig_subreg = self.img[ymin:ymax, xmin:xmax]
        orig_subreg = orig_subreg.astype(np.float32)
        orig_subreg[orig_subreg < self.bg_thresh] = 0

        return orig_subreg
    
    def remove_bg_from_foreground_px(self, ifluo: float, masked_subreg: np.ndarray) -> float:
        n_foreground_px = np.sum(masked_subreg > self.bg_thresh)
        if n_foreground_px == 0:
            return 0
        else:
            return ifluo - (self.bg_thresh * n_foreground_px)

    def set_norm_val(self, val: Union[float, None] = None) -> None:
        """
        Normalizes brightness with respect to the mean brightness of the 10% brightest pixels of the bbox surroundings (in x, y and z directions).
        """
        if val:
            self.norm_val = val
            return
        
        xmin_ext, ymin_ext, xmax_ext, ymax_ext = self.extend_bbox(self.bbox, self.params['margin_dend']) # extended bounding box
    

        subreg_ext = self.img[ymin_ext:ymax_ext, xmin_ext:xmax_ext] # extract the subregion from the image
        
        norm_bbox = np.sort(subreg_ext.ravel())[round(subreg_ext.ravel().size*self.params['perc_dend']):].mean()
        
        if norm_bbox == 0:
            print('WARNING: norm_value = 0') # for image {im_name}')
            norm_bbox = 1

        self.norm_val = norm_bbox

    def weight_fluo_with_2d_gaussian(self, masked_subreg: np.ndarray) -> np.ndarray:
        """
        Circunscribes a non-isotropic 2D Gaussian according to the bounding box, and weights the
        fluorescence of the pixels according to their closeness to the center of the
        Gaussian.
        """
        xmin, ymin, xmax, ymax = self.bbox
        width = xmax - xmin
        height = ymax - ymin
        cx, cy = width // 2, height // 2

        x = np.arange(width)
        y = np.arange(height)
        X, Y = np.meshgrid(x, y)

        sx = (xmax - xmin) * self.params['sigma']
        sy = (ymax - ymin) * self.params['sigma']

        gaussian_mask = np.exp(-(((X - cx) ** 2) / (2 * sx ** 2) + ((Y - cy) ** 2) / (2 * sy ** 2)))

        return gaussian_mask*masked_subreg

    def comp_integ_fluo_2d(self) -> float:
        """
        Computes the integrated fluorescence of the specified image subregion.
        Here, integrated fluorescence is defined as the sum of pixel intensities in the given region.
        """
        if not self.bg_thresh:
            self.set_background_thresh()

        masked_subreg = self.remove_background()

        assert masked_subreg.shape == (self.bbox[3] - self.bbox[1], self.bbox[2] - self.bbox[0]), \
            f'Shape of masked subregion {masked_subreg.shape} does not match bbox {self.bbox}'

        self.masked_subreg = masked_subreg

        if self.params['sigma'] != 1:
            masked_subreg = self.weight_fluo_with_2d_gaussian(masked_subreg)

        ifluo = np.sum(masked_subreg.ravel())
        ifluo = self.remove_bg_from_foreground_px(ifluo, masked_subreg)
        
        if self.norm:
            if not self.norm_val:
                self.set_norm_val()
            normed_ifluo = ifluo/self.norm_val
        else:
            normed_ifluo = ifluo

        return normed_ifluo

def comp_2D_sizes(preds_filepath: str, img_dir: str, params: dict) -> pd.DataFrame:
    try:
        df = pd.read_csv(preds_filepath)
    except pd.errors.EmptyDataError:
        cols = ['filename','width','height','class','score','xmin','ymin','xmax','ymax','id']
        df = pd.DataFrame(columns=cols)

    sizes_2d = [] # integrated fluorescence values

    for i, fn in enumerate(df['filename'].values):
        img_fp = os.path.join(img_dir, os.path.basename(fn).replace('.csv', '.png')) # script for STED analysis
        img = cv2.imread(img_fp, cv2.IMREAD_GRAYSCALE)
        bbox = np.around(df[['xmin', 'ymin', 'xmax', 'ymax']].values[i]).astype(int)

        size_estimator = SpineSizeEstimator2D(img, bbox, params)
        
        ifluo = size_estimator.comp_integ_fluo_2d()

        sizes_2d.append(ifluo)
        
    # if empty df, still make sure size column is added
    if df.empty:
        df['size'] = np.nan
    else:
        df['size'] = sizes_2d        
    
    # remove rows with NaN values in 'size'
    df = df.dropna(subset=['size']).reset_index(drop=True)
    
    return df

def median_of_inliers(vals_2d: np.array, n_std: int = 1) -> float:
    """
    Computes the median of the inliers of spine sizes.
    Inliers are defined as the values within N standard deviations from the mean.
    """
    assert len(vals_2d) > 0, 'No sizes to compute median of inliers'

    b1 = np.mean(vals_2d) - n_std * np.std(vals_2d)
    b2 = np.mean(vals_2d) + n_std * np.std(vals_2d)

    inliers = vals_2d[(vals_2d >= b1) & (vals_2d <= b2)]
    
    return np.median(inliers)

def comp_3D_sizes(sizes2d_filepath: str, operator_3d: str) -> pd.DataFrame:
    operator_dispatcher = {
            'mean': np.mean,
            'median': np.median,
            'max': np.max,
            'median_of_inliers': median_of_inliers,
    }

    df = pd.read_csv(sizes2d_filepath)
    ids = df['id'].unique()

    for spine_id in ids:
        idx_of_interest = df[df['id'] == spine_id].index
        sizes_2d = df['size'].iloc[idx_of_interest].values

        ifluo = operator_dispatcher[operator_3d](sizes_2d)

        assert ifluo != np.nan and ifluo != math.nan, 'Nan value for size'
        df.loc[idx_of_interest, 'size'] = ifluo

    return df

def run_size_computation(
        input_dir, img_dir, out_dir, operator_3d, margin_bg, margin_dend, perc_bg, perc_dend, sigma
        ):
    # Computation in 2D
    preds_files = sorted(glob(os.path.join(input_dir, "*.csv")))

    out_dir_2d = os.path.join(out_dir, '2D')
    os.makedirs(out_dir_2d, exist_ok=True)

    for i, preds_f in enumerate(preds_files):
        fn = os.path.basename(preds_f)
        out_fp = os.path.join(out_dir_2d, fn) # operator_3d,
        if os.path.exists(out_fp):
            print(f'File {fn} already exists. Skipping...')
            continue
        
        params = {
            'margin_bg': margin_bg,
            'margin_dend': margin_dend,
            'perc_bg': perc_bg,
            'perc_dend': perc_dend,
            'sigma': sigma
        }
        sizes2d_df = comp_2D_sizes(preds_f, img_dir, params)
        sizes2d_df.to_csv(out_fp, index=False)

    # Computation in 3D using the 2D sizes and the 3D-operator
    sizes2d_files = sorted(glob(os.path.join(out_dir_2d, "*.csv")))
    out_dir_3d = os.path.join(out_dir_2d.replace('2D', '3D'), operator_3d)
    os.makedirs(out_dir_3d, exist_ok=True)

    for i, sizes_f in enumerate(sizes2d_files):
        fn = os.path.basename(sizes_f)
        out_fp = os.path.join(out_dir_3d, fn)
        if os.path.exists(out_fp):
            print(f'File {fn} already exists. Skipping...')
            continue
    
        sizes3d_df = comp_3D_sizes(sizes_f, operator_3d)
        sizes3d_df.to_csv(out_fp, index=False)

def run_size_computation_args(args):
    run_size_computation(
        args.input_dir,
        args.img_dir,
        args.out_dir_2d,
        args.operator_3d,
        args.margin_bg,
        args.margin_dend,
        args.perc_bg,
        args.perc_dend,
        args.sigma
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Compute the sizes of spines in 2D and 3D"
        )
    parser.add_argument("--img_dir", type=str, required=True, 
        help="Directory containing the images (previously registered within volume)")
    parser.add_argument("--out_dir_2d", type=str, required=True,
        help="Output directory for the 2D computed sizes")
    parser.add_argument("--input_dir", type=str, required=True,
        help="Directory containing the time-tracked predictions")
    parser.add_argument("--operator_3d", type=str, required=True, choices=['mean', 'median', 'max', 'median_of_inliers'],
        help="Operator to use for the 3D computation")
    parser.add_argument("--sigma", type=float, default=0.3,
        help="Sigma for the Gaussian weighting. Default: 0.3. If 1, no weighting is applied.")
    parser.add_argument("--margin_bg", type=int, default=25,
        help="Margin for background estimation. Default: 25")
    parser.add_argument("--margin_dend", type=int, default=25,
        help="Margin for dendrite intensity estimation. Default: 25")
    parser.add_argument("--perc_bg", type=float, default=0.2,
        help="Percentage of darkest pixels to consider for background estimation. Default: 0.2")
    parser.add_argument("--perc_dend", type=float, default=0.99,
        help="Percentage of brightest pixels to consider for dendrite intensity estimation. Default: 0.99")
    
    args = parser.parse_args()

    run_size_computation_args(args)

# TODO: test